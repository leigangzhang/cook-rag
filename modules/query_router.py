from collections import defaultdict
from pathlib import Path
import re
import json
import logging
from enum import Enum


from prompt import LLMPrompts
from config import RAGConf, DEFAULT_CONFIG
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models.moonshot import MoonshotChat

logger = logging.getLogger(__name__)


class SearchStrategy(Enum):
    """搜索策略枚举"""
    HYBRID_TRADITIONAL = "hybrid_traditional"  # 传统混合检索
    GRAPH_RAG = "graph_rag"  # 图RAG检索


class QueryRouter:

    def __init__(self) -> None:
        pass

    def query_analysis(self, llm: MoonshotChat, user_query: str):
        """用户查询复杂度分析(基于LLM)"""
        logger.info(f"分析查询特征: {user_query}")
        try:
            prompt = PromptTemplate.from_template(LLMPrompts.QUERY_ANALYSIS)
        
            chain = (
                {"query":RunnablePassthrough()}
                |prompt
                |llm
                |StrOutputParser()
            )

            response = chain.invoke(user_query)
            parsed = json.loads(response.strip("`").strip("json").strip())

            result = {
                'query_complexity': parsed.get('query_complexity', 0.5),
                'relationship_intensity': parsed.get('relationship_intensity', 0.5),
                'reasoning_required': parsed.get('reasoning_required', False),
                'entity_count': parsed.get('entity_count', 1),
                'recommended_strategy': parsed.get('recommended_strategy', 'hybrid_traditional'),
                'confidence': parsed.get('confidence', 0.5),
                'reasoning': parsed.get('reasoning', '默认分析')
            }

            logging.info(f"查询分析完成：{result['recommended_strategy']}(置信度: {result['confidence']:.2f})")
            return result

        except Exception as e:
            logger.error(f"分析查询特征失败：{e}")
            return {'recommended_strategy': 'hybrid_traditional'}




class GraphRetriever:
    """基于知识图谱的结构化菜谱检索器。

    将 LLM 解析出的维度条件（cuisines / difficulty / category /
    cooking_time_min / carb_amount_g / ingredients / methods / fast）
    映射为图节点和边的查询，取各维度交集后返回匹配的菜名列表。
    """

    def __init__(self):
        graph_path = DEFAULT_CONFIG.graph_json_path
        g = json.loads(Path(graph_path).read_text(encoding="utf-8"))
        self.nodes = g['nodes']
        self.links = g['links']
        self.node_by_id = {n['id']: n for n in self.nodes}

        # 加载菜谱元数据，建 source_file → metadata 快速查询表
        recipes_path = DEFAULT_CONFIG.recipes_metadata_path
        recipes = json.loads(Path(recipes_path).read_text(encoding="utf-8"))
        self.recipe_metadata = {r["dish_name"]: r for r in recipes}

        # 预建各类节点 ID 的快速查找映射
        self._build_maps()

    def _build_maps(self):
        """预建各类节点 ID 的快速查找映射。"""
        self.ingr_ids = {}
        self.cuisine_ids = {}
        self.method_ids = {}
        self.cat_ids = {}
        self.diff_ids = {}
        self.time_ids = {}
        self.carb_ids = {}

        for n in self.nodes:
            nid, label = n['id'], n.get('label', '')
            if nid.startswith('ingr__'):
                self.ingr_ids[label] = nid
            elif nid.startswith('cuisine___'):
                self.cuisine_ids[label] = nid
            elif nid.startswith('meth__'):
                self.method_ids[label] = nid
            elif nid.startswith('cat___'):
                self.cat_ids[label] = nid
            elif nid.startswith('diff__'):
                self.diff_ids[label] = nid
            elif nid.startswith('time___'):
                self.time_ids[label] = nid
            elif nid.startswith('carb__'):
                self.carb_ids[label] = nid

    def parse_user_query(self, llm: MoonshotChat, user_query: str) -> dict:
        """调用 LLM 将用户查询解析为结构化的图检索条件。

        返回 dict，key 名与 GRAPH_CONDITION_EXTRACT_PROMPT 维度一致：
        cuisines, difficulty, cooking_time_min, carb_amount_g,
        ingredients, methods, category, fast。
        """
        logger.info(f"开始解析用户查询为图检索条件")

        prompt = PromptTemplate.from_template(
            LLMPrompts.GRAPH_CONDITION_EXTRACT_PROMPT
        )
        chain = (
            {"query": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        raw = chain.invoke(user_query).strip('')

        try:
            conditions = json.loads(raw.strip("`").strip("json").strip())
        except json.JSONDecodeError as e:
            logger.warning(f"解析用户查询为图检索条件失败: {e}, raw={raw}")
            return {}

        # 归一化：确保 ingredients / methods 为 list 类型
        for key in ('ingredients', 'methods'):
            if key in conditions:
                val = conditions[key]
                if isinstance(val, str):
                    conditions[key] = [val]
                elif not isinstance(val, list):
                    conditions[key] = []

        logger.info(f"Graph conditions parsed: {conditions}")
        return conditions

    def structured_search(self, conditions: dict, top_k: int) -> list[str]:
        """使用结构化条件检索图，返回匹配的菜名列表。

        conditions 的 key 名与 GRAPH_CONDITION_EXTRACT_PROMPT 一致：
        cuisines, difficulty, cooking_time_min, carb_amount_g,
        ingredients, methods, category, fast。
        各维度之间取交集，未指定维度不做限制。
        """
        if not conditions:
            return []

        result_ids = None

        def _intersect(matched: set):
            nonlocal result_ids
            if result_ids is None:
                result_ids = matched
            else:
                result_ids &= matched

        # --- cuisines（支持字符串或列表，精确匹配） ---
        if 'cuisines' in conditions:
            cuisines = conditions['cuisines']
            if isinstance(cuisines, str):
                cuisines = [cuisines]
            matched = set()
            for cuisine in cuisines:
                cid = self.cuisine_ids.get(cuisine)
                if cid:
                    matched.update(e['source'] for e in self.links if e['target'] == cid)
            if matched:
                _intersect(matched)

        # --- difficulty（单值字符串，精确匹配） ---
        if 'difficulty' in conditions:
            did = self.diff_ids.get(conditions['difficulty'])
            if did:
                matched = {e['source'] for e in self.links if e['target'] == did}
                if matched:
                    _intersect(matched)

        # --- category（单值字符串，直接用中文 label 匹配 cat 节点） ---
        if 'category' in conditions:
            cat = conditions['category']
            for label, cid in self.cat_ids.items():
                if label == cat:
                    matched = {e['source'] for e in self.links if e['target'] == cid}
                    if matched:
                        _intersect(matched)
                    break

        # --- ingredients（list / str，逐元素模糊匹配，多元素间取交集） ---
        if 'ingredients' in conditions:
            ingrs = conditions['ingredients']
            if isinstance(ingrs, str):
                ingrs = [ingrs]
            for ing_kw in ingrs:
                matching_ids = {
                    nid for label, nid in self.ingr_ids.items()
                    if ing_kw in label
                }
                if not matching_ids:
                    continue
                matched = {
                    e['source'] for e in self.links
                    if e['target'] in matching_ids
                }
                if matched:
                    _intersect(matched)

        # --- methods（list / str，逐元素模糊匹配，多元素间取交集） ---
        if 'methods' in conditions:
            mtds = conditions['methods']
            if isinstance(mtds, str):
                mtds = [mtds]
            for mtd_kw in mtds:
                matching_ids = {
                    nid for label, nid in self.method_ids.items()
                    if mtd_kw in label
                }
                if not matching_ids:
                    continue
                matched = {
                    e['source'] for e in self.links
                    if e['target'] in matching_ids
                }
                if matched:
                    _intersect(matched)

        # --- cooking_time_min（int → 时间桶反查） ---
        if 'cooking_time_min' in conditions:
            t = conditions['cooking_time_min']
            need_buckets: list[str] = []
            if t <= 15:
                need_buckets = ['快速(<15分钟)']
            elif t <= 30:
                need_buckets = ['快速(<15分钟)', '中速(15-30分钟)']
            # t > 30 时无法通过桶精确过滤，跳过
            if need_buckets:
                matched = set()
                for label in need_buckets:
                    tid = self.time_ids.get(label)
                    if tid:
                        matched.update(
                            e['source'] for e in self.links
                            if e['target'] == tid
                        )
                if matched:
                    _intersect(matched)

        # --- carb_amount_g（int → carb 桶反查） ---
        if 'carb_amount_g' in conditions:
            v = conditions['carb_amount_g']
            if v <= 5:
                carb_labels = ['无糖(<5g)']
            elif v <= 20:
                carb_labels = ['无糖(<5g)', '低碳(5-20g)']
            else:
                carb_labels = None
            if carb_labels:
                matched = set()
                for cl in carb_labels:
                    cid = self.carb_ids.get(cl)
                    if cid:
                        matched.update(
                            e['source'] for e in self.links
                            if e['target'] == cid
                        )
                if matched:
                    _intersect(matched)

        # --- fast（bool → 快捷映射到 <15min 桶） ---
        if conditions.get('fast'):
            tid = self.time_ids.get('快速(<15分钟)')
            if tid:
                matched = {e['source'] for e in self.links if e['target'] == tid}
                if matched:
                    _intersect(matched)

        if result_ids is None:
            return []

        # 仅返回 document 类型的节点（即菜谱）
        return sorted({
            self.node_by_id[rid]['label']
            for rid in result_ids
            if self.node_by_id[rid].get('file_type') == 'document'
        })[:top_k]

    def graph_retrial(self, llm: MoonshotChat, user_query: str, top_k: int) -> list[Document]:
        """使用图检索复杂问题。

        流程：LLM 解析查询 → 图结构化检索 → 读取匹配菜品完整文档 → 附加元数据。
        返回 list[Document]，每个 Document.metadata 包含 dish_name、category 等字段。
        """
        conditions = self.parse_user_query(llm, user_query)
        if not conditions:
            logger.info("未从用户查询中提取到图检索条件，回退到空结果")
            return []

        dish_names = self.structured_search(conditions, top_k)
        logger.info(f"图检索匹配到 {len(dish_names)} 道菜品: {dish_names}")

        if not dish_names:
            return []

        # 从 source_file 读取整篇文档，并附加元数据
        results: list[Document] = []
        data_root = Path("data/cook/")
        for n in self.nodes:
            if n.get('file_type') != 'document':
                continue
            if n['label'] not in dish_names:
                continue
            source_file = n.get('source_file', '')
            if not source_file:
                continue
            filepath = data_root / source_file
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()

                # 从预加载的元数据查找表中获取这道菜的全部元数据
                meta = self.recipe_metadata.get(n["label"], {})
                doc = Document(
                    page_content=content,
                    metadata={
                        'dish_name': meta.get('dish_name', n['label']),
                        'category': meta.get('category', ''),
                        'difficulty': meta.get('difficulty', 3),
                        'ingredients': meta.get('ingredients', []),
                        'methods': meta.get('methods', []),
                        'cuisines': meta.get('cuisines', []),
                        'cooking_time_min': meta.get('cooking_time_min'),
                        'carb_amount_g': meta.get('carb_amount_g'),
                        'source_file': source_file,
                    },
                )
                results.append(doc)
            except Exception as e:
                logger.error(f"读取 {filepath} 失败: {e}")
                continue

        logger.info(f"图检索返回 {len(results)} 条完整菜品文档")

        return results[:top_k]
