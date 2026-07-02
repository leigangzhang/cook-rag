import os
import sys
import logging
import hashlib
import re
from typing import Optional, List

from modules.build_index import BuildIndex
from config import DEFAULT_CONFIG, RAGConf
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class QueryRetrail:
    """查询检索和优化"""

    def __init__(self, vectorstore: FAISS, config: Optional[RAGConf] = None) -> None:
        self.config = config or DEFAULT_CONFIG
        self.vectorstore = vectorstore
    

    # 混合检索
    def hybrid_retrial(self, user_query: str, question_type: str, chunks: List[Document]) -> List[Document]:
        # 设置更大的Top K，用来做初筛；初筛后再做元数据过滤
        top_k = self.config.top_k * 2

        # 分别使用向量检索和BM25检索获取Top K
        vector_docs = self.verctor_retrial(user_query, question_type, top_k)
        bm25_docs = self.bm25_retrial(chunks, user_query, top_k)

        # 使用RRF重排取最终Top K
        reranked_docs = self.rrf_rerank(vector_docs, bm25_docs, top_k)
        return reranked_docs[:top_k]


    # 向量检索
    def verctor_retrial(self, user_query: str, question_type: str, top_k: int, threshold: Optional[float] = None) -> List[Document]:
        threshold = threshold or self.config.threshold

        if question_type == "list":
            vector_retriver = self.vectorstore.as_retriever(
                search_type="similarity", 
                search_kwargs={"k": top_k}
            )
        else:
            vector_retriver = self.vectorstore.as_retriever(
                search_type="similarity_score_threshold", 
                search_kwargs={"k": top_k, 
                               "score_threshold": threshold}
            )
        vector_chunks = vector_retriver.invoke(user_query)
        
                
        # 收集父文档名称和相关性信息用于日志
        parent_info = []
        parent_counter = {}
        for chunk in vector_chunks:
            dish_name = chunk.metadata.get('dish_name', '未知菜品')
            parent_counter[dish_name] = parent_counter.get(dish_name, 0) + 1

        for dish_name, relevance_count in parent_counter.items():
            parent_info.append(f"{dish_name}({relevance_count}块)")

        logger.info(f"使用向量检索器检索到 {len(vector_chunks)} 个相关文档块: {', '.join(parent_info)}")
        return vector_chunks


    # BM25检索
    def bm25_retrial(self, chunks:List[Document], user_query: str, top_k: int) -> List[Document]:
        bm25_retriver = BM25Retriever.from_documents(chunks, k=top_k)
        bm25_chunks = bm25_retriver.invoke(user_query)
        logger.info(f"使用BM25检索器检索到 {len(bm25_chunks)} 个相关文档块")

        return bm25_chunks


    # RRF重排混合
    def rrf_rerank(self, vector_docs: List[Document], bm25_docs: List[Document], k: int = 60) -> List[Document]:
        
        doc_scores = {}
        doc_objs = {}

        # 计算向量检索的RRF得分
        for rank, doc in enumerate(vector_docs):
            # 使用文档内容生成唯一确定的哈希标识
            doc_id = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
            doc_objs[doc_id] = doc

            rrf = 1.0 / (k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf
        
        # 计算BM25检索的RRF得分
        for rank, doc in enumerate(bm25_docs):
            # 使用文档内容生成唯一确定的哈希标识
            doc_id = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
            doc_objs[doc_id] = doc

            rrf = 1.0 / (k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf
        
        # 按最终的RRF得分排序
        sorted_docs = sorted(doc_scores.items(), key = lambda x : x[1], reverse=True)

        # 构建最终的排序结果
        reranked_docs = []
        for doc_id, rrf_score in sorted_docs:
            if doc_id in doc_objs:
                doc = doc_objs[doc_id]
                doc.metadata['rrf_score'] = rrf_score
                reranked_docs.append(doc)

        logger.info(f"RRF重排完成: 向量检索{len(vector_docs)}个文档, BM25检索{len(bm25_docs)}个文档, 合并后{len(reranked_docs)}个文档")

        return reranked_docs

    # 元数据过滤检索
    def metadata_filter_query(self, user_query: str, docs: List[Document], top_k: int = 5) -> List[Document]:
        """在向量检索和BM25检索的基础上做多维度元数据过滤。

        支持的维度: category(品类), difficulty(难度), cuisines(菜系),
                    methods(技法), max_carb_g(碳水上限), max_time_min(最长烹饪时间)
        对于 list 类型的 metadata 字段(如 cuisines/methods/ingredients)，使用子集匹配。
        """
        filters = {}

        # -- 品类过滤 --
        for cat in BuildIndex.get_supported_categories():
            if cat in user_query:
                filters['category'] = cat
                break

        # -- 难度过滤 --
        for diff in BuildIndex.get_supported_difficulties():
            if diff in user_query:
                filters['difficulty'] = diff
                break

        # -- 菜系过滤（归一化菜系名）--
        for cuisine in BuildIndex.get_supported_cuisine():
            if cuisine in user_query:
                filters['cuisines'] = cuisine
                break

        # -- 技法过滤 --
        for method in BuildIndex.get_supported_cooking_methods():
            if method in user_query:
                filters['methods'] = method
                break

        # -- 无糖/低糖（基于 carb_amount_g 和食材列表）--
        if any(kw in user_query for kw in ['无糖', '不加糖']):
            filters['max_carb_g'] = 5
        elif any(kw in user_query for kw in ['低糖', '糖尿病']):
            filters['max_carb_g'] = 20

        # -- 时间过滤 --
        time_match = re.search(
            r'(\d+)\s*(分钟|分)(?:\s*以[内下]|\s*之[内下]|\s*不[超到]|\s*以内|\s*以下)?',
            user_query,
        )
        if time_match:
            filters['max_time_min'] = int(time_match.group(1))
        elif '半小时' in user_query or '30分钟' in user_query:
            filters['max_time_min'] = 30
        elif '一小时' in user_query or '1小时' in user_query or '60分钟' in user_query:
            filters['max_time_min'] = 60
        elif '快速' in user_query or '快手' in user_query:
            filters['max_time_min'] = 15

        # -- 无 filter 时直接返回 --
        if not filters:
            return docs[:top_k]

        # -- 执行过滤 --
        logger.info(f"元数据过滤条件: {filters}")
        filtered_docs = []
        for doc in docs:
            if self._match_filters(doc, filters):
                filtered_docs.append(doc)
                if len(filtered_docs) >= top_k:
                    break

        logger.info(f"元数据过滤：{len(docs)} → {len(filtered_docs)} 个文档")
        return filtered_docs

    SUGAR_INGREDIENTS = {
        "糖", "白糖", "白砂糖", "冰糖", "红糖", "黄糖", "黑糖",
        "蔗糖", "蜂蜜", "糖浆", "麦芽糖", "果糖",
    }

    def _match_filters(self, doc: Document, filters: dict) -> bool:
        """检查单个文档是否满足所有过滤条件。

        支持四种匹配模式：
        - 单值匹配: difficulty, category
        - 列表子集匹配: cuisines / methods / ingredients
        - 数值比较: max_carb_g, max_time_min
        """
        metadata = doc.metadata

        for key, value in filters.items():
            meta_val = metadata.get(key)

            # -- 列表匹配（cuisines / methods / ingredients） --
            if key in ('cuisines', 'methods', 'ingredients'):
                if meta_val is None:
                    return False
                if isinstance(meta_val, list):
                    if value not in meta_val:
                        return False
                else:
                    if value != meta_val:
                        return False

            # -- 碳水过滤（无糖/低糖） --
            elif key == 'max_carb_g':
                carb_amount_g = metadata.get('carb_amount_g')
                # 若明确标注了碳水，优先按数值判断
                if carb_amount_g is not None:
                    if carb_amount_g > value:
                        return False
                # 未标注时，检查食材里是否含糖类原料
                ingredients = metadata.get('ingredients') or []
                if any(ing in self.SUGAR_INGREDIENTS for ing in ingredients):
                    return False

            # -- 时间数值 --
            elif key == 'max_time_min':
                cooking_time_min = metadata.get('cooking_time_min')
                if cooking_time_min is not None and cooking_time_min > value:
                    return False

            # -- 单值精确匹配 --
            else:
                if meta_val != value:
                    return False

        return True

    # 获取父文档
    def get_parent_documents(self, child_chunks: List[Document], documents: List[Document], top_k: int) -> List[Document]:
        
        parent_counter = {}
        parent_docs = {}

        # 通过子文本块去查询父文档并对其计数
        for child_chunk in child_chunks:
            parent_id = child_chunk.metadata.get("parent_id")
            parent_counter[parent_id] = parent_counter.get(parent_id, 0) + 1

            if parent_id not in parent_docs:
                for doc in documents:
                    if parent_id == doc.metadata.get("parent_id"):
                        parent_docs[parent_id] = doc
                        break
        
        # 按照计数从大到小排序
        sorted_counter = sorted(parent_counter.keys(), key=lambda x: parent_counter[x], reverse=True)

        # 构建去重后的父文档列表
        merged_parent_docs = []
        for parent_id in sorted_counter:
            if parent_id in parent_docs:
                merged_parent_docs.append(parent_docs[parent_id])

        # 收集父文档名称和相关性信息用于日志
        parent_info = []
        for doc in merged_parent_docs:
            dish_name = doc.metadata.get('dish_name', '未知菜品')
            parent_id = doc.metadata.get('parent_id')
            relevance_count = parent_counter.get(parent_id, 0)
            parent_info.append(f"{dish_name}({relevance_count}块)")

        logger.info(f"从 {len(child_chunks)} 个子文档中找到 {len(merged_parent_docs)} 个去重父文档: {', '.join(parent_info)}")

        return merged_parent_docs[:top_k]