import json
import re
import uuid
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Optional

from langchain_text_splitters import MarkdownHeaderTextSplitter
from config import DEFAULT_CONFIG, RAGConf

from langchain_core.documents import Document
from modules.embeddings import create_embedding
from langchain_community.vectorstores import FAISS

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 静音 huggingface_hub / urllib3 等第三方库的 INFO 日志
# 模型已本地缓存，无需其输出的 HTTP HEAD 请求探测日志
for _lib in ('huggingface_hub', 'urllib3', 'urllib3.connectionpool', 'filelock',
              'sentence_transformers', 'faiss'):
    logging.getLogger(_lib).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


class BuildIndex:
    """原始数据分块和向量化"""

    # 统一维护的分类与难度配置，供外部复用，避免关键词重复定义
    CATEGORY_MAPPING = {
        'meat_dish': '荤菜',
        'vegetable_dish': '素菜',
        'soup': '汤品',
        'dessert': '甜品',
        'breakfast': '早餐',
        'staple': '主食',
        'aquatic': '水产',
        'condiment': '调料',
        'drink': '饮品',
        'semi_finished': '半成品'
    }

    CATEGORY_LABELS = list(set(CATEGORY_MAPPING.values()))
    DIFFICULTY_LABELS = ['很简单', '简单', '中等', '较难', '困难']

    # -- 菜系关键词 → 标准化名称 --------------------------------------------
    CUISINE_MAP = {
        "川菜": "川菜", "四川": "川菜", "四川省": "川菜", "川菜系": "川菜",
        "粤菜": "粤菜", "广东": "粤菜",
        "湘菜": "湘菜", "湖南": "湘菜",
        "鲁菜": "鲁菜", "山东": "鲁菜",
        "苏菜": "苏菜", "江苏": "苏菜", "淮扬菜": "苏菜", "淮扬": "苏菜",
        "浙菜": "浙菜", "浙江": "浙菜",
        "沪菜": "沪菜", "上海菜": "沪菜", "上海": "沪菜",
        "闽菜": "闽菜", "福建": "闽菜",
        "徽菜": "徽菜", "安徽": "徽菜",
        "鄂菜": "鄂菜", "湖北": "鄂菜",
        "东北菜": "东北菜", "东北": "东北菜",
        "西北菜": "西北", "陕西": "西北", "山西": "西北",
        "黔菜": "黔菜", "贵州": "黔菜",
        "新疆菜": "新疆菜", "新疆": "新疆菜",
        "京菜": "京菜", "北京": "京菜", "京": "京菜",
        "客家": "客家", "潮汕": "潮汕",
        "清真": "清真",
        # 国际菜系
        "日本菜": "日本菜", "日本": "日本菜", "日式": "日本菜", "和食": "日本菜",
        "韩国菜": "韩国菜", "韩国": "韩国菜", "韩式": "韩国菜", "朝鲜": "韩国菜",
        "泰国菜": "泰国菜", "泰国": "泰国菜", "泰式": "泰国菜",
        "印度菜": "印度菜", "印度": "印度菜", "咖喱": "印度菜",
        "巴基斯坦菜": "巴基斯坦菜", "巴基斯坦": "巴基斯坦菜",
        "越南菜": "越南菜", "越南": "越南菜", "越式": "越南菜",
        "意大利菜": "意大利菜", "意大利": "意大利菜", "意式": "意大利菜",
        "法国菜": "法国菜", "法国": "法国菜", "法式": "法国菜",
        "英国菜": "英国菜", "英国": "英国菜", "英式": "英国菜",
        "美国菜": "美国菜", "美国": "美国菜", "美式": "美国菜",
        "德国菜": "德国菜", "德国": "德国菜", "德式": "德国菜",
        "俄罗斯菜": "俄罗斯菜", "俄罗斯": "俄罗斯菜", "俄式": "俄罗斯菜",
        "西班牙菜": "西班牙菜", "西班牙": "西班牙菜", "西班牙式": "西班牙菜",
        "墨西哥菜": "墨西哥菜", "墨西哥": "墨西哥菜", "墨西哥式": "墨西哥菜",
        "土耳其菜": "土耳其菜", "土耳其": "土耳其菜",
        "地中海菜": "地中海菜", "地中海": "地中海菜",
        "西餐": "西餐", "西式": "西餐", "洋食": "西餐", "欧美": "西餐",
        "东南亚菜": "东南亚菜", "东南亚": "东南亚菜",
    }
    CUISINE_LABELS = list(set(CUISINE_MAP.values()))

    # -- 碳水化合物相关解析配置 ---------------------------------------------
    # 优先从明确的营养成分行中提取；缺失时按高碳水食材估算
    CARB_NUTRITION_PATTERN = re.compile(
        r"(?:碳水化合物|碳水)[\s：:]\s*(\d+(?:\.\d+)?)\s*[克g]",
        re.IGNORECASE,
    )
    # 高碳水食材及其估算碳水占比（按可食用部质量计）
    CARB_INGREDIENTS = {
        "面粉": 0.76, "高筋面粉": 0.76, "中筋面粉": 0.76, "低筋面粉": 0.74,
        "全麦面粉": 0.72, "玉米淀粉": 0.90, "淀粉": 0.85,
        "大米": 0.77, "米饭": 0.28, "生米": 0.77,
        "面条": 0.25, "挂面": 0.75, "意大利面": 0.25, "意面": 0.25,
        "通心粉": 0.25, "米粉": 0.25, "粉丝": 0.20, "粉条": 0.20,
        "土豆": 0.17, "马铃薯": 0.17, "红薯": 0.20, "紫薯": 0.20, "南瓜": 0.07,
        "燕麦": 0.66, "燕麦片": 0.66, "麦片": 0.68,
        "年糕": 0.50, "糯米粉": 0.78, "白糖": 1.00, "白砂糖": 1.00,
        "冰糖": 1.00, "红糖": 0.95, "蜂蜜": 0.82, "黄糖": 0.98, "蔗糖": 1.00,
        "牛奶": 0.05, "面包糠": 0.72,
    }

    # -- 烹饪时间提取配置 --------------------------------------------------
    # 优先匹配显式总时长，如“烹饪时长：40 分钟”“制作耗时 10 分钟”“总计 17 分钟”
    TOTAL_TIME_PATTERNS = [
        re.compile(
            r"(?:烹饪时长|烹饪时间|制作耗时|制作时间|总用时|总时长|总时间|耗时|用时)[:：\s]*"
            r"(?:约|大约|大概|约莫|差不多|左右)?\s*"
            r"(\d+(?:\.\d+)?)\s*(?:[-~～]\s*(\d+(?:\.\d+)?))?\s*(分钟|分|小时)",
            re.IGNORECASE,
        ),
        # 分段描述中的“总计/合计/共/总共”后接的时间
        re.compile(
            r"(?:总计|合计|共|总共)[:：\s]*"
            r"(?:约|大约|大概|约莫|差不多|左右)?\s*"
            r"(\d+(?:\.\d+)?)\s*(?:[-~～]\s*(\d+(?:\.\d+)?))?\s*(分钟|分|小时)",
            re.IGNORECASE,
        ),
    ]
    # 操作段中需要累加时间的烹饪/等待动词
    COOKING_TIME_VERBS = [
        "煮", "炖", "蒸", "煎", "炸", "烤", "炒", "焖", "煲", "熬", "烧",
        "烩", "煨", "焗", "涮", "烫", "卤", "爆", "熘", "烙", "烘", "焯水",
        "等待", "腌制", "腌渍", "静置", "发酵", "浸泡", "醒", "加热",
    ]
    # 操作段累加时，单个步骤超过该阈值视为非烹饪等待（如发酵、冷藏、浸泡），不累计
    MAX_FALLBACK_STEP_MINUTES = 180

    COOKING_METHODS = [
        "炒", "蒸", "煎", "炸", "炖", "煮", "焖", "烧", "拌", "烤", "卤", "煲", "爆",
        "熏", "腌", "醉", "糟", "泡", "涮", "焗", "熘", "灼", "熬", "烙", "烘", "炝",
        "红烧", "清蒸", "白灼", "油焖", "干煎", "葱油", "蒜蓉", "麻辣", "糖醋",
        "酱烧", "酱", "拔丝", "油泼", "椒盐", "酸辣", "咕噜", "凉拌"
    ]

    
    def __init__(self, 
                 data_path: str = DEFAULT_CONFIG.data_path,
                 embedding_model: str = DEFAULT_CONFIG.embedding_model, 
                 index_save_path: str = DEFAULT_CONFIG.index_save_path,
                 config: Optional[RAGConf] = None):
        # logger.info("初始化BuildIndex组件")
        self.data_path = data_path
        self.embedding_model = embedding_model
        self.index_save_path = index_save_path
        self.config = config or DEFAULT_CONFIG

    # 加载数据
    def add_documents(self) -> List[Document]:
        data_path_obj = Path(self.data_path)

        documents = []
        for md_file in data_path_obj.rglob("*.md"):
            try:
                # 打开文件
                with open(md_file, "r", encoding="utf-8") as f:
                    content = f.read()

                # 为每个父文档分配确定性的唯一ID（基于数据根目录的相对路径）
                try:
                    data_root = Path(self.data_path).resolve()
                    relative_path = Path(md_file).resolve().relative_to(data_root).as_posix()
                except Exception:
                    relative_path = Path(md_file).as_posix()
                parent_id = hashlib.md5(relative_path.encode("utf-8")).hexdigest()

                # 父文档
                doc = Document(page_content=content, 
                               metadata={"source": str(md_file), 
                                         "parent_id": parent_id, 
                                         "doc_type": "parent"})
                documents.append(doc)
            except Exception as e:
                logger.warning(f"读取文件 {md_file} 失败： {e}")

        logger.info(f"成功加载{len(documents)}个文档")
        return documents
    
    # 提取元数据
    def extract_recipe(self, filepath: Path, recipe_id: str) -> Dict | None:
        """从单个markdown文件中提取结构化菜谱信息"""
        
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            logger.error(f"读取 {filepath} 文件失败：{e}")
            return


        # 菜品名字
        dish_name = filepath.stem

        # -- 菜品分类 -----------------------------------------------
        category = '其它'
        # 向上遍历目录（直到 data_path），匹配 CATEGORY_MAPPING 中的目录名
        data_root = Path(self.data_path).resolve()
        for parent in filepath.resolve().parents:
            if parent == data_root:
                break
            for key, value in self.CATEGORY_MAPPING.items():
                normalized_name = parent.name.replace("-", "_")
                if key == normalized_name or key in normalized_name:
                    category = value
                    break
            if category != '其它':
                break

        # -- 难度（★ → int）------------------------------------------------
        diff_match = re.search(r"烹饪难度[：:]\s*(★+|[1-5])", content)
        if diff_match and "★" in (diff_match.group(1) or ""):
            difficulty = len(diff_match.group(1))
        elif diff_match:
            difficulty = int(diff_match.group(1))
        else:
            difficulty = 3

        # -- 食材 -----------------------------------------------------------
        ingredients: list[str] = []
        m = re.search(
            r"##\s*必备原料和工具\s*\n(.*?)(\n##|\Z)", content, re.DOTALL
        )
        if m:
            for line in m.group(1).split("\n"):
                line = line.strip()
                if line.startswith("-") or line.startswith("*"):
                    item = line.lstrip("-* ").strip()
                    for p in re.split(r"[、，,;；]", item):
                        p = re.sub(r"\s*(or|或)\s+.*$", "", p).strip()
                        p = re.sub(r"[\(（].*?[\)）]", "", p).strip()
                        if p and len(p) <= 10 and not re.match(r"^[\d.]+", p):
                            ingredients.append(p)

        # -- 加工方式（标题 + 前 800 字内匹配）----------------------------------
        methods: set[str] = set()
        text = dish_name + " " + content[:800]
        for mtd in self.COOKING_METHODS:
            if mtd in text:
                methods.add(mtd)

        # -- 菜系（从 # ...的介绍 段提取，并补充标题中的国际菜系线索）------
        cuisines: set[str] = set()
        for keyword, cuisine in self.CUISINE_MAP.items():
            if keyword in content[:800]:
                cuisines.add(cuisine)

        # -- 烹饪时间 -------------------------------------------------------
        # 1. 优先匹配显式总时长
        cooking_time_min: float | None = None
        for pattern in self.TOTAL_TIME_PATTERNS:
            m = pattern.search(content)
            if m:
                cooking_time_min = self._parse_time_value(m.group(1), m.group(2), m.group(3))
                break
        # 2. 无显式总时长时，累加操作段中与烹饪动词相关的时间
        if cooking_time_min is None:
            op_match = re.search(r"##\s*操作\s*\n(.*?)(\n##|\Z)", content, re.DOTALL)
            if op_match:
                op_text = op_match.group(1)
                verb_group = "|".join(re.escape(v) for v in self.COOKING_TIME_VERBS)
                step_matches = re.findall(
                    rf"(?:{verb_group})\s*(?:约|大约|大概|约莫|差不多|左右)?\s*"
                    r"(\d+(?:\.\d+)?)\s*(?:[-~～]\s*(\d+(?:\.\d+)?))?\s*(分钟|分|小时)",
                    op_text,
                    re.IGNORECASE,
                )
                total = 0.0
                for low, high, unit in step_matches:
                    step_min = self._parse_time_value(low, high, unit)
                    # 过滤明显属于发酵/冷藏/长时间浸泡等非烹饪等待步骤
                    if step_min <= self.MAX_FALLBACK_STEP_MINUTES:
                        total += step_min
                if total > 0:
                    cooking_time_min = round(total, 1)
        # -- 碳水化合物含量（g）----------------------------------------------
        # 1. 优先匹配明确的营养成分行，如“碳水化合物：39g”
        carb_amount_g: float | None = None
        nutrition_match = self.CARB_NUTRITION_PATTERN.search(content)
        if nutrition_match:
            carb_amount_g = float(nutrition_match.group(1))
        else:
            # 2. 无明确营养标注时，从“计算”区的高碳水食材估算
            calc_match = re.search(r"##\s*计算\s*\n(.*?)(\n##|\Z)", content, re.DOTALL)
            if calc_match:
                carb_amount_g = self._estimate_carbs_from_ingredients(calc_match.group(1))

        # -- 文件路径(绝对路径) ----------------------------------
        relative_path = filepath.resolve().relative_to(data_root).as_posix()

        if carb_amount_g is not None:
            carb_amount_g = round(carb_amount_g, 1)

        return {
            "dish_name": dish_name,
            "category": category,
            "difficulty": difficulty,
            "ingredients": list(set(ingredients)),
            "methods": list(methods),
            "cuisines": sorted(cuisines),
            "cooking_time_min": cooking_time_min,
            "carb_amount_g": carb_amount_g,
            "source_file": str(relative_path),
            "recipe_id": recipe_id
        }


    # 提取高碳水食材碳水估算克数
    def _estimate_carbs_from_ingredients(self, calc_text: str) -> float | None:
        """从计算区文本中按高碳水食材估算碳水化合物克数。

        规则：
        - 仅识别带有明确数字+单位（g/克）的食材行；
        - 对“X-Yg”范围取平均值；
        - 按 CARB_INGREDIENTS 中配置的碳水占比估算；
        - 无法识别时返回 None。
        """
        total = 0.0
        matched = False
        for line in calc_text.split("\n"):
            line = line.strip()
            if not line.startswith(("-", "*")):
                continue
            line = line.lstrip("-* ").strip()
            # 提取质量数值，支持 "100-120g"、"100g"、"约 100 克" 等
            qty_match = re.search(
                r"(\d+(?:\.\d+)?)\s*(?:[-~～]\s*(\d+(?:\.\d+)?))?\s*(?:克|g)",
                line,
                re.IGNORECASE,
            )
            if not qty_match:
                continue
            low = float(qty_match.group(1))
            high = float(qty_match.group(2)) if qty_match.group(2) else low
            qty_g = (low + high) / 2.0
            # 去掉括号备注、量词，便于食材名匹配
            clean_line = re.sub(r"[（(].*?[）)]", "", line)
            clean_line = re.sub(r"\d+(?:\.\d+)?\s*(?:克|g|ml|毫升|个|瓣|包|杯|cups?|勺)", "", clean_line)
            clean_line = re.sub(r"(?:或|和|、|，|,|；|;|以及|还有|及)\s*$", "", clean_line).strip()
            for ing_name, ratio in self.CARB_INGREDIENTS.items():
                if ing_name in clean_line:
                    total += qty_g * ratio
                    matched = True
                    break
        return total if matched else None


    def _parse_time_value(self, low: str | float | None,
                          high: str | float | None,
                          unit: str | None) -> float:
        """把低/高两个时间数值与单位换算成分钟数。

        规则：
        - high 为空或 None 时只取 low；
        - 单位为“小时”时乘以 60；
        - 有范围时返回平均值。
        """
        if low is None:
            return 0.0
        low_val = float(low)
        if not high:
            high_val = low_val
        else:
            high_val = float(high)
        minutes = (low_val + high_val) / 2.0
        if unit and unit.startswith("小时"):
            minutes *= 60
        return round(minutes, 1)


    def enhance_metadata(self, documents: List[Document]) -> List[Document]:
        """根据文件路径和内容增强文档的元数据，包括菜品分类、名字和难度等级、菜系分类"""

        enhanced_docs = []
        recipes = []
        for doc in documents:
            file_path = Path(doc.metadata.get("source", ""))
            recipe_id = doc.metadata.get("parent_id", "")
            recipe = self.extract_recipe(file_path, recipe_id)
            
            if recipe:
                recipes.append(recipe)

                doc.metadata["category"] = recipe.get("category", None)
                doc.metadata["dish_name"] = recipe.get("dish_name", None)
                doc.metadata["difficulty"] = self.DIFFICULTY_LABELS[recipe.get("difficulty", 3)-1]
                doc.metadata["ingredients"] = recipe.get("ingredients", [])
                doc.metadata["methods"] = recipe.get("methods", [])
                doc.metadata["cuisines"] = recipe.get("cuisines", ["家常菜"])
                doc.metadata["cooking_time_min"] = recipe.get("cooking_time_min", None)
                doc.metadata["carb_amount_g"] = recipe.get("carb_amount_g", None)

            enhanced_docs.append(doc)
        
        logger.info(f"\nExtracted {len(recipes)} recipes")
        
        recipes_metadata = Path(self.config.recipes_metadata_path)
        recipes_metadata.parent.mkdir(parents=True, exist_ok=True)
        recipes_metadata.write_text(
            json.dumps(recipes, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        logger.info(f"\n菜品元数据已写入本地文件 {recipes_metadata}")

        return enhanced_docs
    
    
    # 文本分块
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """使用Markdown的文档结构进行分块"""

        if not documents:
            raise ValueError("没有文档可供分块")
        
        logger.info(f"开始进行文档分块")

        # 定义要分割的Markdown标题层级
        headers_to_split_on = [
            ("#", "主标题"),      # 菜品名称
            ("##", "二级标题"),   # 必备原料、计算、操作等
            ("###", "三级标题")   # 简易版本、复杂版本等
        ]

        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False)
        
        all_chunks = []
        
        for doc in documents:
            # 子块(子文档)
            md_chunks = markdown_splitter.split_text(doc.page_content)

            parent_id = doc.metadata.get("parent_id", 0)
            for i, chunk in enumerate(md_chunks):
                child_id = str(uuid.uuid4())
                
                chunk.metadata.update(doc.metadata) # 继承父文档的元数据
                chunk.metadata.update({
                    "chunk_id": child_id,
                    "parent_id": parent_id,
                    "doc_type": "child",
                    "chunk_index": i
                })

            all_chunks.extend(md_chunks)

        logger.info(f"文档分块结束，共得到 {len(all_chunks)} 个块")
        return all_chunks
    
    # 构建索引
    def build_index(self, chunks: List[Document]) -> FAISS:
        if not chunks:
            raise ValueError("没有文本块可供构建索引")
        
        logger.info(f"开始初始化嵌入模型")
        embedding = create_embedding(
            model_name=self.embedding_model,
        )
        
        logger.info(f"开始构建向量索引")
        vectorstore = FAISS.from_documents(documents=chunks, embedding=embedding)
        return vectorstore

    # 存储到本地
    def save_index(self, vectorstore: FAISS):
        if not vectorstore:
            raise ValueError("没有向量索引可供保存")
        
        # 确保保存目录存在
        Path(self.index_save_path).mkdir(parents=True, exist_ok=True)
        
        vectorstore.save_local(self.index_save_path)
        logger.info(f"向量索引已保存到 {self.index_save_path}")

    # 加载索引
    def load_index(self):
        logger.info(f"尝试加载本地索引")
        embedding = create_embedding(
            model_name=self.embedding_model,
        )
        if not Path(self.index_save_path).exists():
            logger.info(f"索引路径不存在: {self.index_save_path}, 需要重新构建索引")
            return None
        
        try:
            vectorstore = FAISS.load_local(self.index_save_path, 
                                           embedding, 
                                           allow_dangerous_deserialization=True)
            return vectorstore
        except Exception as e:
            logger.error(f"加载索引失败: {e}")
            return None
        

    @classmethod
    def get_supported_categories(cls) -> List[str]:
        return cls.CATEGORY_LABELS
    
    @classmethod
    def get_supported_difficulties(cls) -> List[str]:
        return cls.DIFFICULTY_LABELS
    
    @classmethod
    def get_supported_cuisine(cls) -> List[str]:
        return cls.CUISINE_LABELS
    
    @classmethod
    def get_supported_cooking_methods(cls) -> List[str]:
        return cls.COOKING_METHODS
    
    @classmethod
    def get_supported_carb_ingredients(cls) -> Dict[str, float]:
        return cls.CARB_INGREDIENTS
