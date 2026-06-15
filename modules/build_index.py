import os
import sys
import uuid
import logging
import hashlib
from pathlib import Path
from typing import List, Optional

from langchain_text_splitters import MarkdownHeaderTextSplitter
from config import DEFAULT_CONFIG, RAGConf

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
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
        'drink': '饮品'
    }

    CATEGORY_LABELS = list(set(CATEGORY_MAPPING.values()))
    DIFFICULTY_LABELS = ['非常简单', '简单', '中等', '困难', '非常困难']
    
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
    
    # 增强元数据
    def enhance_metadata(self, documents: List[Document]) -> List[Document]:
        """根据文件路径和内容增强文档的元数据，包括菜品分类、名字和难度等级、菜系分类"""

        enhanced_docs = []
        dish_names = []
        for doc in documents:
            file_path = Path(doc.metadata.get("source", ""))
            path_parts = file_path.parts

            # 菜品分类
            for key, value in self.CATEGORY_MAPPING.items():
                if key in path_parts:
                    doc.metadata["category"] = value
                    break

            # 菜品名字
            doc.metadata["dish_name"] = file_path.stem
            dish_names.append(doc.metadata["dish_name"])

            # 难度等级
            content = doc.page_content
            if '★★★★★' in content:
                doc.metadata['difficulty'] = '非常困难'
            elif '★★★★' in content:
                doc.metadata['difficulty'] = '困难'
            elif '★★★' in content:
                doc.metadata['difficulty'] = '中等'
            elif '★★' in content:
                doc.metadata['difficulty'] = '简单'
            elif '★' in content:
                doc.metadata['difficulty'] = '非常简单'
            else:
                doc.metadata['difficulty'] = '未知'

            enhanced_docs.append(doc)
        
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
        embedding = HuggingFaceEmbeddings(model_name=self.embedding_model,
                                           model_kwargs={"device": "cpu"},
                                           encode_kwargs={'normalize_embeddings': True})
        
        
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
        embedding = HuggingFaceEmbeddings(model_name=self.embedding_model,
                                           model_kwargs={"device": "cpu"},
                                           encode_kwargs={'normalize_embeddings': True})
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