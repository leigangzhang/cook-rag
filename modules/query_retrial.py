import os
import sys
import logging
import hashlib
from typing import Optional, List

from modules.build_index import BuildIndex
from config import DEFAULT_CONFIG, RAGConf
from langchain.vectorstores import FAISS
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
        logger.info(f"使用向量检索器检索到 {len(vector_chunks)} 个相关文档块")
        
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
        """在向量检索和BM25检索的基础上做元数据过滤"""
        # 从用户问题中提取元数据过滤条件
        filters = {}
        category_keywords = BuildIndex.get_supported_categories()
        for cat in category_keywords:
            if cat in user_query:
                filters['category'] = cat
                break

        difficulty_keywords = BuildIndex.get_supported_difficulties()
        for diff in difficulty_keywords:
            if diff in user_query:
                filters['difficulty'] = diff
                break

        # 带元数据过滤的查询检索
        filtered_docs = []
        for doc in docs:
            match = True
            for key, value in filters.items():
                if key in doc.metadata:
                    if isinstance(value, list):
                        if doc.metadata[key] not in value:
                            match = False
                            break
                    else:
                        if doc.metadata[key] != value:
                            match = False
                            break
                else:
                    match = False
                    break
            
            if match:
                filtered_docs.append(doc)
                if len(filtered_docs) >= top_k:
                    break

        logger.info(f"元数据过滤文档：从 {len(docs)} 个文档中过滤菜品品类和难度后只保留 {len(filtered_docs)} 个文档")
        
        return filtered_docs


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