import os 
import sys
import logging

from typing import List

from prompt import LLMPrompts
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import MoonshotChat
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LLMGenerator:

    def __init__(self):
        pass


    def generate_list_answer(self, docs: List[Document], user_query: str) -> str:
        """生成菜品清单"""
    
        if not docs:
            return "抱歉，没有找到相关的菜品信息。"

        logger.info("使用「菜品清单LLM生成器」生成结果中...")
        # 提取菜品名称
        dish_names = []
        for doc in docs:
            dish_name = doc.metadata.get('dish_name', '未知菜品')
            if dish_name not in dish_names:
                dish_names.append(dish_name)

        # 构建简洁的列表回答
        if len(dish_names) == 1:
            return f"为您推荐：{dish_names[0]}"
        elif len(dish_names) <= 3:
            return f"为您推荐以下菜品：\n" + "\n".join([f"{i+1}. {name}" for i, name in enumerate(dish_names)])
        else:
            return f"为您推荐以下菜品：\n" + "\n".join([f"{i+1}. {name}" for i, name in enumerate(dish_names[:3])]) + f"\n\n还有其他 {len(dish_names)-3} 道菜品可供选择。"


    def generate_detail_answer(self, llm: MoonshotChat, docs: List[Document], user_query: str) -> str:
        """生成详细菜品做法"""

        builded_docs = self._build_context(docs)

        logger.info("使用「菜品步骤LLM生成器」生成结果中...")
        prompt = PromptTemplate.from_template(LLMPrompts.GENERATE_DETAIL_PROMPT)

        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: builded_docs}
            |prompt
            |llm
            |StrOutputParser()
        )

        result = chain.invoke(user_query)
        return result
        
    
    def generate_normal_answer(self, llm: MoonshotChat, docs: List[Document], user_query: str):
        """生成正常答复"""
        builded_docs = self._build_context(docs)

        logger.info("使用「正常答复LLM生成器」生成结果中...")
        prompt = PromptTemplate.from_template(LLMPrompts.GENERATE_NORMAL_ANSWER)

        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: builded_docs}
            |prompt
            |llm
            |StrOutputParser()
        )

        result = chain.invoke(user_query)
        return result


    def _build_context(self, docs: List[Document], max_length: int = 4096) -> str:
        """将元数据和父文档整理成格式清晰的文档，方便大模型理解"""
        if not docs:
            return "暂无相关食谱信息。"
        
        context_parts = []
        current_length = 0
        
        for i, doc in enumerate(docs, 1):
            # 添加元数据信息
            metadata_info = f"【食谱 {i}】"
            if 'dish_name' in doc.metadata:
                metadata_info += f" {doc.metadata['dish_name']}"
            if 'category' in doc.metadata:
                metadata_info += f" | 分类: {doc.metadata['category']}"
            if 'difficulty' in doc.metadata:
                metadata_info += f" | 难度: {doc.metadata['difficulty']}"
            
            # 构建文档文本
            doc_text = f"{metadata_info}\n{doc.page_content}\n"
            
            # 检查长度限制
            if current_length + len(doc_text) > max_length:
                break
            
            context_parts.append(doc_text)
            current_length += len(doc_text)
        
        divider = "\n" + "="*50 + "\n"
        result = divider + divider.join(context_parts)
        # logger.info(f"格式统一和拼接后的父文档：\n{result}")
        
        return result