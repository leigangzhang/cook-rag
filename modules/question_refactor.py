import logging

from prompt import LLMPrompts
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models.moonshot import MoonshotChat

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class QuestionRefactor:

    def __init__(self):
        pass

    def question_router(self, llm: MoonshotChat, user_query: str) -> str:
        """根据用户问题定义不同的查询类型，判定方式：llm语义判定"""
        
        prompt = ChatPromptTemplate.from_template(LLMPrompts.QUESTION_ROUTER_PROMPT)
        
        chain = (
            {"query":RunnablePassthrough()}
             |prompt
             |llm
             |StrOutputParser()
        )

        result = chain.invoke(user_query).strip().lower()

        if result in ["list", "detail", "general"]:
            return result
        else:
            return "general"


    def question_rewrite(self, llm: MoonshotChat, user_query: str, question_type: str):
        """对详细查询(detail)和一般查询(general)进行重写，重写方式: llm重写"""

        # 重写类型检查
        assert question_type in ['detail', 'general']

        prompt = ChatPromptTemplate.from_template(LLMPrompts.QUESTION_REWRITE_PROMPT)
        
        chain = (
            {"query": RunnablePassthrough()}
            |prompt
            |llm
            |StrOutputParser()
        )

        result = chain.invoke(user_query).strip()

        if user_query != result:
            logger.info(f"用户查询已重写：{user_query} -> {result}")
        else:
            logger.info(f"用户查询无须改写：{user_query}")


