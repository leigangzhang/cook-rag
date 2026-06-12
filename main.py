import os
import sys
import logging
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from config import DEFAULT_CONFIG, RAGConf
from build_index import BuildIndex

# 添加模块路径
sys.path.append(str(Path(__file__).parent))

from build_index import BuildIndex
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_models.moonshot import MoonshotChat
from langchain_core.output_parsers import StrOutputParser

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RAGSystem:
    def __init__(self, config: Optional[RAGConf] = None): 
        logger.info("初始化RAG系统")
        self.config = config or DEFAULT_CONFIG

        # 检查数据路径
        logger.info(f"检查数据路径: {self.config.data_path}")
        if not Path(self.config.data_path).exists():
            raise FileNotFoundError(f"数据路径不存在: {self.config.data_path}")

        # 检查API密钥
        logger.info(f"检查API密钥环境变量: {self.config.api_key}")
        if not os.getenv(self.config.api_key):
            raise ValueError(f"请设置环境变量 {self.config.api_key}")
        
        # 初始化依赖组件
        # logger.info("初始化索引构建组件")
        self.index_builder = BuildIndex()
        

    def run(self):
        logger.info("开始运行RAG系统")
        # 检查索引是否存在
        vectorstore = self.index_builder.load_index()

        # 如果索引不存在，则构建新的索引
        if not vectorstore:
            logger.info("未找到索引，开始构建新的索引")
            documents =self.index_builder.add_documents();
            chunks = self.index_builder.chunk_documents(documents)
            vectorstore = self.index_builder.build_index(chunks)
            self.index_builder.save_index(vectorstore)
            logger.info("知识库构建完成")

        # 用户输入问题
        print("请输入您的问题（输入 'exit' 退出）：")

        while True:
            try: 
                user_query = input("> ")
                if user_query.lower() == "exit":
                    print("感谢使用，再见！")
                    break
                
                # 查询检索
                vector_retriver = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": self.config.top_k})
                vector_chunks = vector_retriver.invoke(user_query)
                logger.info(f"检索到 {len(vector_chunks)} 个相关文档块")

                # 生成答案
                prompt = ChatPromptTemplate.from_template("""
你是一位专业的烹饪助手。请根据以下食谱信息回答用户的问题。

用户问题: {question}

相关食谱信息:
{context}

请提供详细、实用的回答。如果信息不足，请诚实说明。
""")
                
                llm = MoonshotChat(
                    client=None,  
                    model=self.config.llm_model,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    api_key=os.getenv(self.config.api_key)
                )
                
                chain = (
                    {"question": RunnablePassthrough(), "context": lambda _: vector_chunks} 
                    | prompt
                    | llm
                    | StrOutputParser()
                )

                # 返回结果
                for response in chain.stream(user_query):
                    print(response, end="", flush=True)
                print()
                
            except KeyboardInterrupt:
                print("\n感谢使用，再见！")
                break
            except Exception as e:
                logger.error(f"处理用户输入时发生错误: {e}")
                print("发生错误，请重试。")
                continue

def main():
    rag = RAGSystem()
    rag.run()

if __name__ == "__main__":
    main()
