import os
import sys
import math
import logging
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from config import DEFAULT_CONFIG, RAGConf
from modules.build_index import BuildIndex
from modules.llm_generator import LLMGenerator
from modules.query_retrial import QueryRetrail
from modules.question_refactor import QuestionRefactor
from modules.baike_crawler import BaikeCrawler
from langchain_community.chat_models.moonshot import MoonshotChat


# 添加模块路径
sys.path.append(str(Path(__file__).parent))

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
        
        # 更新菜品词条摘要
        if self.config.is_essential_update_cook_summary:
            logger.info("开始获取百度百科词条菜品摘要并更新到原始菜品markdown文档，这可能需要一点时间，请耐心等待...")
            self.baike_crawler = BaikeCrawler()
            try:
                self.baike_crawler.run()
            except Exception as e:
                logger.error(f"更新词条摘要信息异常: {e}")
            

    def run(self):
        logger.info("开始运行RAG系统")
        self.index_builder = BuildIndex()
        # 检查索引是否存在
        vectorstore = self.index_builder.load_index()

        # 如果索引不存在，则构建新的索引
        documents =self.index_builder.add_documents();
        enhanced_docs = self.index_builder.enhance_metadata(documents)
        chunks = self.index_builder.chunk_documents(enhanced_docs)
        if not vectorstore:
            logger.info("未找到索引，开始构建新的索引")
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

                llm = MoonshotChat(
                    client=None,  
                    model=self.config.llm_model,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    api_key=os.getenv(self.config.api_key)
                )

                # 用户查询优化
                self.llm_generator = LLMGenerator()
                self.question_refactor = QuestionRefactor()
                question_type = self.question_refactor.question_router(llm, user_query)
                logger.info(f"用户问题类型分类为：{question_type}")

                rewrite_query = 'general'
                top_k = self.config.top_k
                if question_type == 'list':
                    rewrite_query = user_query
                    top_k = self.llm_generator.generate_list_topk(llm, rewrite_query, top_k)
                else:
                    rewrite_query = self.question_refactor.question_rewrite(llm, user_query, question_type)
                
                # 查询检索
                self.query_retriver = QueryRetrail(vectorstore)
                # 向量检索的文本块合并成父文档后可能会减少，因此先按1.25倍的Top K检索文本块和实现元数据过滤
                docs = self.query_retriver.verctor_retrial(user_query, question_type, top_k=math.ceil(top_k * 1.25)) 
                filter_docs = self.query_retriver.metadata_filter_query(user_query, docs, top_k=math.ceil(top_k * 1.25))
                parent_docs = self.query_retriver.get_parent_documents(filter_docs, documents, top_k)
                logger.info(f"检索到 {len(parent_docs)} 个相关文档块")

                if len(parent_docs) == 0:
                    print("未检索到相关文档，无法基于LLM生成有效回复，请重新调整后再输入。")
                    continue

                # 生成答案
                response = ""
                if question_type == "list":
                    response = self.llm_generator.generate_list_answer(parent_docs, top_k)
                elif question_type == "detail":
                    response = self.llm_generator.generate_detail_answer(llm, parent_docs, rewrite_query)
                else:
                    response = self.llm_generator.generate_normal_answer(llm, parent_docs, rewrite_query)

                # 返回结果
                print(response)
                
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
