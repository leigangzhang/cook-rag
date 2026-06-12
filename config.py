
class RAGConf:
    
    # 路径配置
    data_path: str = "./data/cook"
    index_save_path: str = "./vector_index"

    # 模型配置
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    llm_model: str = "kimi-k2.5"
    temperature: float = 1
    max_tokens: int = 4096
    api_key: str = "MOONSHOT_API_KEY"

    # 检索配置
    top_k: int = 3

DEFAULT_CONFIG = RAGConf()

