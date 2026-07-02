
from sympy import false


class RAGConf:
    
    # 路径配置
    data_path: str = "./data/cook"
    index_save_path: str = "./vector_index"
    recipes_metadata_path: str = './graphify-out/.graphify_recipes.json'
    graph_json_path: str = './graphify-out/graph.json'

    # 模型配置
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    llm_model: str = "kimi-k2.5"
    temperature: float = 1
    max_tokens: int = 4096
    api_key: str = "MOONSHOT_API_KEY"

    # 检索配置
    top_k: int = 5
    threshold: float = 0.6

    # 是否需要更新百科菜品词条摘到原始菜品文档
    is_essential_update_cook_summary = False

    # 百度API KEY，替换自己的API_KEY
    baidu_api_key = "Bearer <BAIDU_API_KEY>" 

DEFAULT_CONFIG = RAGConf()

