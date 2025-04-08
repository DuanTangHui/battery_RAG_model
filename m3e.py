from typing import List
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

class M3EBaseEmbeddings(Embeddings):
    """
    基于 Hugging Face 官方提供的 m3e-base 模型，
    封装成 langchain 风格的 Embeddings 接口。
    """

    def __init__(self, model_name: str = "moka-ai/m3e-base"):
        """
        初始化时加载 SentenceTransformer 模型。
        Args:
            model_name (str): 模型名称，默认为 'moka-ai/m3e-base'
        """
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            raise ValueError(f"无法加载模型 {model_name}，请检查模型名称或网络连接。原始错误：{e}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        对输入的文档列表生成 embedding。
        
        Args:
            texts (List[str]): 待编码的文本列表。
        
        Returns:
            List[List[float]]: 每个文本对应的 embedding（浮点数列表）。
        """
        # 调用 model.encode 得到 embedding，转换为 list 格式返回
        embeddings = self.model.encode(texts)
        return embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        对单个查询文本生成 embedding。
        
        Args:
            text (str): 待编码的查询文本。
        
        Returns:
            List[float]: 文本的 embedding（浮点数列表）。
        """
        return self.embed_documents([text])[0]
