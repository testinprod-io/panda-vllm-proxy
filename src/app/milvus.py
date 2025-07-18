import os
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from langchain.docstore.document import Document

from .config import get_settings

class MilvusWrapper:
    collection_name : str = "panda_collection_v1"
    embeddings : HuggingFaceEmbeddings = None
    milvus_collection : Milvus = None

    def __init__(self):
        pid = os.getpid()
        cache_dir = os.path.join(os.environ["HF_HOME"], str(pid))
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-en-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": False},
            cache_folder=cache_dir,
        )
        self.milvus_collection = Milvus(
            embedding_function=self.embeddings,
            collection_name=self.collection_name,
            connection_args={"uri": get_settings().MILVUS_URI},
            collection_properties={"collection.ttl.seconds": 86400},
            auto_id=True,
            drop_old=False,
        )

    def from_documents_for_user(self, user_id: str, documents: list[Document]) -> None:
        docs = [Document(page_content=t.page_content, metadata={"user_id": user_id}) for t in documents]
        self.milvus_collection.add_documents(docs)

    def similarity_search_for_user(self, user_id: str, query: str, k: int = 4):
        expr = f'user_id == "{user_id}"'
        return self.milvus_collection.similarity_search(query, k=k, expr=expr)