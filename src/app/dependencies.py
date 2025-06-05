from .config import get_settings
from .milvus import MilvusWrapper
from pymilvus.model.reranker import BGERerankFunction

def get_cors_origins():
    settings = get_settings()
    return settings.CORS_ALLOWED_ORIGINS or ["*"]

_milvus_wrapper_instance: MilvusWrapper | None = None

def get_milvus_wrapper() -> MilvusWrapper:
    global _milvus_wrapper_instance
    if _milvus_wrapper_instance is None:
        _milvus_wrapper_instance = MilvusWrapper()
    return _milvus_wrapper_instance

_reranker_instance: BGERerankFunction | None = None

def get_reranker() -> BGERerankFunction:
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = BGERerankFunction(
            model_name="BAAI/bge-reranker-v2-m3",
            device="cpu"
        )
    return _reranker_instance