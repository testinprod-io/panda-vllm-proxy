from .config import get_settings
from .milvus import MilvusWrapper

def get_cors_origins():
    settings = get_settings()
    return settings.CORS_ALLOWED_ORIGINS or ["*"]

_milvus_wrapper_instance: MilvusWrapper | None = None

def get_milvus_wrapper() -> MilvusWrapper:
    global _milvus_wrapper_instance
    if _milvus_wrapper_instance is None:
        _milvus_wrapper_instance = MilvusWrapper()
    return _milvus_wrapper_instance