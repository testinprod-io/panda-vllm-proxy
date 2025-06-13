from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from typing import Optional

class Settings(BaseSettings):
    # LLM server urls
    VLLM_URL: str
    VLLM_MODEL_URL: str
    SUMMARIZATION_VLLM_URL: str

    # LLM model names
    MODEL_NAME: str
    SUMMARIZATION_MODEL: str | None = None

    # RAG config
    BRAVE_SEARCH_API_KEY: Optional[str] = None
    MILVUS_URI: str

    # JWT config
    JWT_ALGORITHM: str
    JWT_PUB_KEY: str
    APP_ID: str

    # CORS config
    CORS_ALLOWED_ORIGINS: list[str] = ["*"]

    # App server config
    PANDA_APP_SERVER: str
    PANDA_APP_SERVER_TOKEN: str

    TLS_CERT_PATH: str | None = None
    TLS_CERT_PRIVATE_KEY_PATH: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True
    )

@lru_cache()
def get_settings() -> Settings:
    return Settings()