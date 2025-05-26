from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from typing import Optional

class Settings(BaseSettings):
    # LLM server urls
    VLLM_URL: str
    SUMMARIZATION_VLLM_URL: str

    # LLM model names
    MODEL_NAME: str
    SUMMARIZATION_MODEL: str | None = None
    MULTI_MODAL_MODEL: str | None = None
    MAX_MODEL_LENGTH: int | None = None

    # RAG config
    MAX_RESULTS: int = 5
    SEARCH_TIMEOUT: float = 10.0
    USER_AGENT: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    BRAVE_SEARCH_API_KEY: Optional[str] = None

    # JWT config
    JWT_ALGORITHM: str
    JWT_PUB_KEY: str
    APP_ID: str

    # CORS config
    CORS_ALLOWED_ORIGINS: list[str] = ["*"]

    TLS_CERT_PATH: str | None = None
    TLS_CERT_PRIVATE_KEY_PATH: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True
    )

@lru_cache()
def get_settings() -> Settings:
    return Settings() 