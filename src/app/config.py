from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from typing import Optional
import pathlib

from .logger import log

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
    JWT_PUB_KEY_FILE: str
    APP_ID: str

    # CORS config
    CORS_ALLOWED_ORIGINS: list[str] = ["*"]

    TLS_CERT_PATH: str | None = None
    TLS_CERT_PRIVATE_KEY_PATH: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True
    )

    def load_jwt_public_key(self) -> bytes:
        """Read the PEM file and return raw bytes."""
        file_path = pathlib.Path(self.JWT_PUB_KEY_FILE).expanduser()
        with file_path.open("rb") as f:
            return f.read()

@lru_cache()
def get_settings() -> Settings:
    return Settings()