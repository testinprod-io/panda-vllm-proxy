from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    VLLM_URL: str
    MODEL_NAME: str
    SUMMARIZATION_MODEL: str | None = None
    MULTI_MODAL_MODEL: str | None = None

    MAX_RESULTS: int = 5
    SEARCH_TIMEOUT: float = 10.0
    USER_AGENT: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

    JWT_ALGORITHM: str
    JWT_PUB_KEY: str
    APP_ID: str

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