from .config import get_settings

def get_cors_origins():
    settings = get_settings()
    return settings.CORS_ALLOWED_ORIGINS or ["*"]