import os
from typing import List

def get_cors_origins() -> List[str]:
    origins_str = os.getenv("CORS_ALLOWED_ORIGINS")
    if not origins_str:
        return ["*"]

    origins = [origin.strip() for origin in origins_str.split(",")]
    return origins