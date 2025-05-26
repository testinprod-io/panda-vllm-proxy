from fastapi import APIRouter
from fastapi.responses import JSONResponse
from transformers import AutoConfig

from ...config import get_settings
from ...logger import log

router = APIRouter(tags=["info"])

@router.get("/info")
async def info():
    settings = get_settings()
    # If MAX_MODEL_LENGTH is set, use it
    if settings.MAX_MODEL_LENGTH:
        ctx_len = settings.MAX_MODEL_LENGTH
    else:
        try:
            cfg = AutoConfig.from_pretrained(settings.MODEL_NAME)
            # For Deepseek models
            if hasattr(cfg, 'max_position_embeddings'):
                ctx_len = cfg.max_position_embeddings
            else:
                # For Qwen 2.5 models
                if hasattr(cfg, 'talker_config'):
                    ctx_len = cfg.talker_config.max_position_embeddings
                else:
                    return JSONResponse(status_code=400, content={"error": "Max model length not found"})
        except Exception as e:
            log.error(f"Error getting model info: {e}")
            return JSONResponse(status_code=500, content={"error": "Error getting model info"})

    return JSONResponse(content={
        "ctx_len": ctx_len,
        "model_name": settings.MODEL_NAME,
    })