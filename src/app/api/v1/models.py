from fastapi import APIRouter
from fastapi.responses import JSONResponse
import httpx

from ...config import get_settings

router = APIRouter(tags=["models"])

@router.get("/models")
async def models():
    settings = get_settings()

    client = httpx.AsyncClient(timeout=httpx.Timeout(10))
    try:
        # Call vllm to get the model info
        model_info = await client.get(f"{settings.VLLM_MODEL_URL}")
        if model_info.status_code != 200:
            return JSONResponse(status_code=500, content={"error": "Failed to get model info"})
        model_info = model_info.json()

        return JSONResponse(content=model_info)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": "Failed to get model info"})