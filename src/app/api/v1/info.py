from fastapi import APIRouter
from fastapi.responses import JSONResponse
import httpx

from ...config import get_settings

router = APIRouter(tags=["info"])

@router.get("/info")
async def info():
    settings = get_settings()

    client = httpx.AsyncClient(timeout=httpx.Timeout(10))
    try:
        # Call vllm to get the model info
        model_info = await client.get(f"{settings.VLLM_MODEL_URL}")
        if model_info.status_code != 200:
            return JSONResponse(status_code=500, content={"error": "Failed to get model info"})
        model_info = model_info.json()
        data = model_info["data"][0]

        return JSONResponse(content={
            "ctx_len": data["max_model_len"],
            "model_name": data["id"],
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": "Failed to get model info"})