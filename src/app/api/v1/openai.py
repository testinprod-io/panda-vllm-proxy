import json
import os
from hashlib import sha256
import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse, PlainTextResponse
from dotenv import load_dotenv

from ...api.helper.auth import verify_authorization_header
from ...logger import log

load_dotenv()

router = APIRouter(tags=["openai"])

VLLM_URL = os.getenv("VLLM_URL")
TIMEOUT = 60 * 10

def hash(payload: str):
    return sha256(payload.encode()).hexdigest()

async def stream_vllm_response(request_body: bytes):
    # Modify the request body to use the correct model path and lowercasemodel name
    request_json = json.loads(request_body)
    request_json["model"] = request_json["model"].lower()
    modified_request_body = json.dumps(request_json)

    chat_id = None
    h = sha256()

    async def generate_stream(response):
        nonlocal chat_id, h
        async for chunk in response.aiter_text():
            h.update(chunk.encode())
            # Extract the chat id (data.id) from the first chunk
            if not chat_id:
                try:
                    data = chunk.strip("data: ").strip()
                    chunk_data = json.loads(data)
                    chat_id = chunk_data.get("id")
                except Exception as e:
                    error_message = f"Failed to parse the first chunk: {e}"
                    log.error(error_message)
                    raise Exception(error_message)
            yield chunk
        if not chat_id:
            error_message = "Chat id could not be extracted from the response"
            log.error(error_message)
            raise Exception(error_message)

    client = httpx.AsyncClient(timeout=httpx.Timeout(TIMEOUT))
    # Forward the request to the vllm backend
    req = client.build_request("POST", VLLM_URL, content=modified_request_body)
    response = await client.send(req, stream=True)
    # If not 200, return the error response directly without streaming
    if response.status_code != 200:
        error_content = await response.aread()
        await response.aclose()
        await client.aclose()
        return JSONResponse(
            status_code=response.status_code, content=json.loads(error_content)
        )

    return StreamingResponse(
        generate_stream(response),
        media_type="text/event-stream",
    )

@router.post("/chat/completions", dependencies=[Depends(verify_authorization_header)])
async def chat_completions(request: Request, background_tasks: BackgroundTasks):
    request_body = await request.body()
    return await stream_vllm_response(request_body)
