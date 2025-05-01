import httpx, os
import json
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv()

VLLM_URL = os.getenv("VLLM_URL")
TIMEOUT = 60 * 10

async def request_llm(request_body: str, stream: bool = True):
    """
    Request LLM with the given prompt.
    """
    client = httpx.AsyncClient(timeout=httpx.Timeout(TIMEOUT))
    # Forward the request to the vllm backend
    req = client.build_request("POST", VLLM_URL, content=request_body)
    response = await client.send(req, stream=stream)
    # If not 200, return the error response directly without streaming
    if response.status_code != 200:
        error_content = await response.aread()
        await response.aclose()
        await client.aclose()
        return JSONResponse(
            status_code=response.status_code, content=json.loads(error_content)
        )
    
    return response

