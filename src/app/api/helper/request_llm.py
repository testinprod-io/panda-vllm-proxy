import httpx
import json
from fastapi.responses import JSONResponse
from typing import Union, List, Dict, Optional, Any

from ...config import get_settings

settings = get_settings()

VLLM_URL = settings.VLLM_URL
TIMEOUT = 60 * 10
LLMSuccessResponse = Union[Dict[str, Any], List[Any]]

async def arequest_llm(request_body: str, stream: bool = True, vllm_url: str = VLLM_URL) -> Union[httpx.Response, JSONResponse, LLMSuccessResponse]:
    """
    Request LLM.
    - Returns httpx.Response if stream=True and status=200 (for caller to handle streaming).
    - Returns parsed JSON (dict or list) if stream=False and status=200 and response is valid JSON.
    - Returns JSONResponse if status != 200 or if stream=False and response is not valid JSON, or on request errors.
    """
    client = httpx.AsyncClient(timeout=httpx.Timeout(TIMEOUT))
    response: Optional[httpx.Response] = None
    try:
        headers = { "Content-Type": "application/json" }
        req = client.build_request("POST", vllm_url, content=request_body, headers=headers)
        response = await client.send(req, stream=stream)

        # Handle non-200 status codes
        if response.status_code != 200:
            error_content_bytes = await response.aread()
            try:
                error_json = json.loads(error_content_bytes.decode('utf-8'))
                return JSONResponse(status_code=response.status_code, content=error_json)
            except json.JSONDecodeError:
                return JSONResponse(
                    status_code=response.status_code,
                    content={"error": {"message": "Invalid or non-JSON error response from upstream LLM"}}
                )
            finally:
                if hasattr(response, 'aclose'):
                    await response.aclose()

        # Handle successful requests
        if stream:
            return response
        else:
            content_bytes = await response.aread()
            try:
                parsed_data: LLMSuccessResponse = json.loads(content_bytes.decode('utf-8'))
                return parsed_data
            except json.JSONDecodeError:
                return JSONResponse(
                    status_code=502,
                    content={"error": {"message": "Received invalid JSON format from upstream LLM"}}
                )
            finally:
                if hasattr(response, 'aclose'):
                    await response.aclose()

    except httpx.RequestError as e:
        return JSONResponse(status_code=503, content={"error": {"message": f"Service Unavailable: Cannot connect to LLM backend. {e}"}})
    except Exception as e:
        if response and hasattr(response, 'aclose') and not response.is_closed:
            await response.aclose()
        return JSONResponse(status_code=500, content={"error": {"message": f"Internal server error during LLM request: {e}"}})
    finally:
        is_streaming_success = stream and response is not None and response.status_code == 200 and not isinstance(response, JSONResponse)
        if client and not client.is_closed and not is_streaming_success:
            await client.aclose()


def request_llm(request_body: str, stream: bool = True, vllm_url: str = VLLM_URL) -> Union[httpx.Response, JSONResponse, LLMSuccessResponse]:
    """
    Request LLM (Synchronous version).
    - Returns httpx.Response if stream=True and status=200 (for caller to handle streaming).
    - Returns parsed JSON (dict or list) if stream=False and status=200 and response is valid JSON.
    - Returns JSONResponse if status != 200 or if stream=False and response is not valid JSON, or on request errors.
    """
    client = httpx.Client(timeout=httpx.Timeout(TIMEOUT))
    response: Optional[httpx.Response] = None
    try:
        headers = { "Content-Type": "application/json" }
        req = client.build_request("POST", vllm_url, content=request_body, headers=headers)
        response = client.send(req, stream=stream)

        # Handle non-200 status codes
        if response.status_code != 200:
            try:
                error_content_bytes = response.read()
                error_json = json.loads(error_content_bytes.decode('utf-8'))
                return JSONResponse(status_code=response.status_code, content=error_json)
            except json.JSONDecodeError:
                return JSONResponse(
                    status_code=response.status_code,
                    content={"error": {"message": "Invalid or non-JSON error response from upstream LLM"}}
                )
            finally:
                if response and hasattr(response, 'close'):
                    response.close()

        # Handle successful requests
        if stream:
            return response
        else:
            try:
                content_bytes = response.read()
                parsed_data: LLMSuccessResponse = json.loads(content_bytes.decode('utf-8'))
                return parsed_data
            except json.JSONDecodeError:
                return JSONResponse(
                    status_code=502,
                    content={"error": {"message": "Received invalid JSON format from upstream LLM"}}
                )
            finally:
                if response and hasattr(response, 'close'):
                    response.close()

    except httpx.RequestError as e:
        return JSONResponse(status_code=503, content={"error": {"message": f"Service Unavailable: Cannot connect to LLM backend. {e}"}})
    except Exception as e:
        if response and hasattr(response, 'close') and not response.is_closed:
            response.close()
        return JSONResponse(status_code=500, content={"error": {"message": f"Internal server error during LLM request: {e}"}})
    finally:
        is_streaming_success = stream and response is not None and response.status_code == 200 and isinstance(response, httpx.Response)
        if client and not client.is_closed and not is_streaming_success:
            client.close()

