import httpx
import json
from fastapi.responses import JSONResponse
from typing import Union, List, Dict, Optional, Any
from datetime import datetime
import hashlib
import asyncio

from ...config import get_settings
from ...logger import log
from ...dependencies import get_milvus_wrapper, get_reranker
from .get_system_prompt import get_system_prompt

LLMSuccessResponse = Union[Dict[str, Any], List[Any]]
THRESHOLD = 0.6

async def arequest_llm(
    request_body: str,
    stream: bool = True,
    vllm_url: str = get_settings().VLLM_URL,
    user_id: str | None = None,
    is_api_key: bool = False,
    use_vector_db: bool = False
) -> Union[httpx.Response, JSONResponse, LLMSuccessResponse]:
    """
    Request LLM.
    - Returns httpx.Response if stream=True and status=200 (for caller to handle streaming).
    - Returns parsed JSON (dict or list) if stream=False and status=200 and response is valid JSON.
    - Returns JSONResponse if status != 200 or if stream=False and response is not valid JSON, or on request errors.
    """
    client = httpx.AsyncClient(timeout=httpx.Timeout(60 * 10))
    response: Optional[httpx.Response] = None

    # Add system prompt to the request body
    request_body = await _add_system_prompt(request_body, is_api_key)

    # Apply vector DB if enabled
    if use_vector_db:
        request_body = await _apply_vector_db(request_body, user_id)

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


def request_llm(
    request_body: str,
    stream: bool = True,
    vllm_url: str = get_settings().VLLM_URL,
    user_id: str | None = None,
    is_api_key: bool = False,
    use_vector_db: bool = False
) -> Union[httpx.Response, JSONResponse, LLMSuccessResponse]:
    """
    Request LLM (Synchronous version).
    - Returns httpx.Response if stream=True and status=200 (for caller to handle streaming).
    - Returns parsed JSON (dict or list) if stream=False and status=200 and response is valid JSON.
    - Returns JSONResponse if status != 200 or if stream=False and response is not valid JSON, or on request errors.
    """
    client = httpx.Client(timeout=httpx.Timeout(60 * 10))
    response: Optional[httpx.Response] = None

    # Add system prompt to the request body
    request_body = asyncio.run(_add_system_prompt(request_body, is_api_key))

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

async def _add_system_prompt(request_body: str, is_api_key: bool) -> str:
    """Add a system prompt to the messages."""
    request_body = json.loads(request_body)
    default_prompt = await get_system_prompt(request_body["model"], "default", is_api_key)
    if default_prompt:
        request_body["messages"] = [{"role": "system", "content": default_prompt.format(current_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))}] + request_body["messages"]
    return json.dumps(request_body)

def get_user_collection_name(user_id: str) -> str:
    """Get the user collection name."""
    hash_id = hashlib.sha256(user_id.encode()).hexdigest()
    return "user_" + hash_id

async def _apply_vector_db(request_body: str, user_id: str | None) -> str:
    """Apply vector DB to the request body."""
    if user_id is None:
        return request_body

    log.info(f"Applying vector DB to the request body for user {user_id}.")

    user_collection_name = get_user_collection_name(user_id)
    store_wrapper = get_milvus_wrapper()

    # Get the last message
    request_body = json.loads(request_body)
    last_message = request_body["messages"][-1]

    # Extract textual content from the last message.
    last_message_content: str | None = None
    for part in last_message.get("content", []):
        if isinstance(part, dict) and part.get("type") == "text" and "text" in part:
            last_message_content = part["text"].strip()
            break

    # If no text part is found, skip vector DB augmentation for this request.
    if not last_message_content:
        log.warning("Vector DB augmentation skipped: no text content found in the last message.")
        return json.dumps(request_body)

    # Get the top 3 most relevant documents
    docs = store_wrapper.similarity_search_for_user(user_collection_name, last_message_content, k=3)
    reranker = get_reranker()
    reranked_docs = reranker(
        query=last_message_content,
        documents=[doc.page_content for doc in docs],
        top_k=3
    )

    def _unique_keep_top(results, threshold=0.60, max_docs=3):
        seen, picked = set(), []
        for r in sorted(results, key=lambda r: -r.score):
            doc_key = hash(r.text.strip())          
            if r.score >= threshold and doc_key not in seen:
                picked.append(r)
                seen.add(doc_key)
            if len(picked) == max_docs:
                break
        return picked

    filtered_docs = _unique_keep_top(reranked_docs, threshold=THRESHOLD, max_docs=3)
    docs_str = "\n\n".join([filtered_doc.text for filtered_doc in filtered_docs])

    # Augment the request body with the documents
    if len(docs) > 0:
        vector_prompt = await get_system_prompt(request_body["model"], "vector")
        if vector_prompt:
            augmented_message = {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": vector_prompt.format(docs_str=docs_str, doc_count=len(docs))
                    }
                ]
            }
        else:
            augmented_message = None

        request_body["messages"] = request_body["messages"][:-1] + [augmented_message] + request_body["messages"][-1:]
    return json.dumps(request_body)
