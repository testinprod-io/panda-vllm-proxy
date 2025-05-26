from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import Response, JSONResponse

from ...api.helper.auth import verify_authorization_header
from ...api.helper.request_llm import arequest_llm
from ...api.helper.streaming import create_streaming_response
from ...logger import log
from ...actions.registry import get_action_registry
from .models import LLMRequest

router = APIRouter(tags=["openai"])

async def stream_vllm_response(payload: LLMRequest) -> Response:
    """
    Process the LLMRequest payload and handle custom actions if specified.
    Returns either a custom action response or streams the LLM response.
    """
    try:
        action_registry = get_action_registry()
        for action_key, handler in action_registry.items():
            if hasattr(payload, action_key) and getattr(payload, action_key) is True:
                log.info(f"Executing custom action: {action_key} based on LLMRequest field: {getattr(payload, action_key)}")
                return await handler(payload) 

        modified_request_body_str = payload.model_dump_json(exclude_none=True)
        should_stream = payload.stream 

        response_from_llm = await arequest_llm(modified_request_body_str, stream=should_stream)
        
        if isinstance(response_from_llm, JSONResponse):
            return response_from_llm
        
        if should_stream:
            return create_streaming_response(response_from_llm)
        else:
            return JSONResponse(content=response_from_llm)

    except Exception as e:
        log.error(f"Error processing request in stream_vllm_response: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@router.post("/chat/completions", dependencies=[Depends(verify_authorization_header)])
async def chat_completions(payload: LLMRequest, background_tasks: BackgroundTasks) -> Response:
    """OpenAI-compatible chat completions endpoint with support for custom actions."""
    return await stream_vllm_response(payload) # Pass the validated Pydantic model instance
