import json
from typing import cast
from fastapi import APIRouter, BackgroundTasks, Depends, Request, HTTPException
from fastapi.responses import Response, JSONResponse

from ...api.helper.auth import verify_authorization_header
from ...api.helper.request_llm import request_llm
from ...api.helper.streaming import create_streaming_response
from ...logger import log
from ...actions.registry import get_action_registry
from ...actions.models import ActionRequest

router = APIRouter(tags=["openai"])

async def stream_vllm_response(request_body: bytes) -> Response:
    """
    Process the request body and handle custom actions if specified.
    Returns either a custom action response or streams the LLM response.
    """
    try:
        request_data = json.loads(request_body)
        request_json: ActionRequest = cast(ActionRequest, request_data)

        # Check for custom actions
        action_registry = get_action_registry()
        for action_key, handler in action_registry.items():
            if action_key in request_json and request_json[action_key] is True:
                log.info(f"Executing custom action: {action_key}")
                return await handler(request_json)
        
        modified_request_body = json.dumps(request_json)
        
        # Request the LLM and create streaming response
        response = await request_llm(modified_request_body)
        if isinstance(response, JSONResponse):
            return response

        return create_streaming_response(response)

    except json.JSONDecodeError:
        log.error("Invalid JSON in request body")
        raise HTTPException(status_code=400, detail="Invalid JSON in request body")
    except Exception as e:
        log.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@router.post("/chat/completions", dependencies=[Depends(verify_authorization_header)])
async def chat_completions(request: Request, background_tasks: BackgroundTasks) -> Response:
    """OpenAI-compatible chat completions endpoint with support for custom actions."""
    request_body = await request.body()
    return await stream_vllm_response(request_body)
