import httpx
from typing import Optional, List, Union
import json
from fastapi.responses import JSONResponse

from ...config import get_settings
from ...api.helper.get_system_prompt import get_system_prompt
from ...api.v1.schemas import LLMRequest, ToolCall, ContentPart
from ...logger import log
from ...actions.tool_calls.get_tools import get_default_tools

settings = get_settings()
SUMMARIZATION_MODEL = settings.SUMMARIZATION_MODEL
SUMMARIZATION_VLLM_URL = settings.SUMMARIZATION_VLLM_URL

async def request_classification(content: Union[str, ContentPart, List[ContentPart]] , max_tokens: int = 50) -> tuple[bool, List[ToolCall]]:
    """
    Request classification for a given text.
    """
    client = httpx.AsyncClient(timeout=httpx.Timeout(60 * 10))
    response: Optional[httpx.Response] = None

    content_text = content
    if isinstance(content, ContentPart):
        content_text = content.text
    elif isinstance(content, List):
        content_text = " ".join([part.text for part in content])

    classification_prompt = await get_classification_prompt(content_text)

    try:
        request_body: LLMRequest = {
            "model": SUMMARIZATION_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": content_text + "\n\n" + classification_prompt
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.2,
            "stream": False,
            "tools": get_default_tools(),
            "tool_choice": "auto"
        }

        headers = { "Content-Type": "application/json" }
        req = client.build_request("POST", SUMMARIZATION_VLLM_URL, content=json.dumps(request_body), headers=headers)
        response = await client.send(req, stream=False)

        if response.status_code != 200:
            error_content_bytes = await response.aread()
            try:
                error_json = json.loads(error_content_bytes.decode('utf-8'))
                return JSONResponse(status_code=response.status_code, content=error_json)
            except Exception as e:
                log.error(f"Error parsing classification response: {e}", exc_info=True)
                return [False, []]
        response_json = response.json()
        choices = response_json.get("choices", [])
        raw_tool_calls = choices[0].get("message", {}).get("tool_calls", [])
        tool_calls = [ToolCall(**tool_call) for tool_call in raw_tool_calls]

        if len(tool_calls) > 0:
            return [True, tool_calls]
        else:
            return [False, []]
    except Exception as e:
        log.error(f"Error requesting classification: {e}", exc_info=True)
        return [False, []]

async def get_classification_prompt(content: str) -> str:
    """
    Get the classification prompt.
    """
    classification_prompt = await get_system_prompt(SUMMARIZATION_MODEL, "classification")
    if classification_prompt:
        return classification_prompt.format(content=content)
    return ""
