import json
from typing import Any, List, Mapping, Optional, Dict, Iterator, Type

from langchain.callbacks.manager import (
    CallbackManagerForLLMRun,
    AsyncCallbackManagerForLLMRun,
)
from langchain.llms.base import LLM
from langchain_core.outputs.generation import GenerationChunk
from langchain_core.messages import (
    BaseMessageChunk,
    HumanMessageChunk,
    SystemMessageChunk,
    AIMessageChunk,
    ChatMessageChunk,
)
from langchain_core.outputs import ChatGenerationChunk
from pydantic import Field
from ..api.helper.request_llm import request_llm 
from ..api.v1.models import LLMSuccessResponse
from ..logger import log
from fastapi.responses import JSONResponse

class SummarizingLLM(LLM):
    """LangChain LLM instance for a VLLM endpoint, with reasoning capabilities."""

    model_name: str = Field(..., alias="model")
    temperature: float = 0.2
    max_tokens: int = 2048
    vllm_url: str | None = None

    @property
    def _llm_type(self) -> str:
        return "summarizing_llm"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
    
    def _generate(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Run the LLM on the given prompt."""
        return self._call(prompt, stop, run_manager, **kwargs)
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Run the LLM on the given prompt."""

        params = self._identifying_params
        params.update(kwargs)
        if stop:
            params["stop"] = stop

        request_body_dict = {
            "model": params["model"],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": params["temperature"],
            "max_tokens": params["max_tokens"],
            "stream": False,
        }
        request_body_dict = {k: v for k, v in request_body_dict.items() if v is not None}

        request_body_str = json.dumps(request_body_dict)
        response_data = request_llm(request_body=request_body_str, stream=False, vllm_url=self.vllm_url)
        if isinstance(response_data, JSONResponse):
            try:
                error_detail = response_data.body.decode()
            except Exception:
                error_detail = str(response_data.body)
            log.error(f"Error from vLLM ({response_data.status_code}): {error_detail}")
            raise ValueError(f"Error from vLLM ({response_data.status_code}): {error_detail}")
        
        try:
            if isinstance(response_data, LLMSuccessResponse):
                parsed_response = response_data
            elif isinstance(response_data, dict):
                parsed_response = LLMSuccessResponse(**response_data)
            else:
                log.error(f"Unexpected response type from request_llm: {type(response_data)}")
                raise ValueError("Unexpected response type from request_llm.")

            if parsed_response.choices and parsed_response.choices[0].message and parsed_response.choices[0].message.content:
                generated_text = parsed_response.choices[0].message.content
                if run_manager:
                    run_manager.on_llm_new_token(generated_text)
                return generated_text
            else:
                log.error(f"Could not extract content from vLLM response. Choices: {parsed_response.choices}")
                raise ValueError("Could not extract content from vLLM response.")
        except (AttributeError, IndexError, KeyError, TypeError) as e:
            log.error(f"Error parsing LLMSuccessResponse: {e}. Response data: {response_data}")
            raise ValueError(f"Error parsing successful vLLM response: {e}")

    def _stream(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        message_dicts, params = self._create_message_dicts(prompt, stop)
        params = {**params, **kwargs, "stream": True}

        default_chunk_class = SystemMessageChunk
        for chunk in self.completion_with_retry(
            messages=message_dicts, run_manager=run_manager, **params
        ):
            if not isinstance(chunk, dict):
                chunk = chunk.dict()
            if len(chunk["choices"]) == 0:
                continue
            choice = chunk["choices"][0]
            if choice["delta"] is None:
                continue
            chunk = _convert_delta_to_message_chunk(
                choice["delta"], default_chunk_class
            )
            finish_reason = choice.get("finish_reason")
            generation_info = (
                dict(finish_reason=finish_reason) if finish_reason is not None else None
            )
            default_chunk_class = chunk.__class__
            cg_chunk = ChatGenerationChunk(
                message=chunk, generation_info=generation_info
            )
            if run_manager:
                run_manager.on_llm_new_token(cg_chunk.text, chunk=cg_chunk)
            yield cg_chunk

def _convert_delta_to_message_chunk(
    _dict: Mapping[str, Any], default_class: Type[BaseMessageChunk]
) -> BaseMessageChunk:
    role = _dict.get("role")
    content = _dict.get("content") or ""
    additional_kwargs: Dict = {}

    if role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content)
    elif role == "assistant" or default_class == AIMessageChunk:
        return AIMessageChunk(content=content, additional_kwargs=additional_kwargs)
    elif role == "system" or default_class == SystemMessageChunk:
        return SystemMessageChunk(content=content)
    elif role or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role)  # type: ignore[arg-type]
    else:
        return default_class(content=content)  # type: ignore[call-arg]
        