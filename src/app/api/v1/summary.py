import json
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from typing import List
from ...api.helper.auth import verify_authorization_header
from ...api.helper.request_llm import request_llm
from ...config import get_settings
from ...logger import log
from .model import SummaryResponse, LLMRequest, TextContent, ChatMessage

settings = get_settings()
router = APIRouter(tags=["summary"])

SUMMARIZATION_MODEL = settings.SUMMARIZATION_MODEL or settings.MODEL_NAME

async def call_summarization_llm(text: str, max_tokens: int) -> str:
    """Calls the LLM to summarize the provided text."""
    prompt = f"Summarize the following text in approximately {max_tokens} words:\n\n{text}"
    
    request_body_dict: dict = {
        "model": SUMMARIZATION_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant specialized in summarizing text concisely."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "stream": False,
        "temperature": 0.2 # Lower temperature for more focused summary
    }

    try:
        response = await request_llm(json.dumps(request_body_dict), stream=False)
        
        if isinstance(response, JSONResponse):
            error_content = response.body.decode('utf-8') if hasattr(response, 'body') and isinstance(response.body, bytes) else str(response.body)
            log.error(f"LLM call failed for summarization: {error_content}")
            status_code = response.status_code if hasattr(response, 'status_code') else 500
            raise HTTPException(status_code=status_code, detail=f"LLM error: {error_content}")

        # TODO: make this better after applying Langchain wrapper
        if (isinstance(response, dict) and 
            response.get('choices') and 
            isinstance(response['choices'], list) and
            len(response['choices']) > 0 and 
            isinstance(response['choices'][0], dict) and 
            response['choices'][0].get('message') and 
            isinstance(response['choices'][0]['message'], dict) and
            response['choices'][0]['message'].get('content')):
            
            summary_text = response['choices'][0]['message']['content']
            return summary_text.strip()
        else:
            log.error(f"Unexpected LLM response format for summarization: {response}")
            raise HTTPException(status_code=500, detail="Unexpected response format from LLM during summarization.")

    except HTTPException as e:
        raise e
    except Exception as e:
        log.error(f"Error during summarization LLM call: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error during summarization: {str(e)}")
    
def extract_text_from_chat_messages(chat_messages: List[ChatMessage]) -> str:
    """Extracts and concatenates all text content from a list of ChatMessage objects."""
    all_text_parts = []
    for chat_message in chat_messages:
        for content_part in chat_message.content:
            if isinstance(content_part, TextContent):
                    all_text_parts.append(content_part.text)
    return "\n\n".join(all_text_parts)

@router.post("/summary", response_model=SummaryResponse, dependencies=[Depends(verify_authorization_header)])
async def summarize_messages(request: LLMRequest):
    """
    Summarizes text extracted from the provided list of LLMRequest objects.
    """
    if not request.messages:
        raise HTTPException(status_code=400, detail="No LLMRequest messages provided for summarization.")

    full_text = extract_text_from_chat_messages(request.messages)

    if not full_text.strip():
        raise HTTPException(status_code=400, detail="No text content found in the provided messages for summarization.")

    if len(full_text.split()) < 5: 
        log.warning("Attempting to summarize very short text.")

    try:
        summary = await call_summarization_llm(full_text, request.max_tokens or 1000)
        log.info("Successfully generated summary.")
        return SummaryResponse(summary=summary)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error processing summarization request.")
