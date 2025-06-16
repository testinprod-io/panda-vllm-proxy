from fastapi import APIRouter, Depends, HTTPException
from typing import List

from ...api.helper.auth import verify_authorization_header
from ...config import get_settings
from ...logger import log
from .schemas import SummaryResponse, LLMRequest, TextContent, ChatMessage
from ...api.helper.request_summary import call_summarization_llm

settings = get_settings()
router = APIRouter(tags=["summary"])

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
        log.error(f"Internal server error processing summarization request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error processing summarization request.")
