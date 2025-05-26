from fastapi import APIRouter, Depends, HTTPException
from typing import List
from ...api.helper.auth import verify_authorization_header
from ...config import get_settings
from ...logger import log
from .models import SummaryResponse, LLMRequest, TextContent, ChatMessage
from ...rag.summarizing_llm import SummarizingLLM

settings = get_settings()
router = APIRouter(tags=["summary"])

SUMMARIZATION_MODEL = settings.SUMMARIZATION_MODEL or settings.MODEL_NAME
SUMMARIZATION_VLLM_URL = settings.SUMMARIZATION_VLLM_URL

async def call_summarization_llm(text: str, max_tokens: int) -> str:
    """Calls the LLM to summarize the provided text."""
    prompt = f"Summarize the following text in approximately {max_tokens} words:\n\n{text}"
    
    # Instantiate SummarizingLLM
    llm = SummarizingLLM(
        model=SUMMARIZATION_MODEL,
        vllm_url=SUMMARIZATION_VLLM_URL,
        max_tokens=max_tokens,
        temperature=0.2 # Temperature for more focused summary
    )

    try:
        summary_text = await llm.ainvoke(input=prompt)
        log.info(f"Summarization LLM response: {summary_text}")
        
        if not summary_text:
            log.error(f"LLM call returned empty summary for text: {text[:100]}...")
            raise HTTPException(status_code=500, detail="LLM returned an empty summary.")
            
        return summary_text.strip()

    except ValueError as ve:
        log.error(f"Error during summarization LLM call: {ve}")
        raise HTTPException(status_code=500, detail=f"LLM processing error: {ve}")
    except Exception as e:
        log.error(f"Unexpected error during summarization: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

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
