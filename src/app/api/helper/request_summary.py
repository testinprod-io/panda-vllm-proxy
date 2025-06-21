from fastapi import HTTPException
from langchain_text_splitters import RecursiveCharacterTextSplitter
import asyncio

from ...config import get_settings
from ...logger import log
from ...rag.summarizing_llm import SummarizingLLM
from ...api.helper.get_system_prompt import get_system_prompt

settings = get_settings()

SUMMARIZATION_MODEL = settings.SUMMARIZATION_MODEL or settings.MODEL_NAME
SUMMARIZATION_VLLM_URL = settings.SUMMARIZATION_VLLM_URL

SUMMARIZATION_LLM_INPUT_CONTEXT_TOKENS = settings.SUMMARIZATION_LLM_INPUT_CONTEXT_TOKENS
PROMPT_OVERHEAD_TOKENS = 150
CHARS_PER_TOKEN_HEURISTIC = 3

MAX_TEXT_TOKENS_FOR_LLM = SUMMARIZATION_LLM_INPUT_CONTEXT_TOKENS - PROMPT_OVERHEAD_TOKENS
CHARACTER_CHUNK_SIZE = MAX_TEXT_TOKENS_FOR_LLM * CHARS_PER_TOKEN_HEURISTIC

SUMMARIZATION_CONCURRENCY_LIMIT = settings.SUMMARIZATION_CONCURRENCY_LIMIT

async def generate_request_prompt(text_to_summarize: str, target_word_count: int) -> str:
    """Generates the prompt for summarizing a piece of text."""
    summary_prompt = await get_system_prompt(SUMMARIZATION_MODEL, "summary")
    if summary_prompt:
        return summary_prompt.format(text_to_summarize=text_to_summarize, target_word_count=target_word_count)
    return ""

async def _summarize_single_chunk(chunk_text: str, target_word_count_for_chunk: int) -> str:
    """Helper function to summarize a single text chunk."""
    prompt_for_chunk = await generate_request_prompt(chunk_text, target_word_count_for_chunk)
    
    llm = SummarizingLLM(
        model=SUMMARIZATION_MODEL,
        vllm_url=SUMMARIZATION_VLLM_URL,
        max_tokens=target_word_count_for_chunk,
        temperature=0.2
    )
    try:
        summary_text = await llm.ainvoke(input=prompt_for_chunk)
        if not summary_text:
            log.error(f"LLM call returned empty summary for chunk")
            return ""
        return summary_text.strip()
    except Exception as e:
        log.error(f"Error summarizing chunk, Error: {e}", exc_info=True)
        return f"[Error summarizing chunk: {str(e)}]"

async def call_summarization_llm(text: str, max_tokens_for_final_summary: int) -> str:
    """
    Calls the LLM to summarize the provided text.
    If the text is too long, it's split into chunks, each chunk is summarized,
    and then the summaries are combined.
    `max_tokens_for_final_summary` refers to the desired word count for the final summary.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHARACTER_CHUNK_SIZE,
        chunk_overlap=50,
        length_function=len
    )
    
    text_chunks = text_splitter.split_text(text)
    log.info(f"Original text split into {len(text_chunks)} chunks for summarization.")

    if not text_chunks:
        log.warning("Text splitting resulted in no chunks. Returning empty summary.")
        return ""

    if len(text_chunks) == 1:
        return await _summarize_single_chunk(text_chunks[0], max_tokens_for_final_summary)
    else:
        # Distribute the final desired word count among chunks
        approx_words_per_chunk_summary = max(50, max_tokens_for_final_summary // len(text_chunks))

        # Process all chunks in parallel but cap concurrency with a semaphore
        log.info(f"Starting summarization of {len(text_chunks)} chunks (max concurrency {SUMMARIZATION_CONCURRENCY_LIMIT}) …")

        semaphore = asyncio.Semaphore(SUMMARIZATION_CONCURRENCY_LIMIT)

        async def summarize_with_limit(idx: int, chunk_text: str):
            async with semaphore:
                log.info(f"Summarizing chunk {idx+1}/{len(text_chunks)}")
                return await _summarize_single_chunk(chunk_text, approx_words_per_chunk_summary)

        tasks = [asyncio.create_task(summarize_with_limit(i, chunk)) for i, chunk in enumerate(text_chunks)]
        summarization_results = await asyncio.gather(*tasks)
        
        # Filter successful summaries and handle exceptions
        chunk_summaries = []
        for i, result in enumerate(summarization_results):
            if isinstance(result, Exception):
                log.error(f"Chunk {i+1} summarization failed with exception")
                continue
            elif result and not result.startswith("[Error"):
                chunk_summaries.append(result)
            else:
                log.warning(f"Chunk {i+1} summarization returned empty or error result")
        
        log.info(f"Completed summarization: {len(chunk_summaries)}/{len(text_chunks)} chunks successful")
        
        if not chunk_summaries:
            log.error("All chunk summarizations failed or returned empty.")
            raise HTTPException(status_code=500, detail="Failed to summarize any part of the text.")

        combined_summary = "\n\n---\n\n".join(chunk_summaries)

        # One simple approach if combined_summary is too verbose:
        if len(combined_summary.split()) > max_tokens_for_final_summary * 1.2: # If 20% over target
            # The max_tokens for this final pass should be the originally requested one.
            condensed_summary = await _summarize_single_chunk(combined_summary, max_tokens_for_final_summary)
            if condensed_summary and not condensed_summary.startswith("[Error"):
                return condensed_summary
            else:
                log.warning("Final condensation pass failed, returning combined summary of chunks.")
                return combined_summary

        return combined_summary
