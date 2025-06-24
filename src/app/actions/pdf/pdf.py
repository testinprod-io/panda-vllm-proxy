import base64
import json
import asyncio
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi import HTTPException
import fitz
from typing import AsyncGenerator

from ...api.helper.request_llm import arequest_llm, get_user_collection_name
from ...api.helper.request_summary import call_summarization_llm
from ...api.helper.format_sse import format_sse_message, create_random_event_id
from ...logger import log
from ...api.v1.schemas import LLMRequest
from ...actions.pdf.utils import (
    get_multiple_pdf_base64_from_last_message,
    parse_text_from_pdf,
    parse_text_from_pdf_chunked,
    clean_message_of_pdf_urls,
    augment_messages_with_pdf,
)
from ...dependencies import get_milvus_wrapper
from ...config import get_settings

async def pdf_handler(payload: LLMRequest, user_id: str) -> StreamingResponse:
    """
    Handles requests containing PDF data, now expecting an LLMRequest object.
    Extracts text for RAG or converts to images.
    Processes only the single most recent PDF found in the latest N messages.
    """
    return StreamingResponse(pdf_stream(payload, user_id), media_type="text/event-stream")
    
async def pdf_stream(payload: LLMRequest, user_id: str) -> AsyncGenerator[str, None]:
    """
    Handles requests containing PDF data, now expecting an LLMRequest object.
    Extracts text for RAG or converts to images.
    Processes only the single most recent PDF found in the latest N messages.
    """
    log.info("Processing PDF request with LLMRequest payload.")

    pdf_base64_list = get_multiple_pdf_base64_from_last_message(payload.messages[-1])
    log.info(f"Found {len(pdf_base64_list)} PDF(s) in the last message.")
    if len(pdf_base64_list) == 0:
        log.warning("No PDF found in the last message.")
        raise ValueError("No PDF found in the last message.")

    try:
        # Exclude the PDF from the messages
        payload.messages[-1] = clean_message_of_pdf_urls(payload.messages[-1])

        yield format_sse_message(
            data={
                "object": "process.event",
                "id": create_random_event_id(),
                "type": "pdf",
                "message": "Parsing documents",
                "data": {},
            },
        )

        async def parse_single_pdf(i: int, pdf_base64_string: str):
            # Parse the PDF using PyMuPDF and RapidOCR in executor
            pdf_bytes = base64.b64decode(pdf_base64_string)

            pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            page_count = max(pdf_doc.page_count, 1)
            avg_page_size_kb = (len(pdf_bytes) / page_count) / 1024
            pdf_doc.close()

            settings = get_settings()
            needs_ocr = avg_page_size_kb > settings.PDF_PAGE_OCR_THRESHOLD_KB

            log.info(
                f"PDF {i+1}: avg_page_size={avg_page_size_kb:.1f}KB – needs_ocr={needs_ocr}"
            )

            loop = asyncio.get_running_loop()

            # Decide parsing approach based on *document* size to optimise
            # performance, but OCR decision is driven by `needs_ocr`.
            pdf_size_mb = len(pdf_bytes) / (1024 * 1024)

            if pdf_size_mb < settings.PDF_CHUNK_MODE_THRESHOLD_MB:
                log.info(f"PDF {i+1}: Using standard parsing mode")
                parse_func = lambda: parse_text_from_pdf(
                    pdf_bytes,
                    enable_ocr=needs_ocr,
                    max_pages=settings.PDF_MAX_PAGES,
                )
                return await loop.run_in_executor(None, parse_func)
            else:
                log.info(
                    f"PDF {i+1}: Using chunked parallel processing (chunk_size={settings.PDF_CHUNK_SIZE})"
                )
                parse_func = lambda: parse_text_from_pdf_chunked(
                    pdf_bytes,
                    chunk_size=settings.PDF_CHUNK_SIZE,
                    enable_ocr=needs_ocr,
                )
                return await loop.run_in_executor(None, parse_func)

        # Create tasks for all PDFs to process in parallel
        pdf_tasks = [
            parse_single_pdf(i, pdf_base64_string) 
            for i, pdf_base64_string in enumerate(pdf_base64_list)
        ]
        
        # Wait for all PDF parsing to complete in parallel
        docs_list = await asyncio.gather(*pdf_tasks)
        log.info(f"Completed parsing {len(docs_list)} PDFs in parallel.")

        # Save parsed results to vector DB
        user_collection_name = get_user_collection_name(user_id)
        from_doc_jobs = []
        milvus_instance = get_milvus_wrapper()
        for docs in docs_list:
            loop = asyncio.get_running_loop()
            from_doc_jobs.append(loop.run_in_executor(
                None, 
                milvus_instance.from_documents_for_user, 
                user_collection_name, 
                docs
            ))
        log.info(f"Started {len(from_doc_jobs)} jobs to save parsed PDF results to vector DB.")

        # Create background task to handle vector DB completion
        async def handle_vector_db_completion():
            try:
                await asyncio.gather(*from_doc_jobs)
                log.info(f"Successfully saved {len(docs_list)} PDF results to vector DB in background.")
            except Exception as e:
                log.error(f"Error saving PDF results to vector DB in background: {e}", exc_info=True)

        # Start vector DB operations in background
        asyncio.create_task(handle_vector_db_completion())
        log.info(f"Vector DB operations started in background, continuing with LLM request.")

        yield format_sse_message(
            data={
                "object": "process.event",
                "id": create_random_event_id(),
                "type": "pdf",
                "message": "Reading documents",
                "data": {},
            },
        )

        # Summarize the PDF
        parse_results_str = ""
        for docs in docs_list:
            parse_results_str += "\n\n".join([doc.page_content for doc in docs])
        parse_results_str = await call_summarization_llm(parse_results_str, 500)
        log.info(f"Summarized PDF with LLM.")

        yield format_sse_message(
            data="[RAG_DONE]"
        )

        # Augment the request with the parsed results
        augmented_request_dict = payload.model_dump(exclude_none=True) 
        original_messages_dicts = [msg.model_dump(exclude_none=True) for msg in payload.messages]
        
        augmented_request_dict["messages"] = await augment_messages_with_pdf(
            original_messages_dicts, parse_results_str
        )
        
        augmented_request_dict.pop("use_pdf", None)

        modified_request_body_json_str = json.dumps(augmented_request_dict)
        
        log.info(f"User sent request to LLM", extra={"user_id": user_id, "request_type": "pdf"})

        llm_response = await arequest_llm(modified_request_body_json_str, user_id=user_id, use_vector_db=True)
        if isinstance(llm_response, JSONResponse):
            raise ValueError("LLM response is not a StreamingResponse", llm_response)
        
        async for chunk in llm_response.aiter_text():
            yield chunk
    except Exception as e:
        log.error(f"An error occurred during PDF stream: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
