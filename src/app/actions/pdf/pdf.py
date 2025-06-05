import base64
import json
import asyncio
from fastapi.responses import StreamingResponse, JSONResponse

from ...api.helper.request_llm import arequest_llm, get_user_collection_name
from ...api.helper.request_summary import call_summarization_llm
from ...api.helper.streaming import create_streaming_response
from ...logger import log
from ...api.v1.models import LLMRequest
from ...actions.pdf.utils import (
    get_multiple_pdf_base64_from_last_message,
    parse_text_from_pdf,
    clean_message_of_pdf_urls,
    augment_messages_with_pdf,
)
from ...dependencies import get_milvus_wrapper

async def pdf_handler(payload: LLMRequest, user_id: str) -> StreamingResponse:
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

    # Exclude the PDF from the messages
    payload.messages[-1] = clean_message_of_pdf_urls(payload.messages[-1])

    try:
        docs_list = []
        for pdf_base64_string in pdf_base64_list:
            # Parse the PDF using PyMuPDF and RapidOCR
            pdf_bytes = base64.b64decode(pdf_base64_string)
            docs_list.append(parse_text_from_pdf(pdf_bytes))

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

        # Summarize the PDF
        parse_results_str = ""
        for docs in docs_list:
            parse_results_str += "\n\n".join([doc.page_content for doc in docs])
        parse_results_str = await call_summarization_llm(parse_results_str, 5000)
        log.info(f"Summarized PDF: {parse_results_str}")

        # Augment the request with the parsed results
        augmented_request_dict = payload.model_dump(exclude_none=True) 
        original_messages_dicts = [msg.model_dump(exclude_none=True) for msg in payload.messages]
        
        augmented_request_dict["messages"] = augment_messages_with_pdf(
            original_messages_dicts, parse_results_str
        )
        
        modified_request_body_json_str = json.dumps(augmented_request_dict)
        log.debug(f"Augmented request body for LLM: {modified_request_body_json_str[:500]}...")
        
        # Wait for the from_doc_jobs to complete
        await asyncio.gather(*from_doc_jobs)    
        log.info(f"Successfully saved parsed PDF results to vector DB for user {user_id}.")

        llm_response = await arequest_llm(modified_request_body_json_str, user_id=user_id, use_vector_db=True)
        if isinstance(llm_response, JSONResponse):
            return llm_response
        
        return create_streaming_response(llm_response)
        
    except base64.binascii.Error as e:
        log.error(f"Invalid base64 data for PDF: {e}")
        return JSONResponse(status_code=400, content={"error": "Invalid base64 PDF data"})
    except ValueError as e:
        log.error(f"ValueError during PDF processing: {e}")
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        log.error(f"General error processing PDF: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": f"Error processing PDF: {str(e)}"})

