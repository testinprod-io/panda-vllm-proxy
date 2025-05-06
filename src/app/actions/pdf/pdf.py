import base64
from fastapi.responses import StreamingResponse, JSONResponse
from typing import Optional

from app.api.helper.request_llm import request_llm
from app.api.helper.streaming import create_streaming_response
from app.logger import log
from app.actions.models import ActionRequest
from app.actions.pdf.utils import (
    get_last_pdf_base64_from_lastest_messages,
    extract_text_from_pdf,
    convert_pdf_to_images_base64,
    prepare_rag_body_for_pdf_message,
    prepare_multimodal_request
)

async def pdf_handler(request: ActionRequest) -> StreamingResponse:
    """
    Handles requests containing PDF data. Extracts text for RAG or converts to images.
    Processes only the single most recent PDF found in the latest N messages.
    """
    log.info("Processing PDF request.")
    pdf_base64_tuple = get_last_pdf_base64_from_lastest_messages(request, request.get("threshold", 3))

    if not pdf_base64_tuple:
        log.error("PDF handler was called, but no valid PDF base64 data could be extracted.")
        return JSONResponse(
            status_code=400, 
            content={"error": "Invalid or missing PDF data in pdf_url content part."}
        )

    pdf_base64_string, reverse_slice_idx = pdf_base64_tuple

    final_llm_request_body_str: Optional[str] = None
    num_original_messages = len(request["messages"])

    try:
        pdf_bytes = base64.b64decode(pdf_base64_string)
        extracted_text = extract_text_from_pdf(pdf_bytes)

        target_message_actual_index = num_original_messages - 1 - reverse_slice_idx
        if not (0 <= target_message_actual_index < num_original_messages):
            log.error(f"Calculated invalid target_message_actual_index: {target_message_actual_index}")
            return JSONResponse(status_code=400, content={"error": f"Invalid target_message_actual_index: {target_message_actual_index}"})
        
        # TODO: decide the threshold for meaningful text
        if extracted_text and len(extracted_text) > 50:
            log.info(f"Extracted {len(extracted_text)} chars from PDF originating from original message index {target_message_actual_index}. Preparing RAG body.")
            final_llm_request_body_str = prepare_rag_body_for_pdf_message(
                request, extracted_text, target_message_actual_index
            )
        else:
            log.info(f"PDF text extraction failed/minimal (original msg index {target_message_actual_index}). Converting to images.")
            image_urls = convert_pdf_to_images_base64(pdf_bytes)
            if not image_urls:
                log.error(f"Failed to convert PDF (original msg index {target_message_actual_index}) to images.")
                return JSONResponse(status_code=500, content={"error": "Failed to process PDF as text or images"})
            
            final_llm_request_body_str = prepare_multimodal_request(
                request, image_urls, target_message_actual_index
            )
        should_stream = request.get("stream", True)
        log.info("Sending final processed request to LLM.")
        llm_response = await request_llm(final_llm_request_body_str, stream=should_stream)

        if isinstance(llm_response, JSONResponse):
            return llm_response
        
        if should_stream:
            return create_streaming_response(llm_response)
        else:
            return JSONResponse(content=llm_response)

    except base64.binascii.Error as e:
        log.error(f"Invalid base64 data for PDF: {e}")
        return JSONResponse(status_code=400, content={"error": "Invalid base64 PDF data"})
    except ValueError as e:
        log.error(f"ValueError during PDF processing: {e}")
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        log.error(f"General error processing PDF: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": f"Error processing PDF: {str(e)}"})

