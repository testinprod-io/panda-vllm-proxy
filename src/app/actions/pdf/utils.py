import base64, io, json
from typing import List, Optional, Dict, Any, Tuple
from pypdf import PdfReader
from pdf2image import convert_from_bytes

from app.actions.models import ActionRequest
from app.logger import log
from app.config import get_settings

settings = get_settings()

BASE_MODEL = settings.MODEL_NAME
MULTI_MODAL_MODEL = settings.MULTI_MODAL_MODEL or BASE_MODEL
MAX_TEXT_LENGTH = 30000 # TODO: make it configurable for each model
PDF_IMAGE_FORMAT = "JPEG"

def _clean_message_of_pdf_urls(message_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Creates a new message dict with pdf_url content parts removed."""
    if not isinstance(message_dict, dict):
        return message_dict

    cleaned_message = message_dict.copy()
    original_content = cleaned_message.get("content")

    if isinstance(original_content, list):
        new_content_list = []
        for item in original_content:
            if isinstance(item, dict) and item.get("type") == "pdf_url":
                continue # Skip the pdf_url part
            new_content_list.append(item)
        cleaned_message["content"] = new_content_list
    return cleaned_message

def get_last_pdf_base64_from_lastest_messages(request: ActionRequest, threshold: int = 3) -> Optional[Tuple[str, int]]:
    """Extracts base64 PDF data and its original message's reverse slice index from the LATEST 'pdf_url' in the last N messages."""
    # reverse_slice_idx is 0 for a PDF in the last message of the slice, 1 for 2nd to last, etc.
    for reverse_slice_idx, message in enumerate(reversed(request["messages"][-threshold:])):
        if not (message and isinstance(message.get("content"), list)):
            continue

        for content_item_dict in message["content"]:
            if isinstance(content_item_dict, dict) and content_item_dict.get("type") == "pdf_url":
                pdf_url_dict = content_item_dict.get("pdf_url")
                if isinstance(pdf_url_dict, dict):
                    pdf_url_data = pdf_url_dict.get("url", "")
                    if pdf_url_data.startswith("data:application/pdf;base64,"):
                        # Found the latest PDF, return its data and index
                        return (pdf_url_data.split(",", 1)[1], reverse_slice_idx)
    return None # No PDF found in the threshold

def extract_text_from_pdf(pdf_bytes: bytes) -> Optional[str]:
    """Extracts text content from PDF bytes using PyPDF."""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip() if text else None
    except Exception as e:
        log.error(f"PyPDF error extracting text: {e}")
        return None

def convert_pdf_to_images_base64(pdf_bytes: bytes) -> List[str]:
    """Converts PDF bytes to a list of base64 encoded images (JPEG)."""
    images_base64 = []
    try:
        images = convert_from_bytes(pdf_bytes, fmt=PDF_IMAGE_FORMAT.lower())
        for img in images:
            buffered = io.BytesIO()
            img.save(buffered, format=PDF_IMAGE_FORMAT)
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            images_base64.append(f"data:image/{PDF_IMAGE_FORMAT.lower()};base64,{img_str}")
        log.info(f"Converted PDF to {len(images_base64)} images.")
        return images_base64
    except Exception as e:
        log.error(f"pdf2image error converting PDF: {e}")
        raise

def prepare_rag_body_for_pdf_message(
    original_request: ActionRequest, 
    pdf_text: str, 
    target_message_actual_index: int
) -> str:
    """Prepares a JSON request string for the RAG model.
    The PDF text is injected as a system message before the target user message.
    All messages are cleaned of pdf_url content parts.
    Does NOT modify original_request.
    """
    new_llm_messages = []
    for i, msg_dict in enumerate(original_request["messages"]):
        cleaned_msg = _clean_message_of_pdf_urls(msg_dict)
        if i == target_message_actual_index:
            system_prompt = {
                "role": "system",
                "content": f"You are a helpful assistant. Answer the user's query based ONLY on the text provided below, which was extracted from a PDF document.\n--- PDF TEXT START ---\n{pdf_text}\n--- PDF TEXT END ---"
            }
            new_llm_messages.append(system_prompt)
            new_llm_messages.append(cleaned_msg)
        else:
            new_llm_messages.append(cleaned_msg)

    rag_request_body = {
        "model": BASE_MODEL,
        "messages": new_llm_messages,
        "max_tokens": original_request.get("max_tokens"),
        "stream": original_request.get("stream", True)
    }
    return json.dumps(rag_request_body)

def prepare_multimodal_request(
    original_request: ActionRequest, 
    image_urls: List[str], 
    target_message_actual_index: int
) -> str:
    """Prepares the JSON request string for the multi-modal model.
    Associates images with the text from the target_message_actual_index.
    Currently does not include additional conversation history.
    """
    new_llm_messages = []
    for i, msg_dict in enumerate(original_request["messages"]):
        cleaned_msg = _clean_message_of_pdf_urls(msg_dict)
        if i == target_message_actual_index:
            cleaned_msg["content"].append({"type": "image_url", "image_url": {"url": image_urls[0]}})
        new_llm_messages.append(cleaned_msg)

    multimodal_request_body = {
        "model": MULTI_MODAL_MODEL,
        "messages": new_llm_messages,
        "max_tokens": original_request.get("max_tokens", 1024),
        "stream": original_request.get("stream", True)
    }
    return json.dumps(multimodal_request_body)