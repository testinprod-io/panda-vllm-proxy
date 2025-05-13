import base64, io, json
from typing import List, Optional, Dict, Any, Tuple
from pypdf import PdfReader
from pdf2image import convert_from_bytes

from app.api.v1.model import LLMRequest, ChatMessage, ContentPart, TextContent, PdfContent, ImageContent, Url, SenderTypeEnum
from app.logger import log
from app.config import get_settings

settings = get_settings()

BASE_MODEL = settings.MODEL_NAME
MULTI_MODAL_MODEL = settings.MULTI_MODAL_MODEL or BASE_MODEL
MAX_TEXT_LENGTH = 30000 # TODO: make it configurable for each model
PDF_IMAGE_FORMAT = "JPEG"

def _clean_message_of_pdf_urls(chat_message: ChatMessage) -> ChatMessage:
    """Creates a new ChatMessage instance with pdf_url content parts removed."""
    if not chat_message.content:
        return chat_message

    new_content_list: List[ContentPart] = []
    for part in chat_message.content:
        if isinstance(part, PdfContent):
            continue
        new_content_list.append(part)
    
    # Create a new ChatMessage with the cleaned content, preserving other fields
    return ChatMessage(role=chat_message.role, content=new_content_list)

def get_last_pdf_base64_from_lastest_messages(payload: LLMRequest, threshold: int = 3) -> Optional[Tuple[str, int]]:
    """Extracts base64 PDF data and its original message's reverse slice index from the LATEST 'PdfContent' in the last N messages."""
    # reverse_slice_idx is 0 for a PDF in the last message of the slice, 1 for 2nd to last, etc.
    messages_to_check = payload.messages[-threshold:]
    for reverse_slice_idx, message_obj in enumerate(reversed(messages_to_check)):
        if not message_obj.content:
            continue

        for content_item_obj in message_obj.content:
            if isinstance(content_item_obj, PdfContent):
                pdf_url_data = content_item_obj.pdf_url
                actual_url_string = pdf_url_data.url
                if actual_url_string and actual_url_string.startswith("data:application/pdf;base64,"):
                    return (actual_url_string.split(",", 1)[1], reverse_slice_idx)
    return None

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
    original_payload: LLMRequest, 
    pdf_text: str, 
    target_message_actual_index: int
) -> str:
    """Prepares a JSON request string for the RAG model using LLMRequest.
    The PDF text is injected as a system message before the target user message.
    All messages are cleaned of pdf_url content parts.
    """
    new_llm_chat_messages: List[Dict[str, Any]] = []
    for i, msg_obj in enumerate(original_payload.messages):
        cleaned_msg_obj = _clean_message_of_pdf_urls(msg_obj)
        if i == target_message_actual_index:
            system_prompt_content = [
                TextContent(type="text", text=f"You are a helpful assistant. Answer the user's query based ONLY on the text provided below, which was extracted from a PDF document.\n--- PDF TEXT START ---\n{pdf_text}\n--- PDF TEXT END ---").model_dump()
            ]
            system_prompt_message = ChatMessage(role=SenderTypeEnum.SYSTEM, content=system_prompt_content).model_dump()
            new_llm_chat_messages.append(system_prompt_message)
        new_llm_chat_messages.append(cleaned_msg_obj.model_dump())

    rag_request_body_dict = {
        "model": BASE_MODEL,
        "messages": new_llm_chat_messages,
        "max_tokens": original_payload.max_tokens,
        "stream": original_payload.stream,
        "temperature": original_payload.temperature
        # Remove use_pdf, use_search etc. as they are action flags
    }
    return json.dumps(rag_request_body_dict)

def prepare_multimodal_request(
    original_payload: LLMRequest, 
    image_urls: List[str], 
    target_message_actual_index: int
) -> str:
    """Prepares the JSON request string for the multi-modal model using LLMRequest.
    Associates images with the text from the target_message_actual_index.
    """
    new_llm_messages_dicts: List[Dict[str, Any]] = []
    for i, msg_obj in enumerate(original_payload.messages):
        cleaned_msg_obj = _clean_message_of_pdf_urls(msg_obj)
        msg_dict = cleaned_msg_obj.model_dump()

        if i == target_message_actual_index:
            if not isinstance(msg_dict.get("content"), list):
                 msg_dict["content"] = [msg_dict.get("content")] if msg_dict.get("content") else [] # Normalize if single item
            
            for img_url_str in image_urls:
                image_content_part = ImageContent(type="image_url", image_url=Url(url=img_url_str)).model_dump()
                msg_dict["content"].append(image_content_part)
        
        new_llm_messages_dicts.append(msg_dict)

    multimodal_request_body_dict = {
        "model": MULTI_MODAL_MODEL,
        "messages": new_llm_messages_dicts,
        "max_tokens": original_payload.max_tokens,
        "stream": original_payload.stream,
        "temperature": original_payload.temperature
    }
    return json.dumps(multimodal_request_body_dict)