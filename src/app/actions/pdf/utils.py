from typing import List, Dict, Any
from langchain_community.document_loaders.parsers import PyMuPDFParser, RapidOCRBlobParser
from langchain_core.documents.base import Blob
from langchain_core.documents import Document

from ...api.v1.schemas import ChatMessage, ContentPart, PdfContent, SenderTypeEnum
from ...api.helper.get_system_prompt import get_system_prompt
from ...config import get_settings

def clean_message_of_pdf_urls(chat_message: ChatMessage) -> ChatMessage:
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

def get_multiple_pdf_base64_from_last_message(payload: ChatMessage) -> List[str]:
    """Extracts base64 PDF data from the LATEST 'PdfContent' in the last message."""
    if not payload.content:
        return []
    
    url_list = []
    for content_item_obj in payload.content:
        if isinstance(content_item_obj, PdfContent):
            pdf_url_data = content_item_obj.pdf_url
            url_string = pdf_url_data.url
            if url_string and url_string.startswith("data:application/pdf;base64,"):
                url_list.append(url_string.split(",", 1)[1])
            # Error if the format is unexpected
            else:
                raise ValueError("Unexpected PDF URL format.")
    return url_list

def parse_text_from_pdf(pdf_bytes: bytes) -> List[Document]:
    """Extracts text content from PDF bytes using PyPDF."""
    parser = PyMuPDFParser(
        mode="page",
        pages_delimiter = "\n\f",
        images_parser=RapidOCRBlobParser(),
        extract_tables="markdown"
    )
    pdf_blob = Blob.from_data(pdf_bytes)

    # Lazy parse the PDF blob, for memory efficiency
    docs_lazy = parser.lazy_parse(pdf_blob)
    docs = []
    for doc in docs_lazy:
        docs.append(doc)
    return docs

async def augment_messages_with_pdf(
    original_messages: List[Dict[str, Any]], 
    pdf_text: str
) -> List[Dict[str, Any]]:
    """Prepares a JSON request string for the RAG model using LLMRequest.
    The PDF text is injected as a system message before the target user message.
    All messages are cleaned of pdf_url content parts.
    """
    augmented_messages = original_messages
    pdf_prompt = await get_system_prompt(get_settings().SUMMARIZATION_MODEL, "pdf")
    if pdf_prompt:
        system_command = pdf_prompt.format(pdf_text=pdf_text)
        system_prompt_message = ChatMessage(role=SenderTypeEnum.SYSTEM, content=system_command).model_dump()
        last_msg_index = len(original_messages) - 1
        augmented_messages = (
            original_messages[:last_msg_index]
            + [system_prompt_message]
            + [original_messages[last_msg_index]]
        )

    return augmented_messages