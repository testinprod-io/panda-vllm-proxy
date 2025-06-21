import concurrent.futures
from typing import List, Dict, Any, Optional
from langchain_community.document_loaders.parsers import PyMuPDFParser, RapidOCRBlobParser
from langchain_core.documents.base import Blob
from langchain_core.documents import Document
import fitz

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

def parse_text_from_pdf(pdf_bytes: bytes, enable_ocr: bool = True, max_pages: Optional[int] = None) -> List[Document]:
    """
    Extracts text content from PDF bytes using PyMuPDF with optimized settings.
    
    Args:
        pdf_bytes: PDF file bytes
        enable_ocr: Whether to enable OCR for images (slower but more accurate)
        max_pages: Maximum number of pages to process (None for all pages)
    """
    images_parser = RapidOCRBlobParser() if enable_ocr else None
    
    parser = PyMuPDFParser(
        mode="page",
        pages_delimiter="\n\f",
        images_parser=images_parser,
        extract_tables="markdown" if enable_ocr else None,
        extract_images=enable_ocr
    )
    
    pdf_blob = Blob.from_data(pdf_bytes)
    
    # Lazy parse the PDF blob for memory efficiency
    docs_lazy = parser.lazy_parse(pdf_blob)
    docs = []
    
    # Process pages with optional limit
    for i, doc in enumerate(docs_lazy):
        if max_pages and i >= max_pages:
            break
        docs.append(doc)
    return docs

def parse_text_from_pdf_chunked(pdf_bytes: bytes, *, chunk_size: int = 10, enable_ocr: bool = True) -> List[Document]:
    """
    Parse PDF in chunks using parallel processing for large PDFs.
    This variant allows selectively enabling or disabling OCR based on
    heuristics computed outside of this function.

    Args:
        pdf_bytes: PDF file bytes
        chunk_size: Number of pages to process per chunk
        enable_ocr: Whether OCR should be enabled when parsing each chunk
    """
    pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = pdf_doc.page_count
    pdf_doc.close()
    
    if total_pages <= chunk_size:
        return parse_text_from_pdf(pdf_bytes, enable_ocr=enable_ocr)
    # Large PDF, process in parallel chunks
    def process_chunk(start_page: int, end_page: int) -> List[Document]:
        # Create a new PDF with only the chunk pages
        chunk_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        new_doc = fitz.open()
        
        for page_num in range(start_page, min(end_page, total_pages)):
            new_doc.insert_pdf(chunk_doc, from_page=page_num, to_page=page_num)
        
        chunk_bytes = new_doc.write()
        chunk_doc.close()
        new_doc.close()
        
        return parse_text_from_pdf(chunk_bytes, enable_ocr=enable_ocr)
    
    # Create chunks
    chunks = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=get_settings().PDF_CHUNK_CONCURRENCY_LIMIT) as executor:
        futures = []
        for start in range(0, total_pages, chunk_size):
            end = min(start + chunk_size, total_pages)
            future = executor.submit(process_chunk, start, end)
            futures.append(future)
        
        # Collect results
        for future in concurrent.futures.as_completed(futures):
            chunk_docs = future.result()
            chunks.extend(chunk_docs)
    
    return chunks

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