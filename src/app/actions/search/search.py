import json
import asyncio
from fastapi.responses import StreamingResponse, JSONResponse

from ...logger import log
from ...api.helper.request_llm import arequest_llm, get_user_collection_name
from ...api.helper.request_summary import call_summarization_llm
from ...api.helper.streaming import create_streaming_response
from .utils import augment_messages_with_search, extract_keywords_from_query
from ...api.v1.models import LLMRequest, TextContent, SenderTypeEnum 
from ...rag import PandaWebRetriever
from ...dependencies import get_milvus_wrapper

async def search_handler(payload: LLMRequest, user_id: str) -> StreamingResponse:
    """
    Handle search functionality by augmenting the request with search results.
    
    Args:
        payload: The validated LLMRequest object
        
    Returns:
        StreamingResponse: A streaming response with search-augmented generation
    """
    # Extract the search query text from the last user message
    user_chat_messages = [msg for msg in payload.messages if msg.role == SenderTypeEnum.USER.value or msg.role == "user"]
    if not user_chat_messages:
        log.warning("No user message found in LLMRequest for search.")
        raise ValueError("No user message found in the request for search action.")
    
    # Extract text from content parts
    last_user_chat_message = user_chat_messages[-1]
    query_text_parts = []
    for content_part in last_user_chat_message.content:
        if isinstance(content_part, TextContent):
            query_text_parts.append(content_part.text)
    
    if not query_text_parts:
        log.warning("No text content found in the last user message for search.")
        raise ValueError("No text content found in the last user message for search action.")

    actual_search_query = " ".join(query_text_parts)

    # Extract keywords from the search query
    keywords = await extract_keywords_from_query(actual_search_query)
    keywords = " ".join(keyword for keyword in keywords if keyword)

    # Retrieve web search results in the form of Document objects
    retriever = PandaWebRetriever()
    search_results = await retriever.ainvoke(keywords)
    if not search_results:
        log.warning(f"No search results found for user {user_id}.")
        llm_response = await arequest_llm(payload.model_dump(exclude_none=True), user_id=user_id, use_vector_db=True)
        if isinstance(llm_response, JSONResponse):
            return llm_response
        return create_streaming_response(llm_response)

    # Save search results to vector DB
    user_collection_name = get_user_collection_name(user_id)
    milvus_instance = get_milvus_wrapper()
    
    # Run the milvus operation
    loop = asyncio.get_running_loop()
    from_doc_job = loop.run_in_executor(
        None, 
        milvus_instance.from_documents_for_user, 
        user_collection_name, 
        search_results
    )

    # Summarize the search results with the LLM
    search_results_str = "\n\n".join([result.page_content for result in search_results])
    search_results_str = await call_summarization_llm(search_results_str, 1000)

    # Augment the request with the search results
    augmented_request_dict = payload.model_dump(exclude_none=True) 
    original_messages_dicts = [msg.model_dump(exclude_none=True) for msg in payload.messages]

    augmented_request_dict["messages"] = augment_messages_with_search(
        original_messages_dicts, search_results_str
    )

    augmented_request_dict.pop("use_search", None)
    
    modified_request_body_json_str = json.dumps(augmented_request_dict)

    # Wait for the from_doc_job to complete
    await from_doc_job
    log.info(f"Successfully saved search results to vector DB.")

    log.info(f"User sent request to LLM", extra={"user_id": user_id, "request_type": "search"})

    llm_response = await arequest_llm(modified_request_body_json_str, user_id=user_id, use_vector_db=True)
    if isinstance(llm_response, JSONResponse):
        return llm_response

    return create_streaming_response(llm_response)
