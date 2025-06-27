import json
import asyncio
from fastapi.responses import StreamingResponse, JSONResponse
from typing import AsyncGenerator

from ...logger import log
from ...api.helper.request_llm import arequest_llm, get_user_collection_name
from ...api.helper.request_summary import call_summarization_llm
from ...api.v1.schemas import LLMRequest
from ...rag import PandaWebRetriever
from ...dependencies import get_milvus_wrapper
from ...api.helper.format_sse import format_sse_message, create_random_event_id
from .utils import augment_messages_with_search
from .models import SearchToolArgs

async def search_handler(payload: LLMRequest, user_id: str, search_query_args: str) -> StreamingResponse:
    """
    Handle search functionality by augmenting the request with search results.
    """
    return StreamingResponse(search_stream(payload, user_id, search_query_args), media_type="text/event-stream")

async def search_stream(payload: LLMRequest, user_id: str, search_query_args: str) -> AsyncGenerator[str, None]:
    """
    Handle search functionality by augmenting the request with search results.
    """
    
    try:
        decoded_search_query_args = SearchToolArgs.model_validate_json(search_query_args)
    except Exception as e:
        log.error(f"Error validating search query args: {e}", exc_info=True)
        decoded_search_query_args = SearchToolArgs(query=search_query_args)
    
    try:
        yield format_sse_message(
            data={
                "object": "process.event",
                "id": create_random_event_id(),
                "type": "search",
                "message": "Brainstorming the search terms",
                "data": {},
            },
        )

        actual_search_query = decoded_search_query_args.query
        
        yield format_sse_message(
            data={
                "object": "process.event",
                "id": create_random_event_id(),
                "type": "search",
                "message": "",
                "data": {
                    "query": actual_search_query,
                },
            },
        )
        
        retriever = PandaWebRetriever()

        yield format_sse_message(
            data={
                "object": "process.event",
                "id": create_random_event_id(),
                "type": "search",
                "message": "Searching through URLs",
                "data": {}
            },
        )

        search_results = await retriever.ainvoke(actual_search_query)
        if not search_results:
            yield format_sse_message(
                data="[RAG_DONE]"
            )
            log.warning(f"No search results found for user {user_id}.")
            llm_response = await arequest_llm(payload.model_dump_json(exclude_none=True), user_id=user_id, use_vector_db=True)
            async for chunk in llm_response.aiter_text():
                yield chunk
            return

        # Yield the event
        yield format_sse_message(
            data={
                "object": "process.event",
                "id": create_random_event_id(),
                "type": "search",
                "message": "",
                "data": {
                    "urls": list(
                        set(result.metadata["source"] for result in search_results)
                    )
                },
            },
        )
        
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

        # Create background task to handle vector DB completion
        async def handle_vector_db_completion():
            try:
                await from_doc_job
                log.info(f"Successfully saved search results to vector DB in background.")
            except Exception as e:
                log.error(f"Error saving search results to vector DB in background: {e}", exc_info=True)
            
        # Start vector DB operation in background
        asyncio.create_task(handle_vector_db_completion())
        log.info(f"Vector DB operation started in background, continuing with LLM request.")

        yield format_sse_message(
            data={
                "object": "process.event",
                "id": create_random_event_id(),
                "type": "search",
                "message": "Analyzing the web pages",
                "data": {}
            },
        )

        # Summarize the search results with the LLM
        search_results_str = "\n\n".join([result.page_content for result in search_results])
        search_results_str = await call_summarization_llm(search_results_str, 1000)

        yield format_sse_message(
            data="[RAG_DONE]"
        )

        # Augment the request with the search results
        augmented_request_dict = payload.model_dump(exclude_none=True) 
        original_messages_dicts = [msg.model_dump(exclude_none=True) for msg in payload.messages]

        augmented_request_dict["messages"] = await augment_messages_with_search(
            original_messages_dicts, search_results_str
        )

        augmented_request_dict.pop("use_search", None)
        
        modified_request_body_json_str = json.dumps(augmented_request_dict)

        log.info(f"User sent request to LLM", extra={"user_id": user_id, "request_type": "search"})

        llm_response = await arequest_llm(modified_request_body_json_str, user_id=user_id, use_vector_db=True)
        if isinstance(llm_response, JSONResponse):
            raise ValueError("LLM response is not a StreamingResponse", llm_response)
        
        async for chunk in llm_response.aiter_text():
            yield chunk
    except Exception as e:
        log.error(f"An error occurred during search stream: {e}", exc_info=True)
        yield format_sse_message(
            data={
                "object": "process.event",
                "id": create_random_event_id(),
                "type": "search",
                "message": f"An error occurred during search stream: {e}",
                "data": {
                    "status_code": 500,
                }
            },
        )
        # Terminate the stream
        return

