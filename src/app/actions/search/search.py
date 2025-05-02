import json
import asyncio
from typing import Dict, Any, List, Optional
from fastapi.responses import StreamingResponse, JSONResponse
from duckduckgo_search import DDGS

from ...logger import log
from ...api.helper.request_llm import request_llm
from ...api.helper.streaming import create_streaming_response
from .utils import (
    generate_reformulations, keyword_fallback, dedupe_results,
    summarize_url, augment_messages_with_search
)
from ...config.constants import MAX_RESULTS
from .models import SearchRequest, SearchResult, SearchResponse

async def search_handler(request: Dict[str, Any]) -> StreamingResponse:
    """
    Handle search functionality by augmenting the request with search results.
    
    Args:
        request: The validated request containing messages and search configuration
        
    Returns:
        StreamingResponse: A streaming response with search-augmented generation
    """
    # Validate required fields
    if "messages" not in request:
        raise ValueError("Messages field is required for search")
    
    # Extract the search query from the last user message
    user_messages = [msg for msg in request["messages"] if msg.get("role") == "user"]
    if not user_messages:
        raise ValueError("No user message found in the request")
    
    last_user_message = user_messages[-1]["content"]
    log.info(f"Search query: {last_user_message}")
    
    # Perform search
    search_results_str: Optional[str] = await perform_search(last_user_message)
    
    # TODO: add retry logic if no results

    augmented_request = request.copy()
    original_messages = augmented_request.get("messages", [])

    # Augment messages
    augmented_request["messages"] = augment_messages_with_search(
        original_messages, search_results_str
    )

    # Remove the search flag to prevent recursion
    augmented_request.pop("use_search", None)
    
    modified_request_body = json.dumps(augmented_request)
    log.debug(f"Augmented request body for LLM: {modified_request_body[:500]}...")
    
    # Call the final LLM for generation
    response = await request_llm(modified_request_body)
    if isinstance(response, JSONResponse):
        return response

    return create_streaming_response(response)

async def perform_search(query: str) -> Optional[str]:
    """
    Perform a search operation based on the query using multi-query search.
    
    Args:
        query: The search query string
        
    Returns:
        Optional[str]: Formatted search results as a string, or None if no results/error
    """
    try:
        req = SearchRequest(query=query)
        resp = await multi_query_search(req)
        if not resp or not resp.results:
            log.warning(f"No search results found for: {query}")
            return None
        
        # Format results as a string
        formatted_results = "\n\n".join([
            f"Title: {result.title}\nSnippet: {result.snippet}\nURL: {result.url}"
            for result in resp.results[:5]  # Limit to top 5 results
        ])
        
        if resp.fallback_query:
            formatted_results = f"Search using fallback query: {resp.fallback_query}\n\n" + formatted_results
            
        return formatted_results
    except Exception as e:
        log.error(f"Error during search for '{query}': {str(e)}", exc_info=True)
        return None

async def duckduckgo_search(query: str, max_results: int = MAX_RESULTS) -> List[SearchResult]:
    """
    Perform a DuckDuckGo search and return structured results.
    """
    try:
        def _search():
            try:
                with DDGS() as ddgs:
                    results = list(ddgs.text(query, max_results=max_results))
                return results
            except Exception as e:
                log.error(f"DuckDuckGo search error: {str(e)}")
                return []

        raw_results = await asyncio.to_thread(_search)
        if not raw_results:
            log.info(f"No search results found for: {query}")
            return []
        
        # Convert to SearchResult objects
        return [
            SearchResult(
                title=item.get("title", "No title"),
                snippet=item.get("body", "No description"),
                url=item.get("href", "")
            )
            for item in raw_results
        ]
    except Exception as e:
        log.error(f"Error in DuckDuckGo search: {str(e)}")
        return []

async def multi_query_search(req: SearchRequest) -> SearchResponse:
    """
    Perform a multi-query search with reformulations and fallbacks.
    
    Args:
        req: Search request parameters
        
    Returns:
        SearchResponse with results and metadata
    """
    try:
        # Generate reformulations
        reformulations = await generate_reformulations(req.query, req.n_reformulations)
        log.info(f"Generated reformulations: {reformulations}")
        # Run searches in parallel
        search_tasks = [duckduckgo_search(q, req.fallback_threshold) for q in reformulations]
        search_results = await asyncio.gather(*search_tasks)
        log.info(f"Search results: {search_results}")
        # Flatten and dedupe results
        aggregated = []
        for results in search_results:
            aggregated.extend(results)
        aggregated = dedupe_results(aggregated)

        # Keyword fallback if too few results
        fallback_q = None
        if len(aggregated) < req.fallback_threshold:
            fallback_q = keyword_fallback(req.query)
            log.info(f"Using fallback query: {fallback_q}")
            fallback_results = await duckduckgo_search(fallback_q, req.fallback_threshold)
            aggregated.extend(fallback_results)
            aggregated = dedupe_results(aggregated)
        
        # Try to summarize top results
        try:
            # Only summarize top results to avoid timeouts
            top_results = aggregated[:3]
            summarize_tasks = [summarize_url(req.query, result) for result in top_results]
            summarized = await asyncio.gather(*summarize_tasks)
            log.info(f"Summarized: {summarized}")
            # Replace top results with summarized versions
            for i, result in enumerate(summarized):
                if i < len(aggregated):
                    log.info(f"Replacing result: {aggregated[i]} with: {result}")
                    aggregated[i] = result
        except Exception as e:
            log.error(f"Error summarizing results: {str(e)}")
        
        return SearchResponse(
            reformulations=reformulations,
            results=aggregated,
            fallback_query=fallback_q
        )
    except Exception as e:
        log.error(f"Error in multi-query search: {str(e)}")
        # Return minimal response with original query
        return SearchResponse(
            reformulations=[req.query],
            results=[],
            fallback_query=None
        )
