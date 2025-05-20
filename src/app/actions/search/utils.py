import httpx, json
from typing import List, Dict, Any, Optional
from rake_nltk import Rake
from bs4 import BeautifulSoup

from ...logger import log
from ...api.helper.request_llm import arequest_llm
from ...config import get_settings
from .models import SearchResult

settings = get_settings()
DEFAULT_MODEL_NAME = settings.MODEL_NAME
SUMMARIZATION_MODEL = settings.SUMMARIZATION_MODEL or DEFAULT_MODEL_NAME
USER_AGENT = settings.USER_AGENT
SEARCH_TIMEOUT = settings.SEARCH_TIMEOUT

async def generate_reformulations(query: str, n: int) -> List[str]:
    """Generate n focused reformulations of the user's query (non-streaming)."""
    try:
        # TODO: fix this to call another vllm-proxy container or instance
        request_json = {
            "model": SUMMARIZATION_MODEL,
            "messages": [
                {"role": "system", "content": "You are a query reformulation assistant. Respond only with the queries, one per line, no preamble."},
                {"role": "user", "content": f"Generate {n} concise search queries, each on its own line, targeting this intent: \"{query}\""}
            ],
            "temperature": 0.5,
            "max_tokens": 150
        }

        request_body = json.dumps(request_json)

        # Call with stream=False
        response_data = await arequest_llm(request_body, stream=False)
        if response_data.status_code != 200:
            return [query] # Fallback

        # Check if response_data is the expected dict format
        if isinstance(response_data, dict) and 'choices' in response_data:
            try:
                content = response_data['choices'][0]['message']['content']
                lines = content.strip().splitlines()
                reformulations = [line.strip().lstrip("0123456789-.*• ") for line in lines if line.strip()]
                if not reformulations:
                    reformulations = [query]
                return reformulations[:n]
            except (IndexError, KeyError, TypeError) as e:
                log.error(f"Error generating reformulations: {e}")
                return [query] # Fallback
        else:
            log.error(f"Invalid response from LLM: {response_data}")
            return [query] # Fallback

    except Exception as e:
        log.error(f"Error generating reformulations: {e}")
        return [query]

def keyword_fallback(query: str) -> str:
    """Extract top keywords using RAKE from the original query."""
    try:
        rake = Rake()
        rake.extract_keywords_from_text(query)
        keywords = rake.get_ranked_phrases()[:5] or query.split()[:5]
        return " ".join(keywords)
    except Exception as e:
        log.error(f"Error extracting keywords: {e}")
        return query

def dedupe_results(results) -> List:
    """Simple deduplication by URL."""
    seen = set()
    deduped = []
    for r in results:
        if hasattr(r, 'url') and r.url not in seen:
            seen.add(r.url)
            deduped.append(r)
    return deduped

async def fetch_url_content(url: str) -> str:
    """Fetch and extract text content from a URL."""
    try:
        headers = {"User-Agent": USER_AGENT}
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, timeout=SEARCH_TIMEOUT, follow_redirects=True)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "html.parser")
            
            for script in soup(["script", "style"]):
                script.extract()
                
            text = soup.get_text(separator="\n")
            
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            text = "\n".join(lines)
            
            return text[:5000]
    except Exception as e:
        log.error(f"Error fetching URL {url}: {str(e)}")
        return ""

def chunk_text(text: str, chunk_size: int = 1000) -> List[str]:
    """Split text into chunks of up to chunk_size characters."""
    if not text:
        return []
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

async def summarize_url(query: str, result: SearchResult) -> SearchResult:
    """Summarize the contents of a URL, handling request_llm errors."""
    if not result or not hasattr(result, 'url') or not result.url or result.url.startswith("javascript:"):
        return result

    try:
        log.debug(f"Fetching content for summarization: {result.url}")
        content = await fetch_url_content(result.url)
        if not content or len(content) < 100:
            log.debug(f"Content too short or fetch failed for {result.url}, skipping summarization.")
            return result

        chunks = chunk_text(content)
        if not chunks:
            log.debug(f"Could not chunk content for {result.url}, skipping summarization.")
            return result

        # TODO: fix this to call another vllm-proxy container or instance
        # Create payload for summarization LLM
        request_json = {
            "model": SUMMARIZATION_MODEL,
            "messages": [
                {"role": "system", "content": f"You are a concise summarization assistant. Summarize the following text in 1-2 sentences, capturing the main points. The user's query is: \"{query}\""},
                {"role": "system", "content": f"Think about what user is asking for and what is relevant to the query. Summarize the text accordingly."},
                {"role": "user", "content": chunks[0][:8000]}  # limit size
            ],
            "temperature": 0.2, # Keep summaries factual
            "max_tokens": 100   # Limit summary length
        }
        request_body = json.dumps(request_json)

        response_data = await arequest_llm(request_body, stream=False)
        if response_data.status_code != 200:
            return result

        # Handle successful dict response
        if isinstance(response_data, dict) and 'choices' in response_data:
            try:
                summary = response_data['choices'][0]['message']['content']
                if summary:
                    original_snippet = getattr(result, 'snippet', '')
                    combined_snippet = f"{original_snippet}\n\nSummary: {summary.strip()}" if original_snippet else f"Summary: {summary.strip()}"
                    return SearchResult(
                        title=getattr(result, 'title', 'N/A'),
                        snippet=combined_snippet.strip(),
                        url=result.url
                    )
                else:
                    return result
            except (IndexError, KeyError, TypeError):
                return result
        else:
            log.error(f"Received unexpected data type from non-streaming arequest_llm for summarization: {type(response_data)}")
            return result

    except Exception:
        return result

def augment_messages_with_search(
    original_messages: List[Dict[str, Any]],
    search_results_str: Optional[str]
) -> List[Dict[str, Any]]:
    """Augments a message list with search results context."""
    if not search_results_str or not original_messages:
        # Return original if no results or no messages to augment
        return original_messages

    # TODO: do proper prompt engineering for this
    system_command = {
        "role": "system",
        "content": (
            "You are a helpful assistant who answers ONLY from the given search results.\\n"
            "If the results do not contain the answer, reply: \"I couldn't find that in the search results.\"\\n"
            "Based *only* on these search results and the user's query, provide a comprehensive answer. "
            "You must only use English for the whole answer, thinking, reasoning, and response."
        )
    }
    system_information = {
        "role": "system",
        "name": "search_results",
        "content": (
            f"Search results:\\n{search_results_str}\\n\\n"
            "Use these search results to inform your response."
        )
    }

    last_msg_index = len(original_messages) - 1
    augmented_messages = (
        original_messages[:last_msg_index]
        + [system_command] 
        + [system_information]
        + [original_messages[last_msg_index]]
    )

    return augmented_messages