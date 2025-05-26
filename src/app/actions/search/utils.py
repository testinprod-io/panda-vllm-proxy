from typing import List, Dict, Any, Optional
from rake_nltk import Rake

from ...logger import log
from ...config import get_settings
from ...rag.summarizing_llm import SummarizingLLM

settings = get_settings()
KEYWORD_EXTRACTION_MODEL = settings.SUMMARIZATION_MODEL or settings.MODEL_NAME
KEYWORD_EXTRACTION_VLLM_URL = settings.SUMMARIZATION_VLLM_URL

async def extract_keywords_llm(query: str, max_keywords: int = 5) -> List[str]:
    """Helper to extract keywords using LLM."""
    prompt = (
        f"Extract the top {max_keywords} most important keywords from the following query. "
        f"Return the keywords as a comma-separated list. For example, if the query is "
        f"'What are the latest advancements in AI for healthcare?', you should return "
        f"'AI, healthcare, latest advancements'.\n\nQuery: \"{query}\"\n\nKeywords:"
    )
    
    llm = SummarizingLLM(
        model=KEYWORD_EXTRACTION_MODEL,
        vllm_url=KEYWORD_EXTRACTION_VLLM_URL,
        max_tokens=max_keywords * 10,
        temperature=0.1 # Low temperature for deterministic keyword extraction
    )
    try:
        response_text = await llm.ainvoke(input=prompt)
        log.debug(f"LLM response for keyword extraction ('{query}'): {response_text}")
        if response_text:
            # Simple comma separation, stripping whitespace
            keywords = [kw.strip() for kw in response_text.split(',') if kw.strip()]
            # Remove potential quotes or list-like formatting from LLM
            cleaned_keywords = []
            for kw in keywords:
                kw = kw.strip('"''[]()')
                if kw:
                    cleaned_keywords.append(kw)
            
            if cleaned_keywords:
                log.info(f"LLM extracted keywords for '{query}': {cleaned_keywords[:max_keywords]}")
                return cleaned_keywords[:max_keywords]
    except Exception as e:
        log.error(f"Error during LLM keyword extraction for '{query}': {e}", exc_info=True)
    return []

def extract_keywords_rake(query: str, max_keywords: int = 5) -> List[str]:
    """Helper to extract keywords using RAKE."""
    r = Rake()
    r.extract_keywords_from_text(query)
    ranked_phrases_with_scores = r.get_ranked_phrases_with_scores()
    ranked_phrases = sorted(ranked_phrases_with_scores, key=lambda x: (-x[0], len(x[1])))
    
    keywords = [phrase for score, phrase in ranked_phrases if score >= 1.0]

    if keywords:
        return keywords[:max_keywords]
    log.warning(f"RAKE did not extract any keywords")
    return []


async def extract_keywords_from_query(query: str, max_keywords: int = 5) -> List[str]:
    """Extract keywords from a query using LLM. Fallback to RAKE if no keywords are found."""
    if not query.strip():
        return []

    llm_keywords = await extract_keywords_llm(query, max_keywords)
    if llm_keywords:
        return llm_keywords
    
    log.warning(f"LLM keyword extraction failed or returned no keywords. Falling back to RAKE.")
    rake_keywords = extract_keywords_rake(query, max_keywords)
    return rake_keywords

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