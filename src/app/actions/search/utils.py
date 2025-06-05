from typing import List, Dict, Any, Optional
from datetime import datetime
from rake_nltk import Rake

from ...logger import log
from ...config import get_settings
from ...rag.summarizing_llm import SummarizingLLM
from ...prompts.prompts import (
    SEARCH_SYSTEM_PROMPT,
    SEARCH_SYSTEM_INFORMATION_PROMPT,
    EXTRACT_KEYWORDS_PROMPT
)

settings = get_settings()
KEYWORD_EXTRACTION_MODEL = settings.SUMMARIZATION_MODEL or settings.MODEL_NAME
KEYWORD_EXTRACTION_VLLM_URL = settings.SUMMARIZATION_VLLM_URL

async def extract_keywords_llm(query: str) -> List[str]:
    """Helper to extract keywords using LLM."""
    prompt = EXTRACT_KEYWORDS_PROMPT.format(query=query, current_date=datetime.now().strftime("%Y-%m-%d"))
    llm = SummarizingLLM(
        model=KEYWORD_EXTRACTION_MODEL,
        vllm_url=KEYWORD_EXTRACTION_VLLM_URL,
        max_tokens=100,
        temperature=0.1 # Low temperature for deterministic keyword extraction
    )
    try:
        response_text = await llm.ainvoke(input=prompt)
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
                return cleaned_keywords
    except Exception as e:
        log.error(f"Error during LLM keyword extraction: {e}", exc_info=True)
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


async def extract_keywords_from_query(query: str) -> List[str]:
    """Extract keywords from a query using LLM. Fallback to RAKE if no keywords are found."""
    if not query.strip():
        return []

    llm_keywords = await extract_keywords_llm(query)
    if llm_keywords:
        return llm_keywords
    
    log.warning(f"LLM keyword extraction failed or returned no keywords. Falling back to RAKE.")
    rake_keywords = extract_keywords_rake(query)
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
        "content": SEARCH_SYSTEM_PROMPT
    }
    system_information = {
        "role": "system",
        "name": "search_results",
        "content": SEARCH_SYSTEM_INFORMATION_PROMPT.format(search_results_str=search_results_str)
    }

    last_msg_index = len(original_messages) - 1
    augmented_messages = (
        original_messages[:last_msg_index]
        + [system_command] 
        + [system_information]
        + [original_messages[last_msg_index]]
    )

    return augmented_messages