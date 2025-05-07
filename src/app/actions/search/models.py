from pydantic import BaseModel
from typing import List, Optional

class SearchRequest(BaseModel):
    query: str
    n_reformulations: int = 3
    fallback_threshold: int = 5  # if fewer than this total hits, trigger keyword fallback

class SearchResult(BaseModel):
    title: str
    snippet: str
    url: str

class SearchResponse(BaseModel):
    reformulations: List[str]
    results: List[SearchResult]
    fallback_query: Optional[str] = None