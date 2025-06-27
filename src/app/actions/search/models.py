from pydantic import BaseModel

class SearchResult(BaseModel):
    title: str
    snippet: str
    url: str

class SearchToolArgs(BaseModel):
    query: str