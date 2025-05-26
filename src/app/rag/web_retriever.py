from typing import List, Optional, Any, Dict
import json
from pydantic import Field

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_text_splitters import TextSplitter, RecursiveCharacterTextSplitter
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from duckduckgo_search import DDGS
from langchain_community.utilities import BraveSearchWrapper
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer

from ..config import get_settings
from ..logger import log

class PandaWebRetriever(BaseRetriever):
    vector_store: Optional[Any] = None
    text_splitter: TextSplitter = Field(
        default_factory=lambda: RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=50),
        description="Text splitter for splitting web pages into chunks",
    )
    num_search_results: int = Field(
        default=1,
        description="Number of search results to return",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def initialize(
            cls,
            vector_store: Optional[Any] = None,
            num_search_results: int = 1,
            text_splitter: Optional[TextSplitter] = None,
            **kwargs,
        ) -> "PandaWebRetriever":
        instance_kwargs = {
            "vector_store": vector_store,
            "num_search_results": num_search_results,
            **kwargs,
        }
        if text_splitter is not None:
            instance_kwargs["text_splitter"] = text_splitter
        return cls(**instance_kwargs)
    
    def clean_search_query(self, query: str) -> str:
        if query and query[0].isdigit():
            first_quote_pos = query.find('"')
            if first_quote_pos != -1:
                query = query[first_quote_pos + 1 :]
                if query.endswith('"'):
                    query = query[:-1]
        return query.strip()

    def multi_search_result(self, query: str) -> List[Document]:
        query = self.clean_search_query(query)
        
        search_items: List[Dict[str, str]] = []
        search_items.extend(self.search_ddg(query))
        brave_results_list = self.search_brave(query)
        if brave_results_list:
             search_items.extend(brave_results_list)
        
        log.debug(f"Combined search items from DDG and Brave: {search_items}")

        url_to_look = []
        for res in search_items:
            if isinstance(res, dict) and res.get("link"):
                url_to_look.append(res["link"])
            else:
                log.warning(f"Search result item skipped (not a dict or no link): {res}")
        
        if not url_to_look:
            log.warning(f"No URLs found to load")
            return []

        log.info(f"Attempting to load {len(url_to_look)} URLs")
        loader = AsyncHtmlLoader(
            url_to_look, 
            ignore_load_errors=True,
            requests_kwargs={
                # To avoid 400 error due to cookie size
                "max_line_size": 16384,
                "max_field_size": 16384,
            }
        )
        html2text = Html2TextTransformer()
        
        docs = loader.load()
        if not docs:
            log.warning(f"AsyncHtmlLoader returned no documents for URLs: {url_to_look}")
            return []
        docs = list(html2text.transform_documents(docs))
        docs = self.text_splitter.split_documents(docs)
            
        log.info(f"Processed documents: {len(docs)} documents")
        return docs
    
    def search_ddg(self, query: str) -> List[Dict[str, str]]:
        try:
            with DDGS() as ddgs:
                ddg_results = ddgs.text(query, max_results=self.num_search_results)
            formatted_results = [
                {
                    "title": r.get("title", ""),
                    "link": r.get("href", ""),
                    "snippet": r.get("body", "")
                }
                for r in ddg_results if r.get("href")
            ]
            return formatted_results
        except Exception as e:
            log.error(f"Error during DuckDuckGo search: {e}", exc_info=True)
            return []
        
    def search_brave(self, query: str) -> Optional[List[Dict[str, str]]]:
        brave_api_key = get_settings().BRAVE_SEARCH_API_KEY
        if not brave_api_key:
            log.warning("Brave Search API key not configured or empty. Skipping Brave search.")
            return []

        try:
            wrapper = BraveSearchWrapper(
                api_key=brave_api_key,
                search_kwargs={
                    "count": self.num_search_results
                }
            )
            brave_search_output_str = wrapper.run(query=query)

            if not isinstance(brave_search_output_str, str):
                if isinstance(brave_search_output_str, list) and all(isinstance(item, dict) for item in brave_search_output_str):
                    log.debug(f"Brave search returned structured data directly: {brave_search_output_str}")
                    return [r for r in brave_search_output_str if r.get("link")]
                else:
                    log.error(f"Brave search returned unexpected type: {type(brave_search_output_str)}. Output: {brave_search_output_str}")
                    return []

            # Format the output to be a list of dictionaries
            parsed_results = json.loads(brave_search_output_str)
            
            if not isinstance(parsed_results, list):
                log.error(f"Brave search: Parsed JSON is not a list. Got: {type(parsed_results)}")
                return []

            formatted_results = []
            for r_item in parsed_results:
                if isinstance(r_item, dict):
                    formatted_results.append({
                        "title": r_item.get("title", ""),
                        "link": r_item.get("link", ""),
                        "snippet": r_item.get("snippet", "")
                    })
                else:
                    log.warning(f"Skipping non-dict item in Brave search results: {r_item}")
            
            return [r for r in formatted_results if r.get("link")]

        except json.JSONDecodeError as e:
            log.error(f"Error decoding JSON from Brave search: {e}. Raw response: {brave_search_output_str}", exc_info=True)
            return []
        except Exception as e:
            log.error(f"Error during Brave search: {e}", exc_info=True)
            return []

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        search_results_docs = self.multi_search_result(query)
        return search_results_docs

    