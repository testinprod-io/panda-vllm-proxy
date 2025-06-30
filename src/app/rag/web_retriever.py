from typing import List, Optional, Any, Dict
import json
from pydantic import Field
import asyncio
import trafilatura

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_text_splitters import TextSplitter, RecursiveCharacterTextSplitter
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from duckduckgo_search import DDGS
from langchain_community.utilities import BraveSearchWrapper
from langchain_community.document_loaders import AsyncHtmlLoader

from ..config import get_settings
from ..logger import log

class PandaWebRetriever(BaseRetriever):
    vector_store: Optional[Any] = None
    text_splitter: TextSplitter = Field(
        default_factory=lambda: RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=50),
        description="Text splitter for splitting web pages into chunks",
    )
    num_search_results: int = Field(
        default=2,
        description="Number of search results to return",
    )
    max_urls_to_process: int = Field(
        default=5,
        description="Maximum number of URLs to process to control latency",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def initialize(
        cls,
        vector_store: Optional[Any] = None,
        num_search_results: int = 1,
        max_urls_to_process: int = 5,
        text_splitter: Optional[TextSplitter] = None,
        **kwargs,
    ) -> "PandaWebRetriever":
        instance_kwargs = {
            "vector_store": vector_store,
            "num_search_results": num_search_results,
            "max_urls_to_process": max_urls_to_process,
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

    async def multi_search_result(self, query: str) -> List[Document]:
        query = self.clean_search_query(query)
        
        # Run searches in parallel using asyncio
        loop = asyncio.get_running_loop()
        ddg_task = loop.run_in_executor(None, self.search_ddg, query)
        brave_task = loop.run_in_executor(None, self.search_brave, query)
        
        # Wait for both searches to complete
        ddg_results, brave_results = await asyncio.gather(ddg_task, brave_task)
        
        # Combine results
        search_items = []
        search_items.extend(ddg_results)
        if brave_results:
            search_items.extend(brave_results)

        log.info(f"Searched {len(search_items)} items from DDG and Brave")

        url_to_look = []
        seen_urls = set()
        url_to_snippet = {}
        for res in search_items:
            if isinstance(res, dict) and res.get("link"):
                url = res["link"]
                if url not in seen_urls:
                    seen_urls.add(url)
                    url_to_look.append(url)
                    snippet = res.get("snippet")
                    if snippet:
                        url_to_snippet[url] = snippet
                    if len(url_to_look) >= self.max_urls_to_process:
                        log.info(f"Reached max URLs limit ({self.max_urls_to_process}), stopping URL collection")
                        break
            else:
                log.warning(f"Search result item skipped (not a dict or no link)")
        
        if not url_to_look:
            log.warning(f"No URLs found to load")
            return []

        log.info(f"Attempting to load {len(url_to_look)} URLs")
        
        loader = AsyncHtmlLoader(
            url_to_look, 
            ignore_load_errors=True,
            requests_kwargs={
                "max_line_size": 16384,
                "max_field_size": 16384,
            }
        )
        
        try:
            docs_with_html = await loader.aload()
            log.info(f"Loaded {len(docs_with_html)} documents from web pages")
        except Exception as e:
            log.error(f"Error loading documents from web pages: {e}", exc_info=True)
            return []

        if not docs_with_html:
            log.warning(f"AsyncHtmlLoader returned no documents for URLs")
            return []

        def extract_content(doc: Document) -> Optional[Document]:
            html_content = doc.page_content
            if not html_content:
                return None
            
            extracted_text = trafilatura.extract(
                html_content,
                include_comments=False,
                include_tables=True,
                no_fallback=True
            )

            if not extracted_text:
                return None
                
            new_doc = Document(page_content=extracted_text, metadata=doc.metadata)
            source_url = new_doc.metadata.get("source")
            if source_url and source_url in url_to_snippet:
                new_doc.metadata["snippet"] = url_to_snippet[source_url]
            return new_doc

        loop = asyncio.get_running_loop()
        tasks = [loop.run_in_executor(None, extract_content, doc) for doc in docs_with_html]
        processed_docs_results = await asyncio.gather(*tasks)

        docs = [doc for doc in processed_docs_results if doc is not None]
        log.info(f"Successfully extracted content from {len(docs)} documents using trafilatura")

        # Transform documents in parallel using asyncio
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
                    return [r for r in brave_search_output_str if r.get("link")]
                else:
                    log.error(f"Brave search returned unexpected type: {type(brave_search_output_str)}.")
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
                    log.warning(f"Skipping non-dict item in Brave search results")
            
            return [r for r in formatted_results if r.get("link")]

        except json.JSONDecodeError as e:
            log.error(f"Error decoding JSON from Brave search: {e}.", exc_info=True)
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
        loop = asyncio.get_event_loop()
        search_results_docs = loop.run_until_complete(self.multi_search_result(query))
        return search_results_docs

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        search_results_docs = await self.multi_search_result(query)
        return search_results_docs
