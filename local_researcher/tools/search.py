"""
Search Tools for Multi-Agent Research Assistant.
Integrates DuckDuckGo, Wikipedia, and ArXiv with fallback heuristics,
rate-limiting, and normalized SourceDocument schema output.
"""

import logging
import urllib.parse
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from typing import Any
import requests
from bs4 import BeautifulSoup
from local_researcher.models.state import SourceDocument

logger = logging.getLogger(__name__)

# User-Agent for ethical web scraping and API calls
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36 MultiAgentResearchAssistant/1.0"
    )
}


class BaseSearchEngine(ABC):
    """Abstract interface for all search providers."""

    @abstractmethod
    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Execute search query and return list of normalized SourceDocument models."""
        pass


class DuckDuckGoSearchEngine(BaseSearchEngine):
    """DuckDuckGo search provider with DDGS and fallback."""

    def __init__(self, timeout: int = 10):
        self.timeout = timeout

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        documents: list[SourceDocument] = []
        
        # 1. Try duckduckgo_search library (DDGS)
        try:
            from duckduckgo_search import DDGS

            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
                for item in results:
                    title = item.get("title", "").strip()
                    url = item.get("href") or item.get("link") or ""
                    snippet = item.get("body") or item.get("snippet") or ""
                    if title and url:
                        documents.append(
                            SourceDocument(
                                title=title,
                                url=url,
                                snippet=snippet,
                                source_type="web",
                                relevance_score=0.95,
                            )
                        )
            if documents:
                return documents
        except Exception as e:
            logger.debug(f"DDGS search failed: {e}. Attempting HTML fallback.")

        # 2. Fallback: DuckDuckGo HTML parser
        try:
            encoded_query = urllib.parse.quote(query)
            url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
            resp = requests.post(
                "https://html.duckduckgo.com/html/",
                data={"q": query},
                headers=DEFAULT_HEADERS,
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                results = soup.find_all("div", class_="result__body", limit=max_results)
                for res in results:
                    title_elem = res.find("a", class_="result__a")
                    snippet_elem = res.find("a", class_="result__snippet")
                    if title_elem and snippet_elem:
                        title = title_elem.get_text(strip=True)
                        raw_href = title_elem.get("href", "")
                        # Parse actual target URL from DDG redirect url
                        if "uddg=" in raw_href:
                            raw_href = urllib.parse.unquote(
                                raw_href.split("uddg=")[1].split("&")[0]
                            )
                        snippet = snippet_elem.get_text(strip=True)
                        documents.append(
                            SourceDocument(
                                title=title,
                                url=raw_href,
                                snippet=snippet,
                                source_type="web",
                                relevance_score=0.85,
                            )
                        )
        except Exception as e:
            logger.warning(f"DuckDuckGo search error for query '{query}': {e}")

        return documents


class WikipediaSearchEngine(BaseSearchEngine):
    """Wikipedia search provider using Wikipedia OpenSearch and Action APIs."""

    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.api_url = "https://en.wikipedia.org/w/api.php"

    def search(self, query: str, max_results: int = 3) -> list[SourceDocument]:
        documents: list[SourceDocument] = []
        try:
            params: dict[str, Any] = {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "utf8": 1,
                "srlimit": max_results,
            }
            resp = requests.get(
                self.api_url, params=params, headers=DEFAULT_HEADERS, timeout=self.timeout
            )
            if resp.status_code == 200:
                data = resp.json()
                search_items = data.get("query", {}).get("search", [])
                for item in search_items:
                    title = item.get("title", "")
                    page_id = item.get("pageid")
                    # Clean HTML tags in wikipedia snippet
                    raw_snippet = item.get("snippet", "")
                    clean_snippet = BeautifulSoup(raw_snippet, "html.parser").get_text()
                    wiki_url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
                    
                    documents.append(
                        SourceDocument(
                            title=f"Wikipedia: {title}",
                            url=wiki_url,
                            snippet=clean_snippet,
                            source_type="wikipedia",
                            relevance_score=0.90,
                        )
                    )
        except Exception as e:
            logger.warning(f"Wikipedia search error for query '{query}': {e}")

        return documents


class ArxivSearchEngine(BaseSearchEngine):
    """ArXiv search provider querying the ArXiv Export API for scholarly papers."""

    def __init__(self, timeout: int = 12):
        self.timeout = timeout
        self.api_url = "http://export.arxiv.org/api/query"

    def search(self, query: str, max_results: int = 3) -> list[SourceDocument]:
        documents: list[SourceDocument] = []
        try:
            params = {
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": max_results,
            }
            resp = requests.get(
                self.api_url, params=params, headers=DEFAULT_HEADERS, timeout=self.timeout
            )
            if resp.status_code == 200:
                # Parse Atom XML
                root = ET.fromstring(resp.content)
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                for entry in root.findall("atom:entry", ns):
                    title_elem = entry.find("atom:title", ns)
                    summary_elem = entry.find("atom:summary", ns)
                    id_elem = entry.find("atom:id", ns)
                    published_elem = entry.find("atom:published", ns)

                    title = title_elem.text.strip().replace("\n", " ") if title_elem is not None and title_elem.text else ""
                    summary = summary_elem.text.strip().replace("\n", " ") if summary_elem is not None and summary_elem.text else ""
                    url = id_elem.text.strip() if id_elem is not None and id_elem.text else ""
                    pub_date = published_elem.text.strip() if published_elem is not None and published_elem.text else None

                    if title and url:
                        documents.append(
                            SourceDocument(
                                title=f"arXiv: {title}",
                                url=url,
                                snippet=summary[:800],
                                source_type="arxiv",
                                relevance_score=0.92,
                                published_date=pub_date,
                            )
                        )
        except Exception as e:
            logger.warning(f"ArXiv search error for query '{query}': {e}")

        return documents


class UnifiedSearchTool:
    """
    Coordinates multi-provider search queries across Web, Wikipedia, and ArXiv.
    Deduplicates URLs and limits result sets to prevent context explosion.
    """

    def __init__(
        self,
        enable_web: bool = True,
        enable_wikipedia: bool = True,
        enable_arxiv: bool = True,
    ):
        self.providers: list[BaseSearchEngine] = []
        if enable_web:
            self.providers.append(DuckDuckGoSearchEngine())
        if enable_wikipedia:
            self.providers.append(WikipediaSearchEngine())
        if enable_arxiv:
            self.providers.append(ArxivSearchEngine())

    def search(self, query: str, max_total: int = 5) -> list[SourceDocument]:
        """Runs search across providers and returns deduplicated sources."""
        all_docs: list[SourceDocument] = []
        seen_urls: set[str] = set()

        for provider in self.providers:
            try:
                # Distribute quota across providers
                quota = max(2, max_total // len(self.providers))
                docs = provider.search(query, max_results=quota)
                for doc in docs:
                    if doc.url not in seen_urls:
                        seen_urls.add(doc.url)
                        all_docs.append(doc)
            except Exception as e:
                logger.error(f"Search provider {provider.__class__.__name__} failed: {e}")

        # Sort by relevance score descending
        all_docs.sort(key=lambda d: d.relevance_score, reverse=True)
        return all_docs[:max_total]
