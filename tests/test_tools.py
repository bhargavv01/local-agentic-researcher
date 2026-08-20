"""
Unit tests for search tools and web scraping components.
"""

from unittest.mock import MagicMock, patch
from local_researcher.tools.search import (
    WikipediaSearchEngine,
    ArxivSearchEngine,
    UnifiedSearchTool,
)
from local_researcher.tools.scraper import WebScraper


def test_scraper_html_cleaning():
    scraper = WebScraper()
    mock_html = """
    <html>
        <head><title>Test Page</title></head>
        <body>
            <nav><a href="#">Home</a><a href="#">About</a></nav>
            <article>
                <h1>Multi-Agent Architectures</h1>
                <p>Multi-agent systems utilize distinct LLM instances executing specialized tasks in a graph.</p>
                <p>This allows self-correction and confidence evaluation.</p>
            </article>
            <footer>Copyright 2025 Example</footer>
        </body>
    </html>
    """
    cleaned = scraper._extract_with_bs4(mock_html)
    assert "Multi-Agent Architectures" in cleaned
    assert "Home" not in cleaned  # nav removed
    assert "Copyright" not in cleaned  # footer removed


def test_scraper_truncation():
    scraper = WebScraper()
    long_text = "Sentence one. " * 300
    truncated = scraper._clean_and_truncate(long_text, max_chars=200)
    assert len(truncated) <= 250
    assert "truncated" in truncated.lower() or truncated.endswith("...")


def test_scraper_caching():
    scraper = WebScraper()
    scraper._cache["https://example.com/cached-article"] = "Pre-cached content."
    result = scraper.scrape_url("https://example.com/cached-article")
    assert result == "Pre-cached content."


@patch("requests.get")
def test_wikipedia_search_parsing(mock_get):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "query": {
            "search": [
                {
                    "title": "Agent-based model",
                    "snippet": "An <span>agent-based</span> model is a computational model.",
                    "pageid": 12345,
                }
            ]
        }
    }
    mock_get.return_value = mock_resp

    wiki = WikipediaSearchEngine()
    results = wiki.search("agent based model", max_results=1)
    assert len(results) == 1
    assert "Agent-based model" in results[0].title
    assert results[0].source_type == "wikipedia"
    assert "<span>" not in results[0].snippet


@patch("requests.get")
def test_arxiv_search_parsing(mock_get):
    atom_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>http://arxiv.org/abs/2301.00001v1</id>
        <title>Autonomous Multi-Agent AI Framework</title>
        <summary>This paper presents a DAG execution loop for LLMs.</summary>
        <published>2023-01-01T00:00:00Z</published>
      </entry>
    </feed>
    """
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = atom_xml.encode("utf-8")
    mock_get.return_value = mock_resp

    arxiv = ArxivSearchEngine()
    results = arxiv.search("multi-agent framework", max_results=1)
    assert len(results) == 1
    assert "Autonomous Multi-Agent AI Framework" in results[0].title
    assert results[0].source_type == "arxiv"
    assert results[0].published_date == "2023-01-01T00:00:00Z"


def test_unified_search_deduplication():
    tool = UnifiedSearchTool(enable_web=False, enable_wikipedia=False, enable_arxiv=False)
    
    # Mock custom provider
    mock_provider = MagicMock()
    from local_researcher.models.state import SourceDocument
    mock_provider.search.return_value = [
        SourceDocument(title="Doc 1", url="https://example.com/1", snippet="A"),
        SourceDocument(title="Doc 1 Duplicate", url="https://example.com/1", snippet="A duplicate"),
        SourceDocument(title="Doc 2", url="https://example.com/2", snippet="B"),
    ]
    tool.providers = [mock_provider]

    results = tool.search("query", max_total=5)
    assert len(results) == 2
    assert {r.url for r in results} == {"https://example.com/1", "https://example.com/2"}
