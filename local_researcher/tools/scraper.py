"""
Web Scraping and Content Extraction Tool.
Cleans HTML boilerplate using trafilatura and BeautifulSoup with
token-aware truncation designed for small (3B) context windows.
"""

import logging
import re
import urllib.parse
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36 MultiAgentResearchAssistant/1.0"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


class WebScraper:
    """Extracts, cleans, and bounds article content from URLs."""

    def __init__(self, timeout: int = 12, max_cache_size: int = 100):
        self.timeout = timeout
        self._cache: dict[str, str] = {}
        self.max_cache_size = max_cache_size

    def scrape_url(self, url: str, max_chars: int = 3000) -> str:
        """
        Scrapes and extracts the primary textual content from the URL.
        Returns cleaned, sanitized, truncated plain text.
        """
        # Validate URL scheme
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return ""

        # Check memory cache
        if url in self._cache:
            return self._cache[url]

        extracted_text = ""
        try:
            resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=self.timeout)
            if resp.status_code != 200:
                logger.warning(f"Failed to fetch {url}: HTTP {resp.status_code}")
                return ""

            html_content = resp.text

            # 1. Primary Strategy: trafilatura
            try:
                import trafilatura

                extracted = trafilatura.extract(
                    html_content,
                    include_comments=False,
                    include_tables=True,
                    include_links=False,
                    favor_precision=True,
                )
                if extracted and len(extracted.strip()) > 100:
                    extracted_text = extracted.strip()
            except Exception as e:
                logger.debug(f"Trafilatura extraction failed on {url}: {e}")

            # 2. Secondary Strategy: BeautifulSoup fallback
            if not extracted_text:
                extracted_text = self._extract_with_bs4(html_content)

        except Exception as err:
            logger.warning(f"Error scraping URL {url}: {err}")
            return ""

        cleaned = self._clean_and_truncate(extracted_text, max_chars=max_chars)
        
        # Cache result
        if len(self._cache) >= self.max_cache_size:
            self._cache.pop(next(iter(self._cache)))
        self._cache[url] = cleaned

        return cleaned

    def _extract_with_bs4(self, html: str) -> str:
        """Fallback HTML cleaner using BeautifulSoup."""
        soup = BeautifulSoup(html, "html.parser")
        
        # Remove noisy tags
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "svg", "form"]):
            tag.decompose()

        # Find main body or article if present
        main_content = soup.find("article") or soup.find("main") or soup.find("body")
        if not main_content:
            return ""

        text = main_content.get_text(separator="\n", strip=True)
        return text

    def _clean_and_truncate(self, text: str, max_chars: int = 3000) -> str:
        """Sanitizes whitespace, removes duplicate newlines, and truncates smoothly."""
        if not text:
            return ""

        # Normalize whitespace and excessive blank lines
        text = re.sub(r"\r\n|\r", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = text.strip()

        if len(text) <= max_chars:
            return text

        # Truncate at paragraph or sentence boundary
        truncated = text[:max_chars]
        last_newline = truncated.rfind("\n\n")
        if last_newline > max_chars * 0.75:
            return truncated[:last_newline].strip() + "\n\n[...Content truncated for context constraints...]"

        last_period = truncated.rfind(". ")
        if last_period > max_chars * 0.75:
            return truncated[: last_period + 1].strip() + " [...Content truncated for context constraints...]"

        return truncated + "..."
