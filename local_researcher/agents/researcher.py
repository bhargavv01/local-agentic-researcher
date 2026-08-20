"""
Research Agent for Multi-Agent Research Assistant.
Executes targeted search queries via UnifiedSearchTool, scrapes web content,
extracts grounded factual statements, and manages source collections in state.
"""

from __future__ import annotations

import logging
from typing import Optional
from local_researcher.llm.client import BaseLLMClient, get_llm_client
from local_researcher.models.state import (
    ExtractedFact,
    ResearchBatch,
    ResearchState,
    SourceDocument,
)
from local_researcher.tools.scraper import WebScraper
from local_researcher.tools.search import UnifiedSearchTool

logger = logging.getLogger(__name__)


class ResearchAgent:
    """Agent that performs search retrieval, web scraping, and fact extraction."""

    def __init__(
        self,
        llm_client: Optional[BaseLLMClient] = None,
        search_tool: Optional[UnifiedSearchTool] = None,
        scraper: Optional[WebScraper] = None,
        name: str = "ResearchAgent",
    ) -> None:
        self.llm_client = llm_client or get_llm_client()
        self.search_tool = search_tool or UnifiedSearchTool()
        self.scraper = scraper or WebScraper()
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{self.name}")

    def research_query(self, query: str, max_sources: int = 3) -> ResearchBatch:
        """Executes search for a single query and extracts facts into a ResearchBatch."""
        self.logger.info(f"Executing search for query: '{query}'")
        try:
            docs = self.search_tool.search(query, max_total=max_sources)
        except Exception as e:
            self.logger.error(f"Search failed for '{query}': {e}")
            docs = []

        batch_facts: list[ExtractedFact] = []
        for doc in docs:
            snippet = doc.snippet
            if len(snippet.strip()) < 50 and doc.url.startswith("http"):
                try:
                    scraped = self.scraper.scrape_url(doc.url, max_chars=800)
                    if scraped:
                        snippet = scraped[:400]
                        doc.snippet = snippet
                except Exception as e:
                    self.logger.debug(f"Scraping fallback failed for {doc.url}: {e}")

            if snippet.strip():
                fact = ExtractedFact(
                    statement=snippet.strip(),
                    source_url=doc.url,
                    source_title=doc.title,
                    confidence=doc.relevance_score,
                )
                batch_facts.append(fact)

        return ResearchBatch(
            search_query=query,
            sources=docs,
            extracted_facts=batch_facts,
        )

    def research(
        self,
        state: ResearchState,
        max_sources_per_query: int = 3,
    ) -> list[ResearchBatch]:
        """Executes all search queries from current plan and updates state."""
        if not state.current_plan or not state.current_plan.search_queries:
            self.logger.warning("ResearchAgent called with empty search plan.")
            return []

        batches: list[ResearchBatch] = []
        new_sources_count = 0
        new_facts_count = 0

        for query in state.current_plan.search_queries:
            batch = self.research_query(query, max_sources=max_sources_per_query)
            batches.append(batch)

            for doc in batch.sources:
                state.collected_sources.append(doc)
                new_sources_count += 1

            for fact in batch.extracted_facts:
                state.extracted_facts.append(fact)
                new_facts_count += 1

        state.deduplicate_sources()

        state.add_step(
            agent_name="Researcher",
            action=f"Retrieved {new_sources_count} sources and extracted {new_facts_count} facts",
            status="completed",
            details={
                "queries_executed": state.current_plan.search_queries,
                "total_unique_sources": len(state.collected_sources),
                "total_facts": len(state.extracted_facts),
                "iteration": state.iteration,
            },
        )
        return batches

    def run(self, state: ResearchState) -> list[ResearchBatch]:
        """Alias for DAG execution."""
        return self.research(state)
