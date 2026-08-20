"""
Synthesizer Agent for Multi-Agent Research Assistant.
Compiles verified sources, extracted facts, and critique insights into a comprehensive,
structured synthesis report and renders publication-ready GitHub Flavored Markdown.
"""

from __future__ import annotations

import logging
from typing import Optional
from local_researcher.llm.client import BaseLLMClient, get_llm_client
from local_researcher.models.state import ResearchState, SynthesisOutput

logger = logging.getLogger(__name__)

SYNTHESIZER_SYSTEM_PROMPT = """You are a Lead Research Synthesizer and Technical Writer.
Your mission is to synthesize verified research facts into a coherent, authoritative, and well-structured report.

Requirements:
1. Ground every key technical claim in the provided sources.
2. Structure the report with a comprehensive title, executive summary, key findings, thematic sections with inline citations, and limitations/gaps.
3. Consolidate cited URLs in sources_used.
4. Assign a final_confidence_score (0-100) reflecting report completeness.

Always return valid JSON strictly conforming to the SynthesisOutput schema."""


class SynthesizerAgent:
    """Synthesizes research findings into structured outputs and Markdown reports."""

    def __init__(
        self,
        llm_client: Optional[BaseLLMClient] = None,
        name: str = "SynthesizerAgent",
    ) -> None:
        self.llm_client = llm_client or get_llm_client()
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{self.name}")

    def synthesize(self, state: ResearchState) -> SynthesisOutput:
        """Synthesizes collected research in state into SynthesisOutput and renders Markdown."""
        sources_summary = []
        for idx, doc in enumerate(state.collected_sources[:15], start=1):
            sources_summary.append(f"[{idx}] {doc.title}\nURL: {doc.url}\nExcerpt: {doc.snippet[:300]}")
        sources_text = "\n\n".join(sources_summary) if sources_summary else "No sources available."

        facts_summary = [f"- {fact.statement} (Source: {fact.source_url})" for fact in state.extracted_facts[:20]]
        facts_text = "\n".join(facts_summary) if facts_summary else "No extracted facts."

        critique_summary = state.latest_critique.feedback if state.latest_critique else "Research accepted."

        prompt = (
            f"TARGET RESEARCH QUERY: {state.query}\n\n"
            f"=== VERIFIED SOURCES ({len(state.collected_sources)} Total) ===\n{sources_text}\n\n"
            f"=== GROUNDED FACTS ===\n{facts_text}\n\n"
            f"=== CRITIC ASSESSMENT ===\n{critique_summary}\n\n"
            "Compile an authoritative, in-depth research report synthesizing these findings into the structured SynthesisOutput schema."
        )

        self.logger.info(f"Synthesizing report for query: {state.query}")
        synthesis = self.llm_client.generate_structured(
            prompt=prompt,
            response_model=SynthesisOutput,
            system_prompt=SYNTHESIZER_SYSTEM_PROMPT,
        )

        markdown_report = self.render_markdown(synthesis, state)
        state.final_synthesis = synthesis
        state.final_report_markdown = markdown_report

        state.add_step(
            agent_name="Synthesizer",
            action=f"Compiled final research report: '{synthesis.title}'",
            status="completed",
            details={
                "title": synthesis.title,
                "confidence_score": synthesis.final_confidence_score,
                "sections_count": len(synthesis.sections),
                "sources_used_count": len(synthesis.sources_used),
                "iteration": state.iteration,
            },
        )
        return synthesis

    @staticmethod
    def render_markdown(synthesis: SynthesisOutput, state: ResearchState | None = None) -> str:
        """Renders a structured SynthesisOutput model into clean GitHub Flavored Markdown."""
        lines: list[str] = []

        # Title and Header Metadata
        lines.append(f"# {synthesis.title}\n")

        iter_text = f" | **DAG Iterations**: {state.iteration}" if state else ""
        query_text = f"**Research Query:** *{state.query}*\n\n" if state else ""

        lines.append(f"> **Confidence Score**: {synthesis.final_confidence_score}/100{iter_text}\n")
        if query_text:
            lines.append(query_text)

        # Executive Summary
        lines.append("## Executive Summary\n")
        lines.append(f"{synthesis.executive_summary}\n")

        # Key Findings
        lines.append("## Key Findings\n")
        for finding in synthesis.key_findings:
            lines.append(f"- {finding}")
        lines.append("")

        # Sections
        if synthesis.sections:
            lines.append("## Detailed Analysis\n")
            for section in synthesis.sections:
                lines.append(f"### {section.heading}\n")
                lines.append(f"{section.content}\n")
                if section.citations:
                    citations_str = ", ".join([f"<{url}>" for url in section.citations])
                    lines.append(f"*Citations:* {citations_str}\n")

        # Limitations and Gaps
        if synthesis.limitations_and_gaps:
            lines.append("## Limitations & Open Questions\n")
            for gap in synthesis.limitations_and_gaps:
                lines.append(f"- {gap}")
            lines.append("")

        # Sources Used
        if synthesis.sources_used:
            lines.append("## References\n")
            for idx, source_url in enumerate(synthesis.sources_used, start=1):
                lines.append(f"{idx}. <{source_url}>")
            lines.append("")

        return "\n".join(lines)

    def generate_markdown_report(self, synthesis: SynthesisOutput, state: ResearchState) -> str:
        """Alias for render_markdown."""
        return self.render_markdown(synthesis, state)

    def run(self, state: ResearchState) -> SynthesisOutput:
        """Alias for DAG execution."""
        return self.synthesize(state)
