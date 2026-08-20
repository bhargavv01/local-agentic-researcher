"""
Critic Agent for Multi-Agent Research Assistant.
Evaluates research adequacy, factual grounding, relevance, and gaps.
Assigns confidence scores (0-100) and determines if the research loop can proceed to synthesis.
"""

from __future__ import annotations

import logging
from typing import Optional
from local_researcher.llm.client import BaseLLMClient, get_llm_client
from local_researcher.models.state import CritiqueOutput, ResearchState

logger = logging.getLogger(__name__)

CRITIC_SYSTEM_PROMPT = """You are a rigorous, objective Research Critic and Fact-Checking Agent.
Your task is to critically evaluate collected evidence against the user's research query.

Assessment Dimensions:
1. Relevance Score (0-100): Are the collected sources directly addressing the research query?
2. Factual Grounding Score (0-100): Are facts well-supported by credible references and clear snippets?
3. Overall Confidence Score (0-100): Combined metric of depth, coverage, and credibility.
4. Identified Gaps: Identify specific blind spots or unverified claims.
5. Suggested Follow-Up Queries: Concrete queries to resolve gaps.
6. Sufficiency (is_sufficient: bool): True only if the research meets quality standards and sufficiently answers the query.

Always output valid JSON strictly matching the CritiqueOutput schema."""


class CriticAgent:
    """Agent that critiques research findings, scores sufficiency, and flags gaps."""

    def __init__(
        self,
        llm_client: Optional[BaseLLMClient] = None,
        name: str = "CriticAgent",
    ) -> None:
        self.llm_client = llm_client or get_llm_client()
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{self.name}")

    def critique(self, state: ResearchState) -> CritiqueOutput:
        """Evaluates current research state and returns a CritiqueOutput model."""
        sources_summary = []
        for idx, doc in enumerate(state.collected_sources[:10], start=1):
            sources_summary.append(
                f"[{idx}] {doc.title} ({doc.source_type})\nURL: {doc.url}\nSnippet: {doc.snippet[:300]}"
            )
        sources_text = "\n\n".join(sources_summary) if sources_summary else "No sources collected."

        facts_summary = [f"- {fact.statement}" for fact in state.extracted_facts[:15]]
        facts_text = "\n".join(facts_summary) if facts_summary else "No facts extracted."

        prompt = (
            f"TARGET RESEARCH QUERY: {state.query}\n\n"
            f"DAG ITERATION: {state.iteration}/{state.max_iterations}\n"
            f"CONFIDENCE THRESHOLD REQUIRED: {state.confidence_threshold}/100\n\n"
            f"TOTAL SOURCES COLLECTED: {len(state.collected_sources)}\n"
            f"=== TOP COLLECTED SOURCES ===\n{sources_text}\n\n"
            f"=== EXTRACTED FACTUAL STATEMENTS ===\n{facts_text}\n\n"
            "Evaluate the adequacy of this research. Provide confidence_score (0-100), relevance_score (0-100), "
            "factual_grounding_score (0-100), feedback, identified_gaps, suggested_follow_up_queries, and is_sufficient."
        )

        self.logger.info(f"Evaluating research state at iteration {state.iteration}")
        critique = self.llm_client.generate_structured(
            prompt=prompt,
            response_model=CritiqueOutput,
            system_prompt=CRITIC_SYSTEM_PROMPT,
        )

        state.latest_critique = critique
        state.critique_history.append(critique)
        state.add_step(
            agent_name="Critic",
            action=f"Evaluated research quality (Score: {critique.confidence_score}/100)",
            status="completed",
            details={
                "confidence_score": critique.confidence_score,
                "relevance_score": critique.relevance_score,
                "factual_grounding_score": critique.factual_grounding_score,
                "is_sufficient": critique.is_sufficient,
                "feedback": critique.feedback,
                "identified_gaps": critique.identified_gaps,
                "iteration": state.iteration,
            },
        )
        return critique

    def evaluate(self, state: ResearchState) -> CritiqueOutput:
        """Alias for critique."""
        return self.critique(state)

    def run(self, state: ResearchState) -> CritiqueOutput:
        """Alias for DAG execution."""
        return self.critique(state)
