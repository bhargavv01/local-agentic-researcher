"""
Planner Agent for Multi-Agent Research Assistant.
Decomposes complex user research queries into targeted search sub-queries
and refines plans based on critic feedback during iterative DAG loops.
"""

from __future__ import annotations

import logging
from typing import Optional
from local_researcher.llm.client import BaseLLMClient, get_llm_client
from local_researcher.models.state import PlanOutput, ResearchState

logger = logging.getLogger(__name__)

PLANNER_SYSTEM_PROMPT = """You are an expert Research Planning Agent.
Your responsibility is to analyze research queries and decompose them into 2 to 4 strategic, high-signal, non-overlapping search sub-queries.
Ensure queries explore foundational concepts, empirical data/benchmarks, architectural patterns, and real-world trade-offs.

When reviewing feedback or critique from previous iterations, prioritize addressing identified knowledge gaps and exploring suggested follow-up queries.
Always return output strictly matching the PlanOutput schema."""


class PlannerAgent:
    """Agent responsible for initial research decomposition and adaptive re-planning."""

    def __init__(
        self,
        llm_client: Optional[BaseLLMClient] = None,
        name: str = "PlannerAgent",
    ) -> None:
        self.llm_client = llm_client or get_llm_client()
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{self.name}")

    def plan(self, state: ResearchState) -> PlanOutput:
        """Generates or refines search queries for the current research state."""
        is_replanning = state.iteration > 1 and state.latest_critique is not None

        if is_replanning and state.latest_critique:
            prompt = (
                f"TARGET RESEARCH QUERY: {state.query}\n\n"
                f"CURRENT DAG ITERATION: {state.iteration}/{state.max_iterations}\n\n"
                f"PREVIOUS CRITIQUE FEEDBACK: {state.latest_critique.feedback}\n"
                f"IDENTIFIED GAPS: {', '.join(state.latest_critique.identified_gaps) if state.latest_critique.identified_gaps else 'None specified'}\n"
                f"SUGGESTED FOLLOW-UP QUERIES: {', '.join(state.latest_critique.suggested_follow_up_queries) if state.latest_critique.suggested_follow_up_queries else 'None specified'}\n\n"
                "Formulate a refined search plan with 2 to 4 targeted search queries to close these specific gaps."
            )
            action_desc = f"Refined search plan for iteration {state.iteration} based on critique feedback"
        else:
            prompt = (
                f"TARGET RESEARCH QUERY: {state.query}\n\n"
                "Decompose this research query into 2 to 4 targeted sub-queries exploring foundational concepts, "
                "benchmarks/metrics, implementation architectures, and practical trade-offs."
            )
            action_desc = "Generated initial research search plan"

        self.logger.info(f"Generating plan for query: {state.query} (iter={state.iteration})")

        plan = self.llm_client.generate_structured(
            prompt=prompt,
            response_model=PlanOutput,
            system_prompt=PLANNER_SYSTEM_PROMPT,
        )

        state.current_plan = plan
        state.plan_history.append(plan)
        state.add_step(
            agent_name="Planner",
            action=action_desc,
            status="completed",
            details={
                "search_queries": plan.search_queries,
                "focus_aspects": plan.focus_aspects,
                "reasoning": plan.reasoning,
                "iteration": state.iteration,
            },
        )
        return plan

    def run(self, state: ResearchState) -> PlanOutput:
        """Alias for DAG execution."""
        return self.plan(state)
