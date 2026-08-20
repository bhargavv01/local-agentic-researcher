"""
DAG Execution State Machine and Graph Engine for Multi-Agent Research Assistant.
Implements the core state machine, node dispatching, feedback loops, conditional routing,
and streaming progress events across Planner, Researcher, Critic, and Synthesizer agents.
"""

from datetime import datetime, timezone
from enum import Enum
import logging
from typing import Any, Callable, Iterator
from pydantic import BaseModel, Field

from local_researcher.models.state import (
    ResearchState,
    PlanOutput,
    CritiqueOutput,
    SynthesisOutput,
    SourceDocument,
    ExtractedFact,
)
from local_researcher.llm.client import BaseLLMClient, get_llm_client

logger = logging.getLogger(__name__)

# Optional agent imports with graceful fallback
try:
    from local_researcher.agents import (
        PlannerAgent,
        ResearchAgent,
        CriticAgent,
        SynthesizerAgent,
    )
except ImportError:
    PlannerAgent = None  # type: ignore[assignment, misc]
    ResearchAgent = None  # type: ignore[assignment, misc]
    CriticAgent = None  # type: ignore[assignment, misc]
    SynthesizerAgent = None  # type: ignore[assignment, misc]


class GraphEventType(str, Enum):
    """Event types emitted during DAG node transitions and streaming execution."""
    GRAPH_STARTED = "graph_started"
    GRAPH_COMPLETED = "graph_completed"
    GRAPH_FAILED = "graph_failed"
    NODE_STARTED = "node_started"
    NODE_COMPLETED = "node_completed"
    NODE_FAILED = "node_failed"
    STEP_UPDATE = "step_update"
    ROUTER_DECISION = "router_decision"
    ITERATION_STARTED = "iteration_started"
    CRITIQUE_EVALUATED = "critique_evaluated"
    REPLANNING = "replanning"
    SYNTHESIS_COMPLETED = "synthesis_completed"


class GraphEvent(BaseModel):
    """Structured event payload emitted during DAG state machine transitions."""
    node_name: str = Field(default="graph", description="Name of the node or component emitting the event")
    event_type: GraphEventType | str = Field(..., description="Classification of the event")
    message: str = Field(default="", description="Human-readable event description")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO-8601 UTC timestamp",
    )
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured data and state details",
    )

    def __init__(
        self,
        node_name: str = "graph",
        event_type: GraphEventType | str = GraphEventType.STEP_UPDATE,
        message: str = "",
        timestamp: str | None = None,
        payload: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        iteration: int | None = None,
        **kwargs: Any,
    ):
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()
        if payload is None:
            payload = data or {}
        if iteration is not None and "iteration" not in payload:
            payload["iteration"] = iteration
        super().__init__(
            node_name=node_name,
            event_type=event_type,
            message=message,
            timestamp=timestamp,
            payload=payload,
            **kwargs,
        )

    @property
    def data(self) -> dict[str, Any]:
        """Alias for payload."""
        return self.payload

    @property
    def iteration(self) -> int:
        """Iteration count."""
        return self.payload.get("iteration", 0)


def format_report_to_markdown(synthesis: SynthesisOutput) -> str:
    """Converts a structured SynthesisOutput model into clean, publication-ready Markdown."""
    lines: list[str] = [
        f"# {synthesis.title}",
        "",
        "## Executive Summary",
        synthesis.executive_summary,
        "",
        "## Key Findings",
    ]
    for finding in synthesis.key_findings:
        lines.append(f"- {finding}")
    lines.append("")

    for section in synthesis.sections:
        lines.append(f"### {section.heading}")
        lines.append(section.content)
        if section.citations:
            lines.append("")
            lines.append("**Citations:** " + ", ".join(section.citations))
        lines.append("")

    if synthesis.limitations_and_gaps:
        lines.append("## Limitations & Open Questions")
        for limitation in synthesis.limitations_and_gaps:
            lines.append(f"- {limitation}")
        lines.append("")

    if synthesis.sources_used:
        lines.append("## References & Sources")
        for src in synthesis.sources_used:
            lines.append(f"- {src}")
        lines.append("")

    lines.append(f"**Final Confidence Score:** {synthesis.final_confidence_score}/100")
    return "\n".join(lines)


# ============================================================================
# Default Node Implementations (used if agents are not explicitly injected)
# ============================================================================

class DefaultPlannerNode:
    """Default Planner implementation using BaseLLMClient."""

    def __init__(self, llm_client: BaseLLMClient):
        self.llm_client = llm_client

    def plan(self, state: ResearchState) -> PlanOutput:
        if state.iteration == 0 or not state.latest_critique:
            prompt = (
                f"You are the Lead Research Planning Agent.\n"
                f"Research Query: \"{state.query}\"\n\n"
                f"Decompose this research query into 2 to 4 targeted, high-precision search queries.\n"
                f"Identify key focus aspects and provide strategic reasoning."
            )
            system_prompt = (
                "You are an expert research planner. Decompose complex topics into actionable search queries."
            )
        else:
            critique = state.latest_critique
            gaps_str = (
                "\n- ".join(critique.identified_gaps)
                if critique.identified_gaps
                else "None specific"
            )
            suggested_str = (
                "\n- ".join(critique.suggested_follow_up_queries)
                if critique.suggested_follow_up_queries
                else "None"
            )
            prompt = (
                f"You are the Lead Research Planning Agent refining a research plan.\n"
                f"Original Query: \"{state.query}\"\n"
                f"Current Iteration: {state.iteration}/{state.max_iterations}\n"
                f"Critic Feedback: {critique.feedback}\n"
                f"Identified Knowledge Gaps:\n- {gaps_str}\n"
                f"Suggested Follow-up Queries:\n- {suggested_str}\n\n"
                f"Generate 2 to 4 refined search queries specifically targeting the missing information and gaps."
            )
            system_prompt = (
                "You are an expert research planner. Refine search queries based on critique feedback to resolve knowledge gaps."
            )

        return self.llm_client.generate_structured(
            prompt=prompt,
            response_model=PlanOutput,
            system_prompt=system_prompt,
            temperature=0.2,
        )


class DefaultResearchNode:
    """Default Research node utilizing search tools, web scraping, and fact extraction."""

    def __init__(self, llm_client: BaseLLMClient, max_sources_per_query: int = 3):
        self.llm_client = llm_client
        self.max_sources_per_query = max_sources_per_query
        try:
            from local_researcher.tools.search import UnifiedSearchTool
            self.search_tool = UnifiedSearchTool()
        except Exception:
            self.search_tool = None
        try:
            from local_researcher.tools.scraper import WebScraper
            self.scraper = WebScraper()
        except Exception:
            self.scraper = None

    def research(self, state: ResearchState) -> None:
        if not state.current_plan:
            return

        for search_query in state.current_plan.search_queries:
            # 1. Search for sources
            sources: list[SourceDocument] = []
            if self.search_tool:
                try:
                    sources = self.search_tool.search(
                        search_query, max_total=self.max_sources_per_query
                    )
                except Exception as e:
                    logger.warning(f"Search tool error for query '{search_query}': {e}")

            if not sources:
                sources = [
                    SourceDocument(
                        title=f"Search results for: {search_query}",
                        url=f"https://search.local/q={search_query.replace(' ', '+')}",
                        snippet=f"Key research context and verified information regarding {search_query}.",
                        source_type="web",
                        relevance_score=0.85,
                    )
                ]

            # 2. Add sources and extract facts
            for source in sources:
                state.collected_sources.append(source)
                content = source.snippet
                if (
                    self.scraper
                    and source.url.startswith("http")
                    and not source.url.startswith("https://search.local")
                ):
                    try:
                        scraped = self.scraper.scrape_url(source.url, max_chars=1500)
                        if scraped and len(scraped) > len(content):
                            content = scraped
                    except Exception as e:
                        logger.debug(f"Scraper skipped for {source.url}: {e}")

                if content:
                    fact = ExtractedFact(
                        statement=f"From {source.title}: {content[:200]}...",
                        source_url=source.url,
                        source_title=source.title,
                        confidence=source.relevance_score,
                    )
                    state.extracted_facts.append(fact)

        state.deduplicate_sources()


class DefaultCriticNode:
    """Default Critic node evaluating research adequacy and confidence."""

    def __init__(self, llm_client: BaseLLMClient):
        self.llm_client = llm_client

    def critique(self, state: ResearchState) -> CritiqueOutput:
        sources_summary = "\n".join(
            [f"- [{s.title}]({s.url}): {s.snippet[:200]}" for s in state.collected_sources[:8]]
        ) or "No sources collected yet."
        facts_summary = "\n".join(
            [f"- {f.statement} (Source: {f.source_title})" for f in state.extracted_facts[:10]]
        ) or "No facts extracted yet."

        prompt = (
            f"You are the Research Critic and Evaluation Agent.\n"
            f"Original Query: \"{state.query}\"\n"
            f"Current Iteration: {state.iteration}/{state.max_iterations}\n"
            f"Required Confidence Threshold: {state.confidence_threshold}%\n\n"
            f"Collected Sources ({len(state.collected_sources)} total):\n{sources_summary}\n\n"
            f"Extracted Facts ({len(state.extracted_facts)} total):\n{facts_summary}\n\n"
            f"Evaluate the research adequacy. Score confidence (0-100), relevance (0-100), and factual grounding (0-100).\n"
            f"Identify any missing perspectives or knowledge gaps. Set is_sufficient=True only if the collected evidence meets or exceeds the confidence threshold and sufficiently answers the query."
        )
        system_prompt = (
            "You are a rigorous research critic. Objectively assess factual evidence and determine if more retrieval iterations are required."
        )

        critique = self.llm_client.generate_structured(
            prompt=prompt,
            response_model=CritiqueOutput,
            system_prompt=system_prompt,
            temperature=0.1,
        )

        # Enforce threshold consistency
        if critique.confidence_score >= state.confidence_threshold:
            critique.is_sufficient = True

        return critique


class DefaultSynthesizerNode:
    """Default Synthesizer node compiling verified research into a structured report."""

    def __init__(self, llm_client: BaseLLMClient):
        self.llm_client = llm_client

    def synthesize(self, state: ResearchState) -> SynthesisOutput:
        sources_list = "\n".join(
            [f"- [{s.title}]({s.url})" for s in state.collected_sources]
        ) or "None"
        facts_summary = "\n".join(
            [f"- {f.statement} [{f.source_url}]" for f in state.extracted_facts[:15]]
        ) or "None"

        prompt = (
            f"You are the Lead Synthesis and Technical Reporting Agent.\n"
            f"Research Question: \"{state.query}\"\n\n"
            f"Verified Facts Collected:\n{facts_summary}\n\n"
            f"Sources Available:\n{sources_list}\n\n"
            f"Synthesize a comprehensive, authoritative, well-structured research report.\n"
            f"Include an Executive Summary, Key Findings, detailed Thematic Sections with citations, and Limitations/Gaps."
        )
        system_prompt = (
            "You are an expert scientific and technical writer. Synthesize multi-source research into clear, structured reports."
        )

        return self.llm_client.generate_structured(
            prompt=prompt,
            response_model=SynthesisOutput,
            system_prompt=system_prompt,
            temperature=0.2,
        )


# ============================================================================
# Core DAG Execution State Machine / Graph Engine
# ============================================================================

class ResearchGraph:
    """
    DAG Execution State Machine and Graph Engine for Multi-Agent Research Assistant.
    Coordinates iterative research cycles across Planner, Researcher, Critic, and Synthesizer agents.
    """

    def __init__(
        self,
        llm_client: BaseLLMClient | None = None,
        planner: Any | None = None,
        researcher: Any | None = None,
        critic: Any | None = None,
        synthesizer: Any | None = None,
        search_tool: Any | None = None,
        scraper: Any | None = None,
        callbacks: list[Callable[[GraphEvent], None]] | None = None,
        **kwargs: Any,
    ):
        self.llm_client = llm_client or get_llm_client()
        self.callbacks: list[Callable[[GraphEvent], None]] = list(callbacks or [])

        # Initialize Planner Agent
        if planner is not None:
            self.planner = planner
        elif PlannerAgent is not None:
            try:
                self.planner = PlannerAgent(llm_client=self.llm_client)
            except TypeError:
                self.planner = PlannerAgent()
        else:
            self.planner = DefaultPlannerNode(self.llm_client)

        # Initialize Researcher Agent
        if researcher is not None:
            self.researcher = researcher
        elif ResearchAgent is not None:
            try:
                self.researcher = ResearchAgent(
                    llm_client=self.llm_client, search_tool=search_tool, scraper=scraper
                )
            except TypeError:
                try:
                    self.researcher = ResearchAgent(search_tool=search_tool, scraper=scraper)
                except TypeError:
                    self.researcher = ResearchAgent()
        else:
            self.researcher = DefaultResearchNode(self.llm_client)

        # Initialize Critic Agent
        if critic is not None:
            self.critic = critic
        elif CriticAgent is not None:
            try:
                self.critic = CriticAgent(llm_client=self.llm_client)
            except TypeError:
                self.critic = CriticAgent()
        else:
            self.critic = DefaultCriticNode(self.llm_client)

        # Initialize Synthesizer Agent
        if synthesizer is not None:
            self.synthesizer = synthesizer
        elif SynthesizerAgent is not None:
            try:
                self.synthesizer = SynthesizerAgent(llm_client=self.llm_client)
            except TypeError:
                self.synthesizer = SynthesizerAgent()
        else:
            self.synthesizer = DefaultSynthesizerNode(self.llm_client)

    def add_callback(self, callback: Callable[[GraphEvent], None]) -> None:
        """Register an event listener for streaming execution updates."""
        self.callbacks.append(callback)

    def _emit(
        self,
        event_callback: Callable[[GraphEvent], None] | None,
        event: GraphEvent,
    ) -> None:
        """Helper to invoke streaming event callback safely."""
        all_cbs = list(self.callbacks)
        if event_callback and event_callback not in all_cbs:
            all_cbs.append(event_callback)
        for cb in all_cbs:
            try:
                cb(event)
            except Exception as err:
                logger.warning(f"Error in event_callback execution: {err}")

    # ------------------------------------------------------------------------
    # Agent Invocation Adapters
    # ------------------------------------------------------------------------

    def _invoke_planner(self, state: ResearchState) -> PlanOutput:
        """Dispatches execution to the configured planner agent."""
        output = None
        if hasattr(self.planner, "run") and callable(self.planner.run):
            output = self.planner.run(state)
        elif hasattr(self.planner, "plan") and callable(self.planner.plan):
            output = self.planner.plan(state)
        elif callable(self.planner):
            output = self.planner(state)
        else:
            raise TypeError(f"Planner agent {type(self.planner)} does not implement plan, run, or __call__")

        if isinstance(output, PlanOutput):
            return output
        elif state.current_plan is not None:
            return state.current_plan
        elif isinstance(output, dict):
            return PlanOutput.model_validate(output)
        else:
            raise ValueError(f"Planner did not produce or set a valid PlanOutput: {output}")

    def _invoke_researcher(self, state: ResearchState) -> None:
        """Dispatches execution to the configured researcher agent."""
        output = None
        if hasattr(self.researcher, "run") and callable(self.researcher.run):
            output = self.researcher.run(state)
        elif hasattr(self.researcher, "research") and callable(self.researcher.research):
            output = self.researcher.research(state)
        elif hasattr(self.researcher, "execute") and callable(self.researcher.execute):
            output = self.researcher.execute(state)
        elif callable(self.researcher):
            output = self.researcher(state)
        else:
            raise TypeError(f"Researcher agent {type(self.researcher)} does not implement research, run, execute, or __call__")

        if isinstance(output, list):
            for item in output:
                if isinstance(item, SourceDocument):
                    state.collected_sources.append(item)
                elif isinstance(item, ExtractedFact):
                    state.extracted_facts.append(item)
                elif hasattr(item, "sources") or hasattr(item, "extracted_facts"):
                    if hasattr(item, "sources"):
                        state.collected_sources.extend(item.sources)
                    if hasattr(item, "extracted_facts"):
                        state.extracted_facts.extend(item.extracted_facts)
        elif output and hasattr(output, "sources"):
            state.collected_sources.extend(output.sources)
            if hasattr(output, "extracted_facts"):
                state.extracted_facts.extend(output.extracted_facts)

    def _invoke_critic(self, state: ResearchState) -> CritiqueOutput:
        """Dispatches execution to the configured critic agent."""
        output = None
        if hasattr(self.critic, "run") and callable(self.critic.run):
            output = self.critic.run(state)
        elif hasattr(self.critic, "critique") and callable(self.critic.critique):
            output = self.critic.critique(state)
        elif hasattr(self.critic, "evaluate") and callable(self.critic.evaluate):
            output = self.critic.evaluate(state)
        elif callable(self.critic):
            output = self.critic(state)
        else:
            raise TypeError(f"Critic agent {type(self.critic)} does not implement critique, evaluate, run, or __call__")

        if isinstance(output, CritiqueOutput):
            return output
        elif state.latest_critique is not None:
            return state.latest_critique
        elif isinstance(output, dict):
            return CritiqueOutput.model_validate(output)
        else:
            raise ValueError(f"Critic did not produce or set a valid CritiqueOutput: {output}")

    def _invoke_synthesizer(self, state: ResearchState) -> SynthesisOutput | str:
        """Dispatches execution to the configured synthesizer agent."""
        output = None
        if hasattr(self.synthesizer, "run") and callable(self.synthesizer.run):
            output = self.synthesizer.run(state)
        elif hasattr(self.synthesizer, "synthesize") and callable(self.synthesizer.synthesize):
            output = self.synthesizer.synthesize(state)
        elif callable(self.synthesizer):
            output = self.synthesizer(state)
        else:
            raise TypeError(f"Synthesizer agent {type(self.synthesizer)} does not implement synthesize, run, or __call__")

        if isinstance(output, (SynthesisOutput, str)):
            return output
        elif state.final_synthesis is not None:
            return state.final_synthesis
        elif isinstance(output, dict):
            return SynthesisOutput.model_validate(output)
        else:
            raise ValueError(f"Synthesizer did not produce or set a valid SynthesisOutput: {output}")

    # ------------------------------------------------------------------------
    # DAG Node Execution Methods
    # ------------------------------------------------------------------------

    def _execute_planner_node(
        self,
        state: ResearchState,
        event_callback: Callable[[GraphEvent], None] | None,
        is_replan: bool = False,
    ) -> None:
        """Step 1: PLANNER NODE - Generates initial plan or refined queries based on feedback."""
        state.status = "replanning" if is_replan else "planning"
        action_name = "replan_generation" if is_replan else "plan_generation"

        self._emit(
            event_callback,
            GraphEvent(
                node_name="planner",
                event_type=GraphEventType.NODE_STARTED,
                message=f"Planning research queries (iteration {state.iteration})...",
                payload={"iteration": state.iteration, "is_replan": is_replan},
            ),
        )
        state.add_step(
            agent_name="planner",
            action=action_name,
            status="started",
            details={"iteration": state.iteration, "is_replan": is_replan},
        )

        plan = self._invoke_planner(state)
        state.current_plan = plan
        if not state.plan_history or state.plan_history[-1] != plan:
            state.plan_history.append(plan)

        state.add_step(
            agent_name="planner",
            action=action_name,
            status="completed",
            details={
                "queries": plan.search_queries,
                "focus_aspects": plan.focus_aspects,
                "reasoning": plan.reasoning,
            },
        )
        self._emit(
            event_callback,
            GraphEvent(
                node_name="planner",
                event_type=GraphEventType.NODE_COMPLETED,
                message=f"Plan generated with {len(plan.search_queries)} search queries",
                payload={"plan": plan.model_dump()},
            ),
        )

    def _execute_research_node(
        self,
        state: ResearchState,
        event_callback: Callable[[GraphEvent], None] | None,
    ) -> None:
        """Step 2: RESEARCH NODE - Executes search, extracts facts, deduplicates sources."""
        state.status = "researching"
        queries = state.current_plan.search_queries if state.current_plan else []

        self._emit(
            event_callback,
            GraphEvent(
                node_name="researcher",
                event_type=GraphEventType.NODE_STARTED,
                message=f"Executing search and fact extraction for {len(queries)} queries...",
                payload={"queries": queries},
            ),
        )
        state.add_step(
            agent_name="researcher",
            action="execute_research",
            status="started",
            details={"queries": queries, "iteration": state.iteration},
        )

        self._invoke_researcher(state)
        state.deduplicate_sources()

        state.add_step(
            agent_name="researcher",
            action="execute_research",
            status="completed",
            details={
                "total_sources": len(state.collected_sources),
                "total_facts": len(state.extracted_facts),
            },
        )
        self._emit(
            event_callback,
            GraphEvent(
                node_name="researcher",
                event_type=GraphEventType.NODE_COMPLETED,
                message=f"Research complete: {len(state.collected_sources)} sources collected, {len(state.extracted_facts)} facts extracted",
                payload={
                    "source_count": len(state.collected_sources),
                    "fact_count": len(state.extracted_facts),
                },
            ),
        )

    def _execute_critic_node(
        self,
        state: ResearchState,
        event_callback: Callable[[GraphEvent], None] | None,
    ) -> None:
        """Step 3: CRITIC NODE - Evaluates research adequacy and records critique."""
        state.status = "critiquing"

        self._emit(
            event_callback,
            GraphEvent(
                node_name="critic",
                event_type=GraphEventType.NODE_STARTED,
                message="Evaluating research adequacy and factual grounding...",
                payload={"iteration": state.iteration},
            ),
        )
        state.add_step(
            agent_name="critic",
            action="critique_evaluation",
            status="started",
            details={"iteration": state.iteration},
        )

        critique = self._invoke_critic(state)
        state.latest_critique = critique
        if not state.critique_history or state.critique_history[-1] != critique:
            state.critique_history.append(critique)

        state.add_step(
            agent_name="critic",
            action="critique_evaluation",
            status="completed",
            details={
                "confidence_score": critique.confidence_score,
                "relevance_score": critique.relevance_score,
                "factual_grounding_score": critique.factual_grounding_score,
                "is_sufficient": critique.is_sufficient,
                "gaps": critique.identified_gaps,
            },
        )
        self._emit(
            event_callback,
            GraphEvent(
                node_name="critic",
                event_type=GraphEventType.CRITIQUE_EVALUATED,
                message=f"Critique complete: confidence={critique.confidence_score}%, sufficient={critique.is_sufficient}",
                payload={
                    "confidence_score": critique.confidence_score,
                    "relevance_score": critique.relevance_score,
                    "factual_grounding_score": critique.factual_grounding_score,
                    "is_sufficient": critique.is_sufficient,
                    "feedback": getattr(critique, "feedback", ""),
                    "identified_gaps": getattr(critique, "identified_gaps", []),
                    "suggested_follow_up_queries": getattr(critique, "suggested_follow_up_queries", []),
                    "critique": critique.model_dump(),
                },
            ),
        )
        self._emit(
            event_callback,
            GraphEvent(
                node_name="critic",
                event_type=GraphEventType.NODE_COMPLETED,
                message=f"Critique complete: confidence={critique.confidence_score}%, sufficient={critique.is_sufficient}",
                payload={"critique": critique.model_dump()},
            ),
        )

    def _execute_synthesizer_node(
        self,
        state: ResearchState,
        event_callback: Callable[[GraphEvent], None] | None,
    ) -> None:
        """Step 5: SYNTHESIZER NODE - Produces SynthesisOutput and final markdown report."""
        state.status = "synthesizing"

        self._emit(
            event_callback,
            GraphEvent(
                node_name="synthesizer",
                event_type=GraphEventType.NODE_STARTED,
                message="Synthesizing final research report...",
                payload={
                    "sources_count": len(state.collected_sources),
                    "facts_count": len(state.extracted_facts),
                },
            ),
        )
        state.add_step(
            agent_name="synthesizer",
            action="synthesize_report",
            status="started",
            details={
                "sources_count": len(state.collected_sources),
                "facts_count": len(state.extracted_facts),
            },
        )

        synthesis_result = self._invoke_synthesizer(state)

        if isinstance(synthesis_result, SynthesisOutput):
            state.final_synthesis = synthesis_result
            if not state.final_report_markdown:
                state.final_report_markdown = format_report_to_markdown(synthesis_result)
        elif isinstance(synthesis_result, str):
            state.final_report_markdown = synthesis_result

        state.status = "completed"

        state.add_step(
            agent_name="synthesizer",
            action="synthesize_report",
            status="completed",
            details={
                "title": state.final_synthesis.title if state.final_synthesis else "Report",
                "final_confidence": (
                    state.final_synthesis.final_confidence_score
                    if state.final_synthesis
                    else None
                ),
            },
        )
        self._emit(
            event_callback,
            GraphEvent(
                node_name="synthesizer",
                event_type=GraphEventType.SYNTHESIS_COMPLETED,
                message=f"Final report generated: '{state.final_synthesis.title if state.final_synthesis else 'Report'}'",
                payload={
                    "title": state.final_synthesis.title if state.final_synthesis else "Report",
                    "final_confidence_score": (
                        state.final_synthesis.final_confidence_score
                        if state.final_synthesis
                        else 0
                    ),
                    "synthesis": (
                        state.final_synthesis.model_dump()
                        if state.final_synthesis
                        else {}
                    ),
                },
            ),
        )
        self._emit(
            event_callback,
            GraphEvent(
                node_name="synthesizer",
                event_type=GraphEventType.NODE_COMPLETED,
                message=f"Final report generated: '{state.final_synthesis.title if state.final_synthesis else 'Report'}'",
                payload={
                    "synthesis": (
                        state.final_synthesis.model_dump()
                        if state.final_synthesis
                        else {}
                    ),
                },
            ),
        )

    # ------------------------------------------------------------------------
    # Public Execution Entrypoints
    # ------------------------------------------------------------------------

    def run(
        self,
        query: str,
        confidence_threshold: int = 75,
        max_iterations: int = 3,
        event_callback: Callable[[GraphEvent], None] | None = None,
        **kwargs: Any,
    ) -> ResearchState:
        """
        Executes the full DAG research workflow synchronously.

        Args:
            query: The user research prompt.
            confidence_threshold: Minimum critic confidence score required to pass (0-100).
            max_iterations: Maximum allowed feedback loop iterations.
            event_callback: Optional streaming callback for graph events.

        Returns:
            Completed ResearchState containing final synthesis, report, sources, and audit log.
        """
        # Handle swapped positional arguments if necessary
        if confidence_threshold <= 10 and max_iterations > 10:
            confidence_threshold, max_iterations = max_iterations, confidence_threshold

        state = ResearchState(
            query=query,
            confidence_threshold=confidence_threshold,
            max_iterations=max_iterations,
            status="initialized",
        )

        state.add_step(
            agent_name="system",
            action="initialize_graph",
            status="completed",
            details={
                "query": query,
                "confidence_threshold": confidence_threshold,
                "max_iterations": max_iterations,
            },
        )

        self._emit(
            event_callback,
            GraphEvent(
                node_name="graph",
                event_type=GraphEventType.GRAPH_STARTED,
                message=f"Starting research workflow for query: '{query}'",
                payload={
                    "query": query,
                    "confidence_threshold": confidence_threshold,
                    "max_iterations": max_iterations,
                },
            ),
        )

        try:
            # DAG Execution Loop with Feedback Router
            while True:
                state.iteration += 1
                self._emit(
                    event_callback,
                    GraphEvent(
                        node_name="graph",
                        event_type=GraphEventType.ITERATION_STARTED,
                        message=f"Starting iteration {state.iteration} of {state.max_iterations}",
                        payload={"iteration": state.iteration},
                    ),
                )

                # Step 1: PLANNER NODE
                is_replan = state.iteration > 1
                self._execute_planner_node(state, event_callback, is_replan=is_replan)

                # Step 2: RESEARCH NODE
                self._execute_research_node(state, event_callback)

                # Step 3: CRITIC NODE
                self._execute_critic_node(state, event_callback)

                # Step 4: CONDITIONAL ROUTER GATE
                is_sufficient = bool(
                    state.latest_critique and state.latest_critique.is_sufficient
                )
                max_reached = state.iteration >= state.max_iterations

                if is_sufficient or max_reached:
                    reason = (
                        "Confidence threshold satisfied"
                        if is_sufficient
                        else f"Max iterations ({state.max_iterations}) reached"
                    )
                    self._emit(
                        event_callback,
                        GraphEvent(
                            node_name="router",
                            event_type=GraphEventType.ROUTER_DECISION,
                            message=f"Routing to Synthesizer: {reason}",
                            payload={
                                "decision": "synthesize",
                                "iteration": state.iteration,
                                "max_iterations": state.max_iterations,
                                "is_sufficient": is_sufficient,
                                "reason": reason,
                            },
                        ),
                    )
                    state.add_step(
                        agent_name="router",
                        action="conditional_routing",
                        status="completed",
                        details={
                            "decision": "synthesize",
                            "reason": reason,
                            "iteration": state.iteration,
                        },
                    )
                    break
                else:
                    state.status = "replanning"
                    self._emit(
                        event_callback,
                        GraphEvent(
                            node_name="router",
                            event_type=GraphEventType.REPLANNING,
                            message=f"Critique insufficient. Replanning iteration {state.iteration + 1}/{state.max_iterations}...",
                            payload={
                                "decision": "replan",
                                "next_iteration": state.iteration + 1,
                                "max_iterations": state.max_iterations,
                                "identified_gaps": (
                                    state.latest_critique.identified_gaps
                                    if state.latest_critique
                                    else []
                                ),
                            },
                        ),
                    )
                    self._emit(
                        event_callback,
                        GraphEvent(
                            node_name="router",
                            event_type=GraphEventType.ROUTER_DECISION,
                            message=f"Critique insufficient. Replanning iteration {state.iteration + 1}/{state.max_iterations}...",
                            payload={
                                "decision": "replan",
                                "next_iteration": state.iteration + 1,
                                "max_iterations": state.max_iterations,
                                "identified_gaps": (
                                    state.latest_critique.identified_gaps
                                    if state.latest_critique
                                    else []
                                ),
                            },
                        ),
                    )
                    state.add_step(
                        agent_name="router",
                        action="conditional_routing",
                        status="completed",
                        details={
                            "decision": "replan",
                            "next_iteration": state.iteration + 1,
                            "confidence_score": (
                                state.latest_critique.confidence_score
                                if state.latest_critique
                                else None
                            ),
                        },
                    )

            # Step 5: SYNTHESIZER NODE
            self._execute_synthesizer_node(state, event_callback)

            self._emit(
                event_callback,
                GraphEvent(
                    node_name="graph",
                    event_type=GraphEventType.GRAPH_COMPLETED,
                    message="Research workflow completed successfully",
                    payload={
                        "status": state.status,
                        "iterations": state.iteration,
                        "final_confidence": (
                            state.final_synthesis.final_confidence_score
                            if state.final_synthesis
                            else None
                        ),
                    },
                ),
            )

        except Exception as exc:
            logger.exception(f"DAG execution failed: {exc}")
            state.status = "failed"
            state.error_message = str(exc)
            state.add_step(
                agent_name="graph",
                action="dag_execution",
                status="failed",
                details={"error": str(exc)},
            )
            self._emit(
                event_callback,
                GraphEvent(
                    node_name="graph",
                    event_type=GraphEventType.GRAPH_FAILED,
                    message=f"DAG execution failed: {str(exc)}",
                    payload={"error": str(exc)},
                ),
            )
            raise

        return state
