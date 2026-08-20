"""
Core State and Data Contracts for the Multi-Agent Research Assistant.
Defines strongly-typed Pydantic schemas for DAG state transitions,
tool outputs, agent communication, and structured JSON generation.
"""

from datetime import datetime, timezone
from typing import Any, Literal
from pydantic import BaseModel, Field


class SourceDocument(BaseModel):
    """Represents a retrieved source document or webpage snippet."""
    title: str = Field(..., description="Title of the source or article")
    url: str = Field(..., description="URL or identifier of the source")
    snippet: str = Field(..., description="Extracted content snippet or summary")
    source_type: Literal["web", "wikipedia", "arxiv", "mock"] = Field(
        default="web", description="The channel / provider where source was found"
    )
    relevance_score: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Relevance to the query"
    )
    published_date: str | None = Field(
        default=None, description="Publication date if available"
    )


class ExtractedFact(BaseModel):
    """An atomic, grounded fact extracted from retrieved sources."""
    statement: str = Field(..., description="Concrete factual statement")
    source_url: str = Field(..., description="URL of the source backing this fact")
    source_title: str = Field(..., description="Title of the source")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence in this fact extraction"
    )


class PlanOutput(BaseModel):
    """Output from the Planner Agent defining search tasks."""
    original_query: str = Field(..., description="Original user research prompt")
    search_queries: list[str] = Field(
        ..., min_length=1, max_length=5, description="Targeted sub-queries to execute"
    )
    focus_aspects: list[str] = Field(
        default_factory=list, description="Key dimensions or angles to investigate"
    )
    reasoning: str = Field(
        default="", description="Strategic reasoning for this decomposition"
    )


class ResearchBatch(BaseModel):
    """Output of a research batch containing discovered sources and facts."""
    search_query: str = Field(..., description="The query that was executed")
    sources: list[SourceDocument] = Field(
        default_factory=list, description="Retrieved source documents"
    )
    extracted_facts: list[ExtractedFact] = Field(
        default_factory=list, description="Atomic extracted facts"
    )


class CritiqueOutput(BaseModel):
    """Output from the Critic Agent evaluating research adequacy."""
    confidence_score: int = Field(
        ..., ge=0, le=100, description="Overall confidence score (0-100)"
    )
    relevance_score: int = Field(
        ..., ge=0, le=100, description="Relevance of collected data (0-100)"
    )
    factual_grounding_score: int = Field(
        ..., ge=0, le=100, description="Factual grounding and source reliability (0-100)"
    )
    feedback: str = Field(
        ..., description="Constructive critique and assessment summary"
    )
    identified_gaps: list[str] = Field(
        default_factory=list, description="Missing angles or unverified claims"
    )
    suggested_follow_up_queries: list[str] = Field(
        default_factory=list, description="Targeted queries to resolve identified gaps"
    )
    is_sufficient: bool = Field(
        ..., description="True if research meets quality and coverage threshold"
    )


class SectionContent(BaseModel):
    """A section within the final synthesized research report."""
    heading: str = Field(..., description="Section title")
    content: str = Field(..., description="Markdown content for this section")
    citations: list[str] = Field(
        default_factory=list, description="List of source URLs cited in this section"
    )


class SynthesisOutput(BaseModel):
    """Structured final output from the Synthesizer Agent."""
    title: str = Field(..., description="Comprehensive title of the report")
    executive_summary: str = Field(..., description="Concise executive summary")
    key_findings: list[str] = Field(
        ..., min_length=1, description="List of key takeaways"
    )
    sections: list[SectionContent] = Field(
        default_factory=list, description="Detailed thematic sections"
    )
    limitations_and_gaps: list[str] = Field(
        default_factory=list, description="Known uncertainties or unaddressed questions"
    )
    sources_used: list[str] = Field(
        default_factory=list, description="All source URLs incorporated in the synthesis"
    )
    final_confidence_score: int = Field(
        ..., ge=0, le=100, description="Final aggregate confidence score"
    )


class ExecutionStep(BaseModel):
    """Record of an execution event within the DAG."""
    step_id: int
    agent_name: str
    action: str
    status: Literal["started", "completed", "failed", "skipped"]
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    details: dict[str, Any] = Field(default_factory=dict)


class ResearchState(BaseModel):
    """Global state container passed between nodes in the DAG execution loop."""
    query: str = Field(..., description="The original user research question")
    iteration: int = Field(default=0, description="Current DAG feedback iteration count")
    max_iterations: int = Field(default=3, description="Maximum allowed feedback loops")
    confidence_threshold: int = Field(
        default=75, ge=0, le=100, description="Minimum confidence score required to pass"
    )
    status: Literal[
        "initialized",
        "planning",
        "researching",
        "critiquing",
        "replanning",
        "synthesizing",
        "completed",
        "failed",
    ] = Field(default="initialized", description="Current DAG stage")

    # Working Memory
    current_plan: PlanOutput | None = None
    plan_history: list[PlanOutput] = Field(default_factory=list)
    collected_sources: list[SourceDocument] = Field(default_factory=list)
    extracted_facts: list[ExtractedFact] = Field(default_factory=list)
    critique_history: list[CritiqueOutput] = Field(default_factory=list)
    latest_critique: CritiqueOutput | None = None
    
    # Final Output
    final_synthesis: SynthesisOutput | None = None
    final_report_markdown: str | None = None
    
    # Execution Audit Trail
    execution_log: list[ExecutionStep] = Field(default_factory=list)
    error_message: str | None = None

    def add_step(
        self,
        agent_name: str,
        action: str,
        status: Literal["started", "completed", "failed", "skipped"],
        details: dict[str, Any] | None = None,
    ) -> None:
        """Helper to append an execution step audit record."""
        step = ExecutionStep(
            step_id=len(self.execution_log) + 1,
            agent_name=agent_name,
            action=action,
            status=status,
            details=details or {},
        )
        self.execution_log.append(step)

    def deduplicate_sources(self) -> None:
        """Deduplicates sources based on URL."""
        seen_urls: set[str] = set()
        unique_sources: list[SourceDocument] = []
        for s in self.collected_sources:
            if s.url not in seen_urls:
                seen_urls.add(s.url)
                unique_sources.append(s)
        self.collected_sources = unique_sources
