"""
Unit tests for Pydantic state models and data validation.
"""

from local_researcher.models.state import (
    SourceDocument,
    ExtractedFact,
    PlanOutput,
    CritiqueOutput,
    SynthesisOutput,
    SectionContent,
    ResearchState,
)


def test_source_document_validation():
    doc = SourceDocument(
        title="Test Source",
        url="https://example.com/article",
        snippet="This is a relevant research snippet.",
        source_type="web",
        relevance_score=0.9,
    )
    assert doc.title == "Test Source"
    assert doc.url == "https://example.com/article"
    assert doc.relevance_score == 0.9
    assert doc.source_type == "web"


def test_plan_output_validation():
    plan = PlanOutput(
        original_query="What are quantum computing algorithms?",
        search_queries=[
            "quantum algorithms overview",
            "shor algorithm quantum computing",
            "grover search quantum complexity",
        ],
        focus_aspects=["Shor's Algorithm", "Grover's Algorithm"],
        reasoning="Target fundamental speedup algorithms.",
    )
    assert len(plan.search_queries) == 3
    assert plan.original_query.startswith("What are")


def test_critique_output_validation():
    critique = CritiqueOutput(
        confidence_score=85,
        relevance_score=90,
        factual_grounding_score=80,
        feedback="High quality research collected.",
        identified_gaps=[],
        suggested_follow_up_queries=[],
        is_sufficient=True,
    )
    assert critique.confidence_score == 85
    assert critique.is_sufficient is True


def test_synthesis_output_validation():
    synthesis = SynthesisOutput(
        title="Quantum Algorithms Analysis",
        executive_summary="Quantum algorithms provide polynomial to exponential speedups.",
        key_findings=["Shor factors integers in poly time", "Grover gives quadratic speedup"],
        sections=[
            SectionContent(
                heading="1. Overview",
                content="Quantum computing leverages superposition and entanglement.",
                citations=["https://example.com/source1"],
            )
        ],
        limitations_and_gaps=["Requires fault-tolerant logical qubits"],
        sources_used=["https://example.com/source1"],
        final_confidence_score=90,
    )
    assert synthesis.final_confidence_score == 90
    assert len(synthesis.key_findings) == 2


def test_research_state_step_audit_and_deduplication():
    state = ResearchState(query="Quantum Error Correction")
    assert state.iteration == 0
    assert state.status == "initialized"

    # Test audit logging
    state.add_step(
        agent_name="Planner",
        action="Created initial search plan",
        status="completed",
        details={"queries_count": 3},
    )
    assert len(state.execution_log) == 1
    assert state.execution_log[0].agent_name == "Planner"

    # Test source deduplication
    doc1 = SourceDocument(title="Doc 1", url="https://test.org/a", snippet="Text A")
    doc2 = SourceDocument(title="Doc 2", url="https://test.org/a", snippet="Duplicate A")
    doc3 = SourceDocument(title="Doc 3", url="https://test.org/b", snippet="Text B")
    state.collected_sources = [doc1, doc2, doc3]

    state.deduplicate_sources()
    assert len(state.collected_sources) == 2
    assert {s.url for s in state.collected_sources} == {"https://test.org/a", "https://test.org/b"}
