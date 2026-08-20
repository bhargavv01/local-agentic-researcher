"""
Integration tests for the Directed Acyclic Graph (ResearchGraph) execution engine.
Tests single-iteration happy path, multi-iteration re-planning feedback loops,
streaming event callbacks, and error handling.
"""

from unittest.mock import MagicMock
import pytest
from local_researcher.graph.dag import ResearchGraph
from local_researcher.graph.events import GraphEvent, GraphEventType
from local_researcher.llm.client import MockLLMClient
from local_researcher.models.state import (
    CritiqueOutput,
    PlanOutput,
    SourceDocument,
    SynthesisOutput,
)


def _get_mock_search_tool():
    mock_search = MagicMock()
    mock_search.search.return_value = [
        SourceDocument(
            title="DAG Architectures in AI",
            url="https://example.com/dag",
            snippet="DAG execution prevents cycles and enables structured multi-agent collaboration.",
            source_type="web",
            relevance_score=0.9,
        )
    ]
    return mock_search


def test_research_graph_single_iteration_happy_path():
    """Tests standard single-iteration run where critic approves research on first pass."""
    llm_client = MockLLMClient()
    mock_search = _get_mock_search_tool()

    graph = ResearchGraph(llm_client=llm_client, search_tool=mock_search)
    state = graph.run(
        query="Design patterns for agentic workflow orchestration",
        max_iterations=3,
        confidence_threshold=75,
    )

    assert state.status == "completed"
    assert state.iteration == 1
    assert state.current_plan is not None
    assert len(state.collected_sources) >= 1
    assert state.latest_critique is not None
    assert state.latest_critique.confidence_score >= 75
    assert state.final_synthesis is not None
    assert state.final_report_markdown is not None
    assert "Design patterns" in state.final_report_markdown or "Report" in state.final_report_markdown


def test_research_graph_multi_iteration_replanning():
    """Tests multi-iteration loop where critic fails iteration 1 and passes iteration 2."""
    call_count = {"critic": 0}

    mock_critic = MagicMock()

    def mock_critique_run(state):
        call_count["critic"] += 1
        if call_count["critic"] == 1:
            critique = CritiqueOutput(
                confidence_score=40,
                relevance_score=50,
                factual_grounding_score=45,
                feedback="Insufficient quantitative performance data.",
                identified_gaps=["Quantitative benchmarks"],
                suggested_follow_up_queries=["Agent latency benchmarks 2024"],
                is_sufficient=False,
            )
        else:
            critique = CritiqueOutput(
                confidence_score=90,
                relevance_score=95,
                factual_grounding_score=90,
                feedback="Comprehensive data collected with strong grounding.",
                identified_gaps=[],
                suggested_follow_up_queries=[],
                is_sufficient=True,
            )
        return critique

    mock_critic.run.side_effect = mock_critique_run

    llm_client = MockLLMClient()
    mock_search = _get_mock_search_tool()

    graph = ResearchGraph(
        llm_client=llm_client,
        search_tool=mock_search,
        critic=mock_critic,
    )

    state = graph.run(
        query="Explain agentic feedback loop performance",
        max_iterations=3,
        confidence_threshold=75,
    )

    assert state.status == "completed"
    assert state.iteration == 2
    assert call_count["critic"] == 2
    assert len(state.critique_history) == 2
    assert len(state.plan_history) == 2
    assert state.final_synthesis is not None


def test_research_graph_event_callback_streaming():
    """Tests that event callbacks receive real-time execution lifecycle events."""
    events_received: list[GraphEvent] = []

    def on_event(event: GraphEvent):
        events_received.append(event)

    llm_client = MockLLMClient()
    mock_search = _get_mock_search_tool()

    graph = ResearchGraph(llm_client=llm_client, search_tool=mock_search, callbacks=[on_event])
    state = graph.run(query="Test query for event streaming", max_iterations=2, confidence_threshold=75)

    assert state.status == "completed"
    assert len(events_received) > 0

    event_types = [e.event_type for e in events_received]
    assert GraphEventType.GRAPH_STARTED in event_types
    assert GraphEventType.ITERATION_STARTED in event_types
    assert GraphEventType.NODE_STARTED in event_types
    assert GraphEventType.NODE_COMPLETED in event_types
    assert GraphEventType.CRITIQUE_EVALUATED in event_types
    assert GraphEventType.SYNTHESIS_COMPLETED in event_types
    assert GraphEventType.GRAPH_COMPLETED in event_types


def test_research_graph_max_iterations_exhaustion():
    """Tests that graph forces synthesis when max_iterations is reached even if confidence is low."""
    mock_critic = MagicMock()
    mock_critic.run.return_value = CritiqueOutput(
        confidence_score=30,
        relevance_score=30,
        factual_grounding_score=30,
        feedback="Still incomplete.",
        identified_gaps=["Many gaps"],
        suggested_follow_up_queries=[],
        is_sufficient=False,
    )

    llm_client = MockLLMClient()
    mock_search = _get_mock_search_tool()

    graph = ResearchGraph(
        llm_client=llm_client,
        search_tool=mock_search,
        critic=mock_critic,
    )

    state = graph.run(query="Exhaustion test query", max_iterations=2, confidence_threshold=80)

    assert state.status == "completed"
    assert state.iteration == 2
    assert state.final_synthesis is not None


def test_research_graph_node_failure_handling():
    """Tests that unhandled node exceptions transition graph state to failed."""
    failing_planner = MagicMock()
    failing_planner.run.side_effect = RuntimeError("Planner LLM connection failed")

    graph = ResearchGraph(planner=failing_planner)

    with pytest.raises(RuntimeError, match="Planner LLM connection failed"):
        graph.run(query="Fail test", max_iterations=2)
