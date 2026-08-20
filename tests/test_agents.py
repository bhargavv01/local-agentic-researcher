"""
Unit and integration tests for individual Agent classes (Planner, Researcher, Critic, Synthesizer).
Uses MockLLMClient for deterministic and offline test execution.
"""

from unittest.mock import MagicMock
from local_researcher.agents.planner import PlannerAgent
from local_researcher.agents.researcher import ResearchAgent
from local_researcher.agents.critic import CriticAgent
from local_researcher.agents.synthesizer import SynthesizerAgent
from local_researcher.llm.client import MockLLMClient
from local_researcher.models.state import (
    CritiqueOutput,
    PlanOutput,
    ResearchState,
    SourceDocument,
    SynthesisOutput,
)


def test_planner_agent_initial_plan():
    client = MockLLMClient()
    planner = PlannerAgent(llm_client=client)

    state = ResearchState(query="Explain state graphs in autonomous AI agents")
    plan = planner.run(state)

    assert isinstance(plan, PlanOutput)
    assert len(plan.search_queries) >= 1
    assert state.current_plan == plan
    assert len(state.plan_history) == 1
    assert len(state.execution_log) == 1
    assert state.execution_log[0].agent_name == "Planner"
    assert state.execution_log[0].status == "completed"


def test_planner_agent_replanning_with_critique_feedback():
    client = MockLLMClient()
    planner = PlannerAgent(llm_client=client)

    state = ResearchState(query="Explain state graphs in autonomous AI agents")
    state.iteration = 2
    state.latest_critique = CritiqueOutput(
        confidence_score=45,
        relevance_score=60,
        factual_grounding_score=40,
        feedback="Missing real-world benchmark metrics and DAG error recovery strategies.",
        identified_gaps=["DAG error recovery", "Agent benchmark results"],
        suggested_follow_up_queries=["AI agent DAG error recovery", "Multi agent benchmarks 2024"],
        is_sufficient=False,
    )

    plan = planner.run(state)
    assert isinstance(plan, PlanOutput)
    assert len(state.plan_history) == 1
    assert state.current_plan == plan
    assert "Refined search plan" in state.execution_log[-1].action


def test_research_agent_execution_and_extraction():
    mock_search = MagicMock()
    mock_search.search.return_value = [
        SourceDocument(
            title="DAG Agent Execution Patterns",
            url="https://example.com/dag-patterns",
            snippet="Directed Acyclic Graphs allow deterministic task orchestration across AI agents.",
            source_type="web",
            relevance_score=0.95,
        ),
        SourceDocument(
            title="State Machine Agents",
            url="https://example.com/state-machine",
            snippet="Finite state machines prevent infinite loops in agentic feedback loops.",
            source_type="web",
            relevance_score=0.90,
        ),
    ]
    mock_scraper = MagicMock()

    researcher = ResearchAgent(search_tool=mock_search, scraper=mock_scraper)

    state = ResearchState(query="Multi-Agent DAG Architecture")
    state.current_plan = PlanOutput(
        original_query="Multi-Agent DAG Architecture",
        search_queries=["DAG Agent Execution Patterns"],
        focus_aspects=["Architecture"],
        reasoning="Explore DAG design patterns.",
    )

    batches = researcher.run(state)
    assert len(batches) == 1
    assert len(state.collected_sources) == 2
    assert len(state.extracted_facts) == 2
    assert state.extracted_facts[0].source_url == "https://example.com/dag-patterns"
    assert len(state.execution_log) == 1
    assert state.execution_log[0].agent_name == "Researcher"


def test_critic_agent_evaluation():
    client = MockLLMClient()
    critic = CriticAgent(llm_client=client)

    state = ResearchState(query="Multi-Agent Systems", confidence_threshold=75)
    state.collected_sources = [
        SourceDocument(
            title="Agent Architecture",
            url="https://example.org/agents",
            snippet="Autonomous agents operate in structured feedback loops.",
            source_type="web",
            relevance_score=0.9,
        )
    ]

    critique = critic.run(state)
    assert isinstance(critique, CritiqueOutput)
    assert 0 <= critique.confidence_score <= 100
    assert state.latest_critique == critique
    assert len(state.critique_history) == 1
    assert len(state.execution_log) == 1
    assert state.execution_log[0].agent_name == "Critic"


def test_synthesizer_agent_and_markdown_rendering():
    client = MockLLMClient()
    synthesizer = SynthesizerAgent(llm_client=client)

    state = ResearchState(query="Local Small LLM Agent Frameworks")
    state.collected_sources = [
        SourceDocument(
            title="Small Models for Agentic Workflows",
            url="https://example.org/small-models",
            snippet="3B models with strict JSON grammar achieve high task reliability.",
            source_type="web",
            relevance_score=0.95,
        )
    ]

    synthesis = synthesizer.run(state)
    assert isinstance(synthesis, SynthesisOutput)
    assert state.final_synthesis == synthesis
    assert state.final_report_markdown is not None
    assert synthesis.title in state.final_report_markdown
    assert "Executive Summary" in state.final_report_markdown
    assert "Key Findings" in state.final_report_markdown
    assert len(state.execution_log) == 1
    assert state.execution_log[0].agent_name == "Synthesizer"
