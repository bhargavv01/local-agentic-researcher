"""
Unit tests for LLM client layer, JSON schema extraction, and mock generation.
"""

from local_researcher.llm.client import BaseLLMClient, MockLLMClient, get_llm_client
from local_researcher.models.state import PlanOutput, CritiqueOutput, SynthesisOutput


def test_json_extraction_from_markdown():
    markdown_wrapped = """
    Here is the requested output:
    ```json
    {
      "original_query": "Explain DAGs in AI agents",
      "search_queries": ["DAG agent execution", "state graph AI workflow"],
      "focus_aspects": ["Graph topology", "Looping mechanisms"],
      "reasoning": "Standard decomposition"
    }
    ```
    Hope this helps!
    """
    extracted = BaseLLMClient._extract_json(markdown_wrapped)
    assert extracted.startswith("{")
    assert extracted.endswith("}")
    assert "DAG agent execution" in extracted


def test_json_extraction_with_trailing_comma():
    malformed_json = """
    {
      "original_query": "Test query",
      "search_queries": ["query 1", "query 2",],
      "focus_aspects": [],
      "reasoning": "Cleaned reasoning",
    }
    """
    cleaned = BaseLLMClient._extract_json(malformed_json)
    # The regex removes commas before } and ]
    assert ",}" not in cleaned
    assert ",]" not in cleaned


def test_mock_llm_structured_plan_generation():
    client = MockLLMClient()
    plan = client.generate_structured(
        prompt="Explain multi-agent state machines in Python",
        response_model=PlanOutput,
        system_prompt="You are a research planner.",
    )
    assert isinstance(plan, PlanOutput)
    assert len(plan.search_queries) >= 1
    assert plan.reasoning != ""


def test_mock_llm_structured_critique_generation():
    client = MockLLMClient()
    critique = client.generate_structured(
        prompt="Critique the gathered research on graph execution.",
        response_model=CritiqueOutput,
        system_prompt="You are a critic agent.",
    )
    assert isinstance(critique, CritiqueOutput)
    assert 0 <= critique.confidence_score <= 100
    assert critique.is_sufficient is True


def test_mock_llm_structured_synthesis_generation():
    client = MockLLMClient()
    synthesis = client.generate_structured(
        prompt="Synthesize the research into a complete report.",
        response_model=SynthesisOutput,
        system_prompt="You are a synthesizer agent.",
    )
    assert isinstance(synthesis, SynthesisOutput)
    assert synthesis.final_confidence_score > 0
    assert len(synthesis.key_findings) > 0


def test_llm_factory_fallback():
    # Attempting connection to a non-existent port should safely return MockLLMClient
    client = get_llm_client(
        model_name="llama3.2:3b",
        base_url="http://127.0.0.1:59999",
        force_mock=False,
    )
    assert isinstance(client, MockLLMClient)
