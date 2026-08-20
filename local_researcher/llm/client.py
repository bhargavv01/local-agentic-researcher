"""
LLM Client Layer for Local Multi-Agent Research Assistant.
Provides robust structured JSON generation, automatic schema repair,
retry mechanisms optimized for small (3B) Ollama models, and a Mock client for testing.
"""

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Type, TypeVar
import requests
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMResponse(BaseModel):
    """Encapsulates raw and parsed LLM responses."""
    model_config = {"protected_namespaces": ()}

    raw_text: str
    model_name: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    parsed_json: dict[str, Any] | None = None


class BaseLLMClient(ABC):
    """Abstract base class for all LLM providers."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.2,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Generate raw text or raw JSON string from prompt."""
        pass

    def generate_structured(
        self,
        prompt: str,
        response_model: Type[T],
        system_prompt: str | None = None,
        temperature: float = 0.1,
        max_retries: int = 3,
    ) -> T:
        """
        Generate and validate a structured response adhering to a Pydantic model.
        Optimized for 3B parameter models with strict schema prompting and repair.
        """
        schema_json = json.dumps(response_model.model_json_schema(), indent=2)
        enhanced_system_prompt = (
            (system_prompt or "You are a specialized agent in a research workflow.")
            + f"\nTARGET SCHEMA: {response_model.__name__}\n"
            + "\nCRITICAL INSTRUCTION FOR OUTPUT FORMAT:"
            + "\nYou MUST respond strictly in valid JSON matching this exact JSON Schema:"
            + f"\n```json\n{schema_json}\n```"
            + "\nDo NOT include conversational filler, markdown commentary, or apologies. Return ONLY JSON."
        )

        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                # Add slight temperature variation on retries to break repetition loops
                temp = min(temperature + (attempt - 1) * 0.1, 0.7)
                response = self.generate(
                    prompt=prompt,
                    system_prompt=enhanced_system_prompt,
                    temperature=temp,
                    json_mode=True,
                )

                cleaned_json_str = self._extract_json(response.raw_text)
                parsed_dict = json.loads(cleaned_json_str)
                validated_obj = response_model.model_validate(parsed_dict)
                return validated_obj

            except (json.JSONDecodeError, ValidationError) as err:
                last_error = err
                logger.warning(
                    f"Attempt {attempt}/{max_retries} failed for schema {response_model.__name__}: {err}"
                )
                if attempt < max_retries:
                    prompt += f"\n\n[ERROR IN PREVIOUS ATTEMPT]: Output could not be parsed as valid {response_model.__name__}. Error: {str(err)}. Ensure valid JSON syntax."

        raise ValueError(
            f"Failed to generate valid structured output for {response_model.__name__} after {max_retries} attempts. Last error: {last_error}"
        )

    @staticmethod
    def _extract_json(text: str) -> str:
        """
        Extracts JSON content from raw LLM text, handling markdown fences,
        trailing commas, and pre/post-amble text produced by smaller models.
        """
        text = text.strip()

        # Handle ```json ... ``` code blocks
        json_fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        if json_fence_match:
            text = json_fence_match.group(1).strip()

        # If it doesn't start with { or [, try to find the outer JSON bounds
        start_idx_brace = text.find("{")
        start_idx_bracket = text.find("[")

        if start_idx_brace != -1 and (start_idx_bracket == -1 or start_idx_brace < start_idx_bracket):
            end_idx_brace = text.rfind("}")
            if end_idx_brace != -1:
                text = text[start_idx_brace : end_idx_brace + 1]
        elif start_idx_bracket != -1:
            end_idx_bracket = text.rfind("]")
            if end_idx_bracket != -1:
                text = text[start_idx_bracket : end_idx_bracket + 1]

        # Clean trailing commas before closing braces/brackets (common small LLM bug)
        text = re.sub(r",\s*([\}\]])", r"\1", text)

        return text


class OllamaClient(BaseLLMClient):
    """
    Client for local Ollama instance (default http://localhost:11434).
    Enforces format='json' and small model optimizations.
    """

    def __init__(
        self,
        model_name: str = "llama3.2:3b",
        base_url: str = "http://localhost:11434",
        timeout_seconds: int = 120,
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def is_available(self) -> bool:
        """Check if the local Ollama daemon is running and reachable."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False

    def list_models(self) -> list[str]:
        """List available models in Ollama."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            pass
        return []

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.2,
        json_mode: bool = False,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": 0.9,
                "num_ctx": 4096,  # Sane context window for 3B models
            },
        }
        if system_prompt:
            payload["system"] = system_prompt
        if json_mode:
            payload["format"] = "json"

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
            raw_text = data.get("response", "")
            return LLMResponse(
                raw_text=raw_text,
                model_name=self.model_name,
                prompt_tokens=data.get("prompt_eval_count", 0),
                completion_tokens=data.get("eval_count", 0),
            )
        except requests.exceptions.RequestException as err:
            raise RuntimeError(
                f"Ollama request failed at {self.base_url} for model '{self.model_name}'. "
                f"Ensure Ollama is running (`ollama serve`). Details: {err}"
            ) from err


class MockLLMClient(BaseLLMClient):
    """
    Deterministic Mock LLM Client for offline development, integration tests,
    and CI environments where an active GPU / Ollama daemon is not present.
    """

    def __init__(self, model_name: str = "mock-3b"):
        self.model_name = model_name

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.2,
        json_mode: bool = False,
    ) -> LLMResponse:
        content_to_check = f"{system_prompt or ''}\n{prompt}"
        lowered = content_to_check.lower()

        if "target schema: planoutput" in lowered or '"title": "planoutput"' in lowered:
            mock_data = {
                "original_query": prompt.split("\n")[0] if prompt else "Research query",
                "search_queries": [
                    "core architecture and foundational concepts",
                    "benchmarks performance and trade-offs",
                    "recent real-world applications and breakthroughs",
                ],
                "focus_aspects": [
                    "Architectural design",
                    "Performance metrics",
                    "Use cases and limitations",
                ],
                "reasoning": "Decomposed into foundational principles, empirical metrics, and practical implementations.",
            }
        elif "target schema: critiqueoutput" in lowered or '"title": "critiqueoutput"' in lowered:
            mock_data = {
                "confidence_score": 85,
                "relevance_score": 90,
                "factual_grounding_score": 85,
                "feedback": "Collected sources provide reliable and grounded technical insights with adequate depth.",
                "identified_gaps": [],
                "suggested_follow_up_queries": [],
                "is_sufficient": True,
            }
        elif "target schema: synthesisoutput" in lowered or '"title": "synthesisoutput"' in lowered:
            mock_data = {
                "title": "Comprehensive Research Report: State of the Art and Analysis",
                "executive_summary": "This report synthesizes verified empirical findings, architectural patterns, and performance trade-offs.",
                "key_findings": [
                    "Small open-weight models (3B) achieve strong task performance with structured JSON constraints.",
                    "DAG execution loops with critic scoring reduce hallucination rates by over 40%.",
                    "Iterative retrieval ensures factual grounding against live web documents.",
                ],
                "sections": [
                    {
                        "heading": "1. Architectural Paradigms",
                        "content": "Autonomous research agents benefit from decoupled roles: planning, execution, evaluation, and synthesis.",
                        "citations": ["https://example.org/multi-agent-research"],
                    },
                    {
                        "heading": "2. Verification and Grounding",
                        "content": "A critic feedback gate scores incoming evidence before allowing compilation into final outputs.",
                        "citations": ["https://example.org/confidence-scoring"],
                    },
                ],
                "limitations_and_gaps": [
                    "Context window limitations require tight snippet distillation."
                ],
                "sources_used": [
                    "https://example.org/multi-agent-research",
                    "https://example.org/confidence-scoring",
                ],
                "final_confidence_score": 88,
            }
        else:
            mock_data = {
                "status": "ok",
                "message": "Generic mock response",
                "extracted_facts": [
                    {
                        "statement": "Multi-agent systems utilize specialized reasoning loops.",
                        "source_url": "https://example.org/source1",
                        "source_title": "Multi-Agent Overview",
                        "confidence": 0.95,
                    }
                ],
            }

        return LLMResponse(
            raw_text=json.dumps(mock_data, indent=2),
            model_name=self.model_name,
            prompt_tokens=50,
            completion_tokens=150,
            parsed_json=mock_data,
        )


def get_llm_client(
    model_name: str = "llama3.2:3b",
    base_url: str = "http://localhost:11434",
    force_mock: bool = False,
) -> BaseLLMClient:
    """Factory method to get an appropriate LLM client."""
    if force_mock:
        logger.info("Initializing MockLLMClient (mock mode enabled)")
        return MockLLMClient(model_name=f"mock-{model_name}")

    client = OllamaClient(model_name=model_name, base_url=base_url)
    if client.is_available():
        logger.info(f"Connected to local Ollama instance ({base_url}) with model '{model_name}'")
        return client

    logger.warning(
        f"Ollama not reachable at {base_url}. Falling back to MockLLMClient for offline execution."
    )
    return MockLLMClient(model_name=f"mock-{model_name}")
