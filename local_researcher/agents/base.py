"""
Base agent interface and common functionality for the multi-agent research system.
"""

from __future__ import annotations

import abc
import json
import logging
import re
from typing import Any, Dict, List, Optional, Type, TypeVar
from pydantic import BaseModel

from local_researcher.llm.client import BaseLLMClient

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class BaseAgent(abc.ABC):
    """Abstract base class for all specialized research agents."""

    def __init__(
        self,
        llm_client: BaseLLMClient,
        name: str = "BaseAgent",
        system_prompt: str = "You are an AI research assistant.",
    ) -> None:
        """Initialize the BaseAgent.

        Args:
            llm_client: The LLM client used to perform inference.
            name: Human-readable name of the agent.
            system_prompt: Default system instruction for this agent.
        """
        self.llm_client = llm_client
        self.name = name
        self.system_prompt = system_prompt
        self.logger = logging.getLogger(f"{__name__}.{self.name}")

    def _call_llm(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> str:
        """Call the underlying LLM client with fallback across common method signatures."""
        sys_prompt = system_prompt if system_prompt is not None else self.system_prompt
        call_kwargs = dict(kwargs)
        if temperature is not None:
            call_kwargs["temperature"] = temperature

        client = self.llm_client

        if hasattr(client, "generate") and callable(client.generate):
            try:
                return client.generate(prompt=prompt, system_prompt=sys_prompt, **call_kwargs)
            except TypeError:
                try:
                    return client.generate(prompt, **call_kwargs)
                except Exception as e:
                    self.logger.warning(f"client.generate failed with TypeError fallback: {e}")

        if hasattr(client, "chat") and callable(client.chat):
            try:
                return client.chat(prompt=prompt, system_prompt=sys_prompt, **call_kwargs)
            except TypeError:
                return client.chat(prompt, **call_kwargs)

        if hasattr(client, "invoke") and callable(client.invoke):
            try:
                res = client.invoke(prompt, **call_kwargs)
                return getattr(res, "content", str(res))
            except Exception as e:
                self.logger.warning(f"client.invoke failed: {e}")

        if hasattr(client, "complete") and callable(client.complete):
            return client.complete(prompt=prompt, **call_kwargs)

        if callable(client):
            return client(prompt, **call_kwargs)

        raise AttributeError(
            f"LLM client {type(client).__name__} does not implement any recognized generation method."
        )

    def _extract_json_block(self, text: str) -> str:
        """Extract a valid JSON substring from raw LLM text output."""
        text = text.strip()
        json_code_block = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        if json_code_block:
            return json_code_block.group(1).strip()

        start_obj = text.find("{")
        start_arr = text.find("[")
        if start_obj != -1 and (start_arr == -1 or start_obj < start_arr):
            end_obj = text.rfind("}")
            if end_obj != -1 and end_obj > start_obj:
                return text[start_obj : end_obj + 1]
        elif start_arr != -1:
            end_arr = text.rfind("]")
            if end_arr != -1 and end_arr > start_arr:
                return text[start_arr : end_arr + 1]

        return text

    def _call_llm_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = 0.2,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Call the LLM and parse the response as a JSON dictionary."""
        client = self.llm_client
        if hasattr(client, "generate_json") and callable(client.generate_json):
            try:
                res = client.generate_json(
                    prompt=prompt,
                    system_prompt=system_prompt or self.system_prompt,
                    **kwargs,
                )
                if isinstance(res, dict):
                    return res
                if isinstance(res, str):
                    return json.loads(self._extract_json_block(res))
            except Exception as e:
                self.logger.debug(f"client.generate_json failed, falling back to text parsing: {e}")

        raw_response = self._call_llm(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            **kwargs,
        )

        extracted = self._extract_json_block(raw_response)
        try:
            parsed = json.loads(extracted)
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, list):
                return {"items": parsed}
            return {"result": parsed}
        except json.JSONDecodeError as err:
            self.logger.error(f"Failed to decode JSON from LLM response: {err}\nRaw text: {raw_response}")
            sanitized = re.sub(r",\s*([\]}])", r"\1", extracted)
            try:
                parsed = json.loads(sanitized)
                if isinstance(parsed, dict):
                    return parsed
                return {"result": parsed}
            except Exception:
                raise ValueError(f"Could not parse valid JSON from LLM response: {raw_response}") from err

    def _call_llm_structured(
        self,
        prompt: str,
        response_model: Type[T],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = 0.2,
        **kwargs: Any,
    ) -> T:
        """Call the LLM and parse the output into a Pydantic model instance."""
        client = self.llm_client
        if hasattr(client, "generate_structured") and callable(client.generate_structured):
            try:
                res = client.generate_structured(
                    prompt=prompt,
                    response_model=response_model,
                    system_prompt=system_prompt or self.system_prompt,
                    **kwargs,
                )
                if isinstance(res, response_model):
                    return res
            except Exception as e:
                self.logger.debug(f"client.generate_structured failed, falling back: {e}")

        json_data = self._call_llm_json(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            **kwargs,
        )

        if hasattr(response_model, "model_validate"):
            return response_model.model_validate(json_data)
        if hasattr(response_model, "parse_obj"):
            return response_model.parse_obj(json_data)
        return response_model(**json_data)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"
