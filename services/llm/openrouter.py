"""OpenRouter LLM service implementation"""

import json
import asyncio
import os
from typing import Type, TypeVar, Any, Dict
import httpx
from pydantic import BaseModel
from dotenv import load_dotenv

from services.llm.base import LLMProvider

load_dotenv()

T = TypeVar("T", bound=BaseModel)


class OpenRouterProvider(LLMProvider):
    """
    OpenRouter implementation of LLM provider.
    Supports any model available on OpenRouter through a unified interface.
    """

    API_URL = "https://openrouter.ai/api/v1/chat/completions"
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_MAX_TOKENS = 2000

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not found. "
                "Pass it directly or set it in your .env file."
            )

    async def complete_text(
        self,
        prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        messages: list[dict] | None = None,
        enable_reasoning: bool = False,
        omit_temperature: bool = False,
        omit_reasoning: bool = False,
        reasoning_effort: str | None = None,
    ) -> str:
        text, _cost = await self.complete_text_with_cost(
            prompt=prompt, model=model, temperature=temperature,
            max_tokens=max_tokens, messages=messages,
            enable_reasoning=enable_reasoning, omit_temperature=omit_temperature,
            omit_reasoning=omit_reasoning, reasoning_effort=reasoning_effort,
        )
        return text

    async def complete_text_with_cost(
        self,
        prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        messages: list[dict] | None = None,
        enable_reasoning: bool = False,
        omit_temperature: bool = False,
        omit_reasoning: bool = False,
        reasoning_effort: str | None = None,
    ) -> tuple[str, float | None]:
        """Like complete_text, but also returns OpenRouter's reported USD cost for
        the call (None if the endpoint doesn't report one)."""
        if model is None:
            raise ValueError("model is required for OpenRouter calls")
        if prompt is None and messages is None:
            raise ValueError("Either prompt or messages is required")

        # When omit_temperature is True we don't send a temperature at all, so the
        # model falls back to its own server-side default instead of ours.
        temp = None if omit_temperature else (
            temperature if temperature is not None else self.DEFAULT_TEMPERATURE
        )
        tokens = max_tokens if max_tokens is not None else self.DEFAULT_MAX_TOKENS

        msgs = messages if messages is not None else [{"role": "user", "content": prompt}]

        try:
            response = await self._call_api(
                msgs, model, temp, tokens,
                enable_reasoning=enable_reasoning,
                omit_reasoning=omit_reasoning,
                reasoning_effort=reasoning_effort,
            )
            cost = (response.get("usage") or {}).get("cost")
            return self._extract_text(response), cost
        except httpx.HTTPStatusError as e:
            # Surface the provider's error body so callers can react to it (e.g. the
            # identity runner detects "reasoning is mandatory" 400s and retries).
            body = ""
            try:
                body = e.response.text
            except Exception:
                pass
            raise RuntimeError(f"OpenRouter API request failed: {e} :: {body}") from e
        except httpx.HTTPError as e:
            raise RuntimeError(f"OpenRouter API request failed: {e}") from e

    async def complete_text_with_usage(
        self,
        prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        messages: list[dict] | None = None,
        enable_reasoning: bool = False,
    ) -> tuple[str, dict]:
        """Like complete_text but also returns the usage dict from the response.

        Reasoning is off by default: we send `reasoning: {enabled: false}`
        to OpenRouter so thinking-mode models (e.g. Kimi K2.6) don't burn
        the token budget on chain-of-thought before emitting `content`.
        Pass `enable_reasoning=True` to let the model use its native
        reasoning behavior. No-op for models that don't expose reasoning
        controls.
        """
        if model is None:
            raise ValueError("model is required for OpenRouter calls")
        if prompt is None and messages is None:
            raise ValueError("Either prompt or messages is required")

        temp = temperature if temperature is not None else self.DEFAULT_TEMPERATURE
        tokens = max_tokens if max_tokens is not None else self.DEFAULT_MAX_TOKENS
        msgs = messages if messages is not None else [{"role": "user", "content": prompt}]

        response = await self._call_api(
            msgs, model, temp, tokens,
            enable_reasoning=enable_reasoning,
        )
        text = self._extract_text(response)
        usage = response.get("usage", {})
        return text, usage

    async def complete_structured(
        self,
        prompt: str,
        response_model: Type[T],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        enable_reasoning: bool = False,
    ) -> T:
        if model is None:
            raise ValueError("model is required for OpenRouter calls")

        temp = temperature if temperature is not None else self.DEFAULT_TEMPERATURE
        tokens = max_tokens if max_tokens is not None else self.DEFAULT_MAX_TOKENS

        schema = response_model.model_json_schema()
        structured_prompt = (
            f"{prompt}\n\n"
            f"Respond with valid JSON matching this schema:\n"
            f"{json.dumps(schema, indent=2)}\n\n"
            f"Return ONLY the JSON object, no other text."
        )

        try:
            response = await self._call_api(
                [{"role": "user", "content": structured_prompt}],
                model, temp, tokens,
                response_format="json_object",
                enable_reasoning=enable_reasoning,
            )
            response_text = self._extract_text(response)
            json_str = self._extract_json(response_text)
            return response_model.model_validate_json(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM response as JSON: {e}") from e
        except Exception as e:
            raise ValueError(
                f"Failed to parse response into {response_model.__name__}: {e}"
            ) from e

    async def _call_api(
        self,
        messages: list[dict],
        model: str,
        temperature: float | None,
        max_tokens: int,
        retry_count: int = 0,
        max_retries: int = 3,
        response_format: str | None = None,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        enable_reasoning: bool = False,
        omit_reasoning: bool = False,
        reasoning_effort: str | None = None,
    ) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            # Ask OpenRouter to report token usage + USD cost in the response.
            "usage": {"include": True},
        }

        # A None temperature means "let the model use its own default" — omit it.
        if temperature is not None:
            payload["temperature"] = temperature

        if response_format:
            payload["response_format"] = {"type": response_format}

        if logprobs:
            payload["logprobs"] = True
            if top_logprobs is not None:
                payload["top_logprobs"] = top_logprobs

        # Reasoning is off by default. Callers opt in via `enable_reasoning=True`,
        # which sends {enabled: true} so the request ACTIVELY turns reasoning on —
        # important for providers (e.g. Anthropic) whose default is thinking-off:
        # merely omitting the disable flag leaves them not reasoning at all.
        # Sending {enabled: true/false} is a no-op for models without reasoning
        # controls.
        #
        # Some endpoints (e.g. GPT-5, Gemini 2.5 Pro) 400 when reasoning is
        # explicitly disabled because reasoning is mandatory; for those, callers
        # pass omit_reasoning=True to leave the field out entirely and let the
        # model use its own (unavoidable) default.
        # An explicit effort ("low"/"medium"/"high") wins: it's the closest to
        # "off" available for endpoints where reasoning is mandatory and
        # {enabled: false} 400s.
        if reasoning_effort is not None:
            payload["reasoning"] = {"effort": reasoning_effort}
        elif omit_reasoning:
            pass
        elif enable_reasoning:
            payload["reasoning"] = {"enabled": True}
        else:
            payload["reasoning"] = {"enabled": False}

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    self.API_URL, headers=headers, json=payload
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429 and retry_count < max_retries:
                wait_time = 2**retry_count
                print(f"  ⚠ 429 rate limit hit for {model}, retrying in {wait_time}s (attempt {retry_count + 1}/{max_retries})")
                await asyncio.sleep(wait_time)
                return await self._call_api(
                    messages,
                    model,
                    temperature,
                    max_tokens,
                    retry_count + 1,
                    max_retries,
                    response_format,
                    enable_reasoning=enable_reasoning,
                    omit_reasoning=omit_reasoning,
                    reasoning_effort=reasoning_effort,
                )
            raise

    def _extract_text(self, response: Dict[str, Any]) -> str:
        try:
            choices = response.get("choices", [])
            if not choices:
                raise ValueError("No choices in response")

            message = choices[0].get("message", {})
            content = message.get("content", "")

            if not content:
                raise ValueError("Empty content in response")

            return content
        except (KeyError, IndexError) as e:
            raise ValueError(f"Unexpected OpenRouter response format: {response}") from e

    def _extract_json(self, text: str) -> str:
        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end == -1 or start >= end:
            raise ValueError("No valid JSON object found in response")

        return text[start : end + 1]
