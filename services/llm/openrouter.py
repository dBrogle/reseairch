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
        prompt: str,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        if model is None:
            raise ValueError("model is required for OpenRouter calls")

        temp = temperature if temperature is not None else self.DEFAULT_TEMPERATURE
        tokens = max_tokens if max_tokens is not None else self.DEFAULT_MAX_TOKENS

        try:
            response = await self._call_api(prompt, model, temp, tokens)
            return self._extract_text(response)
        except httpx.HTTPError as e:
            raise RuntimeError(f"OpenRouter API request failed: {e}") from e

    async def complete_structured(
        self,
        prompt: str,
        response_model: Type[T],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
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
                structured_prompt, model, temp, tokens, response_format="json_object"
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
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
        retry_count: int = 0,
        max_retries: int = 3,
        response_format: str | None = None,
    ) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if response_format:
            payload["response_format"] = {"type": response_format}

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
                await asyncio.sleep(wait_time)
                return await self._call_api(
                    prompt,
                    model,
                    temperature,
                    max_tokens,
                    retry_count + 1,
                    max_retries,
                    response_format,
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
