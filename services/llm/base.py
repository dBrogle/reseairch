"""Abstract base class for LLM providers"""

from abc import ABC, abstractmethod
from typing import TypeVar, Type
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMProvider(ABC):
    """
    Abstract base class for LLM operations.

    Implementations (OpenRouter, Anthropic, OpenAI, etc.) provide concrete behavior.
    """

    @abstractmethod
    async def complete_text(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """
        Generate a text completion from a prompt.

        Args:
            prompt: Input prompt text
            model: Model identifier (uses provider default if None)
            temperature: Sampling temperature (uses provider default if None)
            max_tokens: Max tokens to generate (uses provider default if None)

        Returns:
            Generated text response
        """
        ...

    @abstractmethod
    async def complete_structured(
        self,
        prompt: str,
        response_model: Type[T],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> T:
        """
        Generate a structured completion that conforms to a Pydantic model.

        Args:
            prompt: Input prompt text
            response_model: Pydantic model class to structure the response
            model: Model identifier (uses provider default if None)
            temperature: Sampling temperature (uses provider default if None)
            max_tokens: Max tokens to generate (uses provider default if None)

        Returns:
            Instance of response_model populated with LLM output
        """
        ...
