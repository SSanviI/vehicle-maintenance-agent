"""
Abstract LLM provider interface.
Provider-agnostic abstraction supporting multiple LLM backends.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, AsyncIterator
from pydantic import BaseModel
from enum import Enum


class LLMRole(str, Enum):
    """Logical model roles — not all requests need the strongest model."""
    FAST = "fast"         # Feature extraction, intent detection, simple JSON
    REASONING = "reasoning"  # Agent planning, ambiguous responses, complex reasoning
    JUDGE = "judge"       # Independent response and quality evaluation


class LLMMessage(BaseModel):
    role: str  # "system", "user", "assistant"
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None


class LLMResponse(BaseModel):
    content: str
    model: str
    provider: str
    usage: Optional[Dict[str, int]] = None
    finish_reason: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, name: str, model: str, api_key: str, 
                 base_url: Optional[str] = None, **kwargs):
        self.name = name
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.is_available = True

    @abstractmethod
    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.3,
        max_tokens: int = 1000,
        response_format: Optional[Dict] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        """Generate a completion from the LLM."""
        pass

    @abstractmethod
    async def generate_stream(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.3,
        max_tokens: int = 1000,
    ) -> AsyncIterator[str]:
        """Generate a streaming completion."""
        pass

    async def generate_structured(
        self,
        messages: List[LLMMessage],
        schema_description: str,
        temperature: float = 0.1,
        max_tokens: int = 1000,
    ) -> LLMResponse:
        """
        Generate a structured (JSON) response.
        Default implementation adds JSON instruction to system message.
        """
        # Add JSON format instruction
        json_messages = list(messages)
        system_msg = LLMMessage(
            role="system",
            content=f"You must respond ONLY with valid JSON matching this schema:\n{schema_description}\nDo not include any other text, markdown, or explanation."
        )
        json_messages.insert(0, system_msg)

        return await self.generate(
            json_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the provider is reachable."""
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, model={self.model})"
