"""
OpenAI-compatible LLM provider.
Works with: OpenAI, Groq, OpenRouter, Together AI, local Ollama, any OpenAI-compatible API.
"""

import json
from typing import Dict, Any, Optional, List, AsyncIterator
from openai import AsyncOpenAI, APIError, RateLimitError, APIConnectionError, APITimeoutError
from backend.app.llm.base import LLMProvider, LLMMessage, LLMResponse
from backend.app.core.logging import get_logger

logger = get_logger(__name__)


class OpenAICompatibleProvider(LLMProvider):
    """
    Provider for any OpenAI-compatible API.
    Supports: OpenAI, Groq, OpenRouter, Together, Ollama, etc.
    """

    def __init__(self, name: str, model: str, api_key: str, 
                 base_url: Optional[str] = None, **kwargs):
        super().__init__(name, model, api_key, base_url, **kwargs)
        
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=30.0,
            max_retries=0,  # We handle retries in the router
        )

    @property
    def api_model(self) -> str:
        """Resolve API model name for provider endpoints. Environment configuration is the source of truth."""
        return self.model

    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.3,
        max_tokens: int = 1000,
        response_format: Optional[Dict] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        """Generate a completion."""
        try:
            api_messages = []
            for message in messages:
                payload = {"role": message.role, "content": message.content}
                if message.name:
                    payload["name"] = message.name
                if message.tool_call_id:
                    payload["tool_call_id"] = message.tool_call_id
                api_messages.append(payload)
            
            kwargs = {
                "model": self.api_model,
                "messages": api_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            if response_format:
                kwargs["response_format"] = response_format
            if tools:
                kwargs["tools"] = tools

            response = await self.client.chat.completions.create(**kwargs)

            message = response.choices[0].message
            tool_calls = [
                {"id": call.id, "name": call.function.name, "arguments": call.function.arguments}
                for call in (message.tool_calls or [])
            ]
            return LLMResponse(
                content=message.content or "",
                model=response.model,
                provider=self.name,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                },
                finish_reason=response.choices[0].finish_reason,
                tool_calls=tool_calls or None,
            )

        except RateLimitError as e:
            logger.warning(f"Rate limited by {self.name}: {e}")
            raise
        except APITimeoutError as e:
            logger.warning(f"Timeout from {self.name}: {e}")
            raise
        except APIConnectionError as e:
            logger.error(f"Connection error to {self.name}: {e}")
            raise
        except APIError as e:
            logger.error(f"API error from {self.name}: {e}")
            raise

    async def generate_stream(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.3,
        max_tokens: int = 1000,
    ) -> AsyncIterator[str]:
        """Generate a streaming completion."""
        try:
            api_messages = [{"role": m.role, "content": m.content} for m in messages]
            
            stream = await self.client.chat.completions.create(
                model=self.api_model,
                messages=api_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except RateLimitError:
            logger.warning(f"Rate limited during stream from {self.name}")
            raise
        except Exception as e:
            logger.error(f"Stream error from {self.name}: {e}")
            raise

    async def health_check(self) -> bool:
        """Check provider availability with a minimal request."""
        try:
            response = await self.client.chat.completions.create(
                model=self.api_model,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=1,
                temperature=0,
            )
            return True
        except Exception as e:
            logger.warning(f"Health check failed for {self.name} ({self.api_model}): {e}")
            return False
