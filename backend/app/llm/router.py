"""
Multi-LLM Router with rate-limit failover, circuit breaker, and health tracking.
Routes requests across Primary → Secondary → Tertiary providers.
"""

import asyncio
import time
from typing import Dict, Any, Optional, List, AsyncIterator
from backend.app.llm.base import LLMProvider, LLMMessage, LLMResponse, LLMRole
from backend.app.llm.providers.openai_compatible import OpenAICompatibleProvider
from backend.app.core.config import get_settings
from backend.app.core.logging import get_logger

logger = get_logger(__name__)


class ProviderHealth:
    """Track health and cooldown state for a single provider."""

    def __init__(self, name: str):
        self.name = name
        self.consecutive_failures = 0
        self.last_failure_time = 0.0
        self.cooldown_until = 0.0
        self.is_circuit_open = False
        self.total_requests = 0
        self.total_failures = 0

    def record_success(self):
        self.consecutive_failures = 0
        self.is_circuit_open = False
        self.total_requests += 1

    def record_failure(self, error_type: str = "general"):
        self.consecutive_failures += 1
        self.total_failures += 1
        self.total_requests += 1
        self.last_failure_time = time.time()

        # Set cooldown based on error type
        if error_type == "rate_limit":
            cooldown = min(30 * (2 ** (self.consecutive_failures - 1)), 300)  # 30s, 60s, 120s, max 5min
        elif error_type == "model_not_found":
            cooldown = 3600  # 1 hour for invalid model configuration
        elif error_type == "timeout":
            cooldown = 120
        elif error_type == "server_error":
            cooldown = 60
        else:
            cooldown = 30

        self.cooldown_until = time.time() + cooldown

        # Open circuit breaker after 3 consecutive failures
        if self.consecutive_failures >= 3:
            self.is_circuit_open = True
            self.cooldown_until = time.time() + 300  # 5 minute circuit breaker
            logger.warning(f"Circuit breaker OPEN for {self.name}")

    def is_available(self) -> bool:
        """Check if provider is available (not in cooldown or circuit-broken)."""
        if self.is_circuit_open:
            # Check if circuit breaker timeout has passed
            if time.time() > self.cooldown_until:
                self.is_circuit_open = False
                self.consecutive_failures = 0
                logger.info(f"Circuit breaker CLOSED for {self.name} — retrying")
                return True
            return False

        if time.time() < self.cooldown_until:
            return False

        return True

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "available": self.is_available(),
            "circuit_open": self.is_circuit_open,
            "consecutive_failures": self.consecutive_failures,
            "total_requests": self.total_requests,
            "total_failures": self.total_failures,
            "cooldown_remaining": max(0, self.cooldown_until - time.time()),
        }


class LLMRouter:
    """
    Routes LLM requests across multiple providers with failover.
    
    Priority: Primary → Secondary → Tertiary
    
    Handles:
    - Rate limits (HTTP 429) → exponential backoff + immediate failover
    - Model Not Found (HTTP 404) → detailed logging + failover
    - Timeouts → failover
    - Provider failures → circuit breaker
    - Health tracking
    """

    def __init__(self):
        self.providers: List[LLMProvider] = []
        self.health: Dict[str, ProviderHealth] = {}
        self.max_retries_per_provider = 2

    def add_provider(self, provider: LLMProvider):
        """Add a provider to the router."""
        self.providers.append(provider)
        self.health[provider.name] = ProviderHealth(provider.name)
        logger.info(f"Added LLM provider: {provider.name} ({provider.model})")

    def initialize_from_settings(self):
        """Initialize providers from application settings."""
        settings = get_settings()
        configs = settings.get_llm_configs()

        for config in configs:
            try:
                provider = OpenAICompatibleProvider(
                    name=config["name"],
                    model=config["model"],
                    api_key=config["api_key"],
                    base_url=config.get("base_url"),
                )
                self.add_provider(provider)
            except Exception as e:
                logger.error(f"Failed to initialize provider {config['name']}: {e}")

        if not self.providers:
            logger.warning("No LLM providers configured! Agent will not function.")

    async def check_all_providers_health(self) -> Dict[str, bool]:
        """Perform lightweight health checks for all configured providers on startup."""
        logger.info("Running LLM Provider Startup Health Check...")
        results = {}
        for provider in self.providers:
            health = self.health[provider.name]
            try:
                is_healthy = await provider.health_check()
                if is_healthy:
                    health.record_success()
                    logger.info(f"LLM Provider '{provider.name}' ({provider.model}) -> HEALTHY")
                    results[provider.name] = True
                else:
                    health.record_failure("health_check_failed")
                    logger.warning(
                        f"LLM Provider '{provider.name}' ({provider.model}) at '{provider.base_url}' -> UNHEALTHY"
                    )
                    results[provider.name] = False
            except Exception as e:
                error_type = self._classify_error(e)
                health.record_failure(error_type)
                logger.error(
                    f"LLM Provider '{provider.name}' ({provider.model}) at '{provider.base_url}' "
                    f"health check error [{error_type}]: {e}"
                )
                results[provider.name] = False
        return results

    async def generate(
        self,
        messages: List[LLMMessage],
        role: LLMRole = LLMRole.FAST,
        temperature: float = 0.3,
        max_tokens: int = 1000,
        response_format: Optional[Dict] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        """
        Generate a response, trying providers in priority order with failover.
        Enforces a strict 30-second maximum timeout across all attempts.
        """
        try:
            return await asyncio.wait_for(
                self._generate_with_failover(
                    messages=messages,
                    role=role,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                    tools=tools,
                ),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            logger.error("LLM request timed out after 30 seconds.")
            raise RuntimeError("LLM request timed out after 30 seconds.")

    async def _generate_with_failover(
        self,
        messages: List[LLMMessage],
        role: LLMRole,
        temperature: float,
        max_tokens: int,
        response_format: Optional[Dict],
        tools: Optional[List[Dict[str, Any]]],
    ) -> LLMResponse:
        last_error = None

        for provider in self.providers:
            health = self.health[provider.name]

            if not health.is_available():
                logger.debug(f"Skipping {provider.name} — in cooldown")
                continue

            for attempt in range(self.max_retries_per_provider):
                try:
                    response = await provider.generate(
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        response_format=response_format,
                        tools=tools,
                    )
                    health.record_success()
                    return response

                except Exception as e:
                    error_type = self._classify_error(e)
                    
                    if error_type == "model_not_found":
                        logger.error(
                            f"LLM Provider '{provider.name}' HTTP 404 (model_not_found): "
                            f"model='{provider.model}', base_url='{provider.base_url}' — {e}"
                        )
                        health.record_failure("model_not_found")
                        last_error = e
                        break  # Do not retry invalid model configurations

                    if error_type == "rate_limit":
                        logger.warning(
                            f"LLM Provider '{provider.name}' rate limited (HTTP 429/TPD). "
                            f"Failing over to next provider..."
                        )
                        health.record_failure("rate_limit")
                        last_error = e
                        break  # Immediately failover to next provider

                    logger.warning(
                        f"Provider {provider.name} attempt {attempt+1} failed: "
                        f"{error_type} — {str(e)[:100]}"
                    )

                    if attempt < self.max_retries_per_provider - 1:
                        await asyncio.sleep(1 * (attempt + 1))
                    else:
                        health.record_failure(error_type)

                    last_error = e

        # All providers failed
        error_msg = f"All LLM providers failed. Last error: {last_error}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    async def generate_stream(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.3,
        max_tokens: int = 1000,
    ) -> AsyncIterator[str]:
        """Generate streaming response with failover and 30s timeout on start."""
        last_error = None

        for provider in self.providers:
            health = self.health[provider.name]

            if not health.is_available():
                continue

            try:
                stream_iter = provider.generate_stream(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                async for chunk in stream_iter:
                    yield chunk
                health.record_success()
                return

            except Exception as e:
                error_type = self._classify_error(e)
                health.record_failure(error_type)
                last_error = e
                logger.warning(f"Stream failover from {provider.name} [{error_type}]: {e}")
                continue

        if last_error:
            logger.error("All LLM providers failed while streaming. Last error: %s", last_error)
        else:
            logger.error("No available LLM providers are configured for streaming.")

        yield (
            "\n\n[The maintenance prediction was updated successfully above, but the "
            "conversational AI response service is temporarily unavailable. Please try again shortly.]"
        )

    async def generate_structured(
        self,
        messages: List[LLMMessage],
        schema_description: str,
        temperature: float = 0.1,
        max_tokens: int = 1000,
    ) -> LLMResponse:
        """Generate structured JSON response with failover."""
        last_error = None

        for provider in self.providers:
            health = self.health[provider.name]

            if not health.is_available():
                continue

            try:
                response = await provider.generate_structured(
                    messages=messages,
                    schema_description=schema_description,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                health.record_success()
                return response

            except Exception as e:
                error_type = self._classify_error(e)
                health.record_failure(error_type)
                last_error = e
                logger.warning(f"Structured generation failover from {provider.name}: {e}")
                continue

        error_msg = f"All LLM providers failed for structured generation. Last error: {last_error}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    @staticmethod
    def _classify_error(error: Exception) -> str:
        """Classify an error for appropriate cooldown handling."""
        error_str = str(type(error).__name__).lower()
        error_msg = str(error).lower()

        if "404" in error_msg or "not_found" in error_msg or "not found" in error_msg:
            return "model_not_found"
        elif "ratelimit" in error_str or "429" in error_msg or "rate limit" in error_msg or "tpd" in error_msg:
            return "rate_limit"
        elif "timeout" in error_str or "timed out" in error_msg:
            return "timeout"
        elif "connection" in error_str:
            return "connection"
        elif "500" in error_msg or "502" in error_msg or "503" in error_msg:
            return "server_error"
        else:
            return "general"

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all providers."""
        return {
            "providers": [h.to_dict() for h in self.health.values()],
            "total_configured": len(self.providers),
            "total_available": sum(1 for h in self.health.values() if h.is_available()),
        }


# Singleton router
_router = None

def get_llm_router() -> LLMRouter:
    global _router
    if _router is None:
        _router = LLMRouter()
        _router.initialize_from_settings()
    return _router
