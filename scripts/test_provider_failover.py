"""
Test provider fallback with Groq secondary
"""

import asyncio
from backend.app.llm.base import LLMMessage
from backend.app.llm.router import LLMRouter
from backend.app.llm.providers.openai_compatible import OpenAICompatibleProvider
from backend.app.core.config import get_settings

async def test_failover():
    settings = get_settings()
    if not settings.primary_llm_api_key or not settings.secondary_llm_api_key:
        raise RuntimeError("Set PRIMARY_LLM_API_KEY and SECONDARY_LLM_API_KEY in the environment before running this test.")

    router = LLMRouter()
    
    # Primary (Rate limited / intentionally broken model to force fallback)
    p1 = OpenAICompatibleProvider(
        name="primary",
        model="openai/gpt-oss-120b-invalid",
        api_key=settings.primary_llm_api_key,
        base_url="https://api.groq.com/openai/v1"
    )
    
    # Secondary
    p2 = OpenAICompatibleProvider(
        name="secondary",
        model="qwen/qwen3.6-27b",
        api_key=settings.secondary_llm_api_key,
        base_url="https://api.groq.com/openai/v1"
    )
    
    router.add_provider(p1)
    router.add_provider(p2)
    
    messages = [LLMMessage(role="user", content="Hello, test secondary provider.")]
    
    print("Testing failover from primary -> secondary...")
    try:
        res = await router.generate(messages)
        print(f"Success! Provider used: {res.provider}, Model: {res.model}")
        print(f"Response: {res.content}")
    except Exception as e:
        print(f"Failover failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_failover())
