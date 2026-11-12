"""Test the configured Groq model without exposing credentials in source code."""

import asyncio

import httpx

from backend.app.core.config import get_settings


async def main():
    settings = get_settings()
    if not settings.primary_llm_api_key:
        raise RuntimeError("PRIMARY_LLM_API_KEY must be set in the environment before running this test.")

    headers = {
        "Authorization": f"Bearer {settings.primary_llm_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings.primary_llm_model,
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 10,
    }

    print(f"Testing configured Groq model: {settings.primary_llm_model}")
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(
            f"{settings.primary_llm_base_url}/chat/completions",
            headers=headers,
            json=payload,
        )
    print(f"Status: {response.status_code}")
    if response.status_code != 200:
        print(f"Response: {response.text[:150]}")


if __name__ == "__main__":
    asyncio.run(main())
