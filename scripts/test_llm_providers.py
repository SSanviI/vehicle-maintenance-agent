"""
Test LLM providers and models independently
"""

import asyncio
import httpx
from backend.app.core.config import get_settings

async def test_provider(name, base_url, api_key, model):
    print(f"\n--- Testing {name} ---")
    print(f"Base URL: {base_url}")
    print(f"Model ID: {model}")
    print(f"API Key:  {api_key[:10]}...")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 10
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.post(f"{base_url}/chat/completions", headers=headers, json=payload)
            print(f"HTTP Status: {resp.status_code}")
            print(f"Response: {resp.text[:300]}")
        except Exception as e:
            print(f"Error: {e}")

async def main():
    settings = get_settings()
    configs = settings.get_llm_configs()
    for cfg in configs:
        await test_provider(cfg["name"], cfg["base_url"], cfg["api_key"], cfg["model"])

if __name__ == "__main__":
    asyncio.run(main())
