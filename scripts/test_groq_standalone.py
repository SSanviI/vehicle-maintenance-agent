import os
import sys
import asyncio
from pathlib import Path

root = Path(__file__).parent.parent
sys.path.insert(0, str(root))

from backend.app.core.config import get_settings
from backend.app.llm.providers.openai_compatible import OpenAICompatibleProvider
from backend.app.llm.base import LLMMessage
from httpx import AsyncClient, ASGITransport
from backend.app.main import app

async def run_diagnostics():
    print("==================================================")
    print("  STEP-BY-STEP DIAGNOSTIC VERIFICATION")
    print("==================================================")

    # 3. Verify backend loading .env file
    print("\n--- STEP 3: Verify backend is loading .env ---")
    # Force reload settings
    import backend.app.core.config as config_mod
    config_mod._settings = None
    settings = config_mod.get_settings()
    print(f".env loaded: YES (project_root={settings.project_root})")

    # 4. Verify PRIMARY_LLM_PROVIDER is groq
    print("\n--- STEP 4: Verify PRIMARY_LLM_PROVIDER is 'groq' ---")
    provider_name = settings.primary_llm_provider
    print(f"PRIMARY_LLM_PROVIDER: '{provider_name}'")
    if provider_name.lower() != "groq":
        print("FAIL: PRIMARY_LLM_PROVIDER is not 'groq'")
        return

    # 5. Verify Groq API key detected WITHOUT printing the key
    print("\n--- STEP 5: Verify Groq API key is detected WITHOUT printing key ---")
    api_key = settings.primary_llm_api_key
    if not api_key:
        print("FAIL: PRIMARY_LLM_API_KEY is missing or empty")
        return
    print(f"Groq API Key detected: YES (Length: {len(api_key)}, Prefix: {api_key[:4]}...)")

    # 6 & 7. Minimal Groq request with 15-second timeout
    print("\n--- STEP 6 & 7: Minimal Groq API request with 15s timeout ---")
    model_name = settings.primary_llm_model
    base_url = settings.primary_llm_base_url
    print(f"Model: {model_name}, Base URL: {base_url}")

    provider = OpenAICompatibleProvider(
        name="groq",
        model=model_name,
        api_key=api_key,
        base_url=base_url
    )
    provider.client.timeout = 15.0

    try:
        messages = [LLMMessage(role="user", content="Respond with 'OK' only.")]
        response = await asyncio.wait_for(
            provider.generate(messages=messages, temperature=0.1, max_tokens=10),
            timeout=15.0
        )
        print("PASS: Groq request succeeded!")
        print(f"Groq Response: {repr(response.content.strip())}")
    except Exception as e:
        print("FAIL: Groq request failed!")
        print(f"Error message: {type(e).__name__}: {e}")
        return

    # 8. Verify POST /api/chat is reached
    print("\n--- STEP 8: Verify POST /api/chat is reached ---")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test", timeout=35.0) as client:
        try:
            res = await client.post("/api/chat", json={"message": "My car is 5 years old."})
            print(f"Status Code: {res.status_code}")
            print(f"Response: {res.json()}")
            if res.status_code == 200:
                print("PASS: POST /api/chat is reached and succeeded!")
            else:
                print(f"FAIL: POST /api/chat returned status {res.status_code}")
        except Exception as e:
            print(f"FAIL: POST /api/chat exception: {type(e).__name__}: {e}")

    # 8 (Stream): Verify POST /api/chat/stream
    print("\n--- STEP 8 (Stream): Verify POST /api/chat/stream is reached ---")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test", timeout=35.0) as client:
        try:
            async with client.stream("POST", "/api/chat/stream", json={"message": "My car is 5 years old."}) as res:
                print(f"Stream Status Code: {res.status_code}")
                first_chunk = None
                async for chunk in res.aiter_text():
                    if not first_chunk:
                        first_chunk = chunk
                    print(f"Chunk: {repr(chunk[:100])}")
                print("PASS: POST /api/chat/stream works and streams response!")
        except Exception as e:
            print(f"FAIL: POST /api/chat/stream exception: {type(e).__name__}: {e}")

if __name__ == "__main__":
    asyncio.run(run_diagnostics())
