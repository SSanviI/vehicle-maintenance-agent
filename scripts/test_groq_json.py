import sys
import asyncio
import json
import re
from pathlib import Path

root = Path(__file__).parent.parent
sys.path.insert(0, str(root))

from backend.app.core.config import get_settings
from backend.app.llm.providers.openai_compatible import OpenAICompatibleProvider
from backend.app.llm.base import LLMMessage

def clean_json_response(content: str) -> dict:
    """Remove reasoning tags like <think>...</think> and parse JSON."""
    # Strip thinking tags if present
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    # Strip markdown codeblocks if present
    if content.startswith("```"):
        content = re.sub(r'^```[a-zA-Z]*\n', '', content)
        content = re.sub(r'\n```$', '', content)
    content = content.strip()
    return json.loads(content)

async def main():
    settings = get_settings()
    provider = OpenAICompatibleProvider(
        name="groq",
        model=settings.primary_llm_model,
        api_key=settings.primary_llm_api_key,
        base_url=settings.primary_llm_base_url
    )

    system_prompt = """You are a vehicle information extraction assistant. Extract structured vehicle data.
Respond ONLY with a JSON object. No Markdown, no thinking text outside or inside.

JSON Schema:
{
  "intent": "vehicle_information",
  "extracted_features": {"Vehicle_Age": 5},
  "unknown_features": [],
  "responding_to_feature": null
}"""

    messages = [
        LLMMessage(role="system", content=system_prompt),
        LLMMessage(role="user", content="My car is 5 years old and I have noticed 3 problems recently.")
    ]

    print("Test 1: Calling Groq WITHOUT response_format...")
    try:
        res1 = await provider.generate(messages=messages, temperature=0.1, max_tokens=300)
        print("Raw Content:", repr(res1.content))
        data = clean_json_response(res1.content)
        print("Parsed JSON successfully:", data)
    except Exception as e:
        print("Test 1 Failed:", e)

    print("\nTest 2: Calling Groq WITH response_format={'type': 'json_object'}...")
    try:
        res2 = await provider.generate(messages=messages, temperature=0.1, max_tokens=300, response_format={"type": "json_object"})
        print("Raw Content:", repr(res2.content))
        data2 = clean_json_response(res2.content)
        print("Parsed JSON successfully:", data2)
    except Exception as e:
        print("Test 2 Failed (Groq 400 error):", e)

if __name__ == "__main__":
    asyncio.run(main())
