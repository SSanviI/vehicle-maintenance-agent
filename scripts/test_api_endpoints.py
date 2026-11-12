"""
API Endpoint verification script for Phase 2
"""

import asyncio
import httpx

BASE_URL = "http://localhost:7860"

async def test_apis():
    print("=" * 80)
    print("  VERIFYING BACKEND API ENDPOINTS")
    print("=" * 80)

    async with httpx.AsyncClient(timeout=10.0) as client:
        # Health
        resp = await client.get(f"{BASE_URL}/api/health")
        print(f"GET /api/health                -> Status: {resp.status_code}")
        print(f"   Body: {resp.json()}")

        # LLM Health
        resp = await client.get(f"{BASE_URL}/api/health/llm")
        print(f"\nGET /api/health/llm            -> Status: {resp.status_code}")
        print(f"   Body: {resp.json()}")

        # ML Health
        resp = await client.get(f"{BASE_URL}/api/health/ml")
        print(f"\nGET /api/health/ml             -> Status: {resp.status_code}")
        print(f"   Body: {resp.json()}")

        # Create Conversation
        resp = await client.post(f"{BASE_URL}/api/conversations", json={"title": "Test Chat"})
        print(f"\nPOST /api/conversations        -> Status: {resp.status_code}")
        conv_id = resp.json().get("id")
        print(f"   Conversation ID: {conv_id}")

        # Post Message
        chat_req = {
            "conversation_id": conv_id,
            "message": "My car is 10 years old."
        }
        resp = await client.post(f"{BASE_URL}/api/chat", json=chat_req)
        print(f"\nPOST /api/chat                 -> Status: {resp.status_code}")
        res_data = resp.json()
        print(f"   Agent Response: \"{res_data.get('response')}\"")
        if res_data.get("prediction"):
            p = res_data["prediction"]
            print(f"   ML Prediction: ID={p.get('prediction_id')[:8]}, score={p.get('model_score')}")

if __name__ == "__main__":
    asyncio.run(test_apis())
