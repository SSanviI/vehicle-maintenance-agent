import sys
import asyncio
from pathlib import Path

root = Path(__file__).parent.parent
sys.path.insert(0, str(root))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from httpx import AsyncClient, ASGITransport
from backend.app.main import app

async def main():
    print("Testing Full Agent Loop with Groq LLM & ML Engine...")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test", timeout=35.0) as client:
        # Turn 1: User says car is 5 years old
        print("\n--- Turn 1: 'My car is 5 years old.' ---")
        res1 = await client.post("/api/chat", json={"message": "My car is 5 years old."})
        assert res1.status_code == 200
        data1 = res1.json()
        conv_id = data1["conversation_id"]
        print(f"Conversation ID: {conv_id}")
        print(f"Agent Response: {repr(data1['message'])}")

        # Turn 2: User says "I don't know the brake condition" ("I don't know" handling)
        print("\n--- Turn 2: 'I don't know my brake condition.' ---")
        res2 = await client.post("/api/chat", json={"message": "I don't know my brake condition.", "conversation_id": conv_id})
        assert res2.status_code == 200
        data2 = res2.json()
        print(f"Agent Response: {repr(data2['message'])}")
        print("Full feature_state dict:", data2.get("feature_state"))
        declined = data2.get("feature_state", {}).get("declined_features", [])
        print(f"Declined features in state: {declined}")
        assert "Brake_Condition" in declined

        # Turn 3: User says "I have had 4 problems recently and battery is weak." (Incremental prediction trigger)
        print("\n--- Turn 3: 'I have had 4 problems recently and battery is weak.' ---")
        res3 = await client.post("/api/chat", json={"message": "I have had 4 problems recently and battery is weak.", "conversation_id": conv_id})
        assert res3.status_code == 200
        data3 = res3.json()
        print(f"Agent Response: {repr(data3['message'])}")
        pred = data3.get("prediction")
        if pred:
            print(f"Prediction Generated: Risk={pred['risk_level']}, Score={pred['model_score']}")
            assert pred["prediction_class"] == 1

        print("\nPASS: Complete Agent Loop executed successfully!")

if __name__ == "__main__":
    asyncio.run(main())
