"""
Test streaming endpoint /api/chat/stream directly against FastAPI app.
"""

import asyncio
import json
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from httpx import AsyncClient, ASGITransport
from backend.app.main import app

async def test_streaming_api():
    print("--> Testing /api/chat/stream via ASGI transport...")
    transport = ASGITransport(app=app)
    
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "message": "My brakes are worn out and battery seems weak.",
            "conversation_id": None
        }
        
        chunks_received = []
        full_text = ""
        prediction_received = None
        explanation_received = None
        
        async with client.stream("POST", "/api/chat/stream", json=payload, timeout=30.0) as response:
            print(f"--> Status code: {response.status_code}")
            assert response.status_code == 200, f"Unexpected status: {response.status_code}"
            
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    try:
                        event = json.loads(data_str)
                        chunks_received.append(event)
                        event_type = event.get("type")
                        if event_type == "text":
                            full_text += event.get("content", "")
                        elif event_type == "prediction":
                            prediction_received = event.get("data")
                        elif event_type == "explanation":
                            explanation_received = event.get("data")
                        elif event_type == "done":
                            print("--> SSE 'done' event received.")
                    except json.JSONDecodeError:
                        pass

        print(f"--> Total SSE events received: {len(chunks_received)}")
        print(f"--> Streamed text length: {len(full_text)}")
        print(f"--> Streamed text content: {full_text}")
        print(f"--> Prediction received: {prediction_received is not None}")
        if prediction_received:
            print(f"    Risk: {prediction_received.get('risk_level')}, Score: {prediction_received.get('model_score')}")
        print(f"--> Explanation received: {explanation_received is not None}")
        if explanation_received:
            print(f"    Available: {explanation_received.get('available')}")
            print(f"    Top factors: {explanation_received.get('top_factors')}")
            
    print("\n--> STREAMING API TEST SUCCESSFUL!")

if __name__ == "__main__":
    asyncio.run(test_streaming_api())
