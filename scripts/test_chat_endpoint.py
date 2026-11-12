import sys
import asyncio
from pathlib import Path

root = Path(__file__).parent.parent
sys.path.insert(0, str(root))

from httpx import AsyncClient, ASGITransport
from backend.app.main import app

async def test_health():
    print("--- STEP 2: Verify GET /api/health works ---")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res = await client.get("/api/health")
        print(f"Status Code: {res.status_code}")
        print(f"Body: {res.json()}")
        if res.status_code == 200:
            print("PASS: GET /api/health works!")
        else:
            print("FAIL: GET /api/health failed")

async def test_post_chat():
    print("\n--- STEP 8: Verify POST /api/chat is reached ---")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test", timeout=35.0) as client:
        try:
            res = await client.post("/api/chat", json={"message": "My car is 5 years old."})
            print(f"Status Code: {res.status_code}")
            print(f"Response: {res.json()}")
            if res.status_code == 200:
                print("PASS: POST /api/chat works!")
            else:
                print(f"FAIL: POST /api/chat returned status {res.status_code}")
        except Exception as e:
            print(f"FAIL: POST /api/chat exception: {type(e).__name__}: {e}")

async def test_post_chat_stream():
    print("\n--- STEP 8 (Stream): Verify POST /api/chat/stream is reached ---")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test", timeout=35.0) as client:
        try:
            async with client.stream("POST", "/api/chat/stream", json={"message": "My car is 5 years old."}) as res:
                print(f"Stream Status Code: {res.status_code}")
                chunks = []
                async for chunk in res.aiter_text():
                    chunks.append(chunk)
                    print(f"Chunk received: {repr(chunk[:100])}")
                print(f"Total chunks: {len(chunks)}")
                print("PASS: POST /api/chat/stream works!")
        except Exception as e:
            print(f"FAIL: POST /api/chat/stream exception: {type(e).__name__}: {e}")

async def main():
    await test_health()
    await test_post_chat()
    await test_post_chat_stream()

if __name__ == "__main__":
    asyncio.run(main())
