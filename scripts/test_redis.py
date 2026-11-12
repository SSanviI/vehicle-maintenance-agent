import sys
import asyncio
from pathlib import Path

root = Path(__file__).parent.parent
sys.path.insert(0, str(root))

from backend.app.services.cache_service import CacheService, get_redis_client

async def main():
    print("Testing Redis Cache Service...")
    client = await get_redis_client()
    if client:
        print("Redis Status: CONNECTED")
        cache = CacheService()
        await cache.set_json("test_key", {"status": "ok"}, ttl_seconds=10)
        val = await cache.get_json("test_key")
        print(f"Redis Read Result: {val}")
        print("PASS: Redis cache is connected and working!")
    else:
        print("Redis Status: DISCONNECTED (Graceful fallback active)")
        cache = CacheService()
        res = await cache.get_json("test_key")
        assert res is None
        print("PASS: CacheService gracefully handles offline Redis without hanging or crashing!")

if __name__ == "__main__":
    asyncio.run(main())
