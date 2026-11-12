"""
Redis Cache Service — Manages short-lived state, rate limiting cooldowns, and caching.
"""

from typing import Optional, Any
import json
import redis.asyncio as redis
from backend.app.core.config import get_settings
from backend.app.core.logging import get_logger

logger = get_logger(__name__)

_redis_client: Optional[redis.Redis] = None


async def get_redis_client() -> Optional[redis.Redis]:
    """Get async Redis client."""
    global _redis_client
    if _redis_client is None:
        try:
            settings = get_settings()
            _redis_client = redis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_timeout=3.0,
            )
            await _redis_client.ping()
            logger.info("Connected to Redis successfully")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Running without Redis cache.")
            _redis_client = None
    return _redis_client


class CacheService:
    """Cache service for Redis."""

    def __init__(self):
        pass

    async def get_json(self, key: str) -> Optional[Any]:
        client = await get_redis_client()
        if not client:
            return None
        try:
            val = await client.get(key)
            return json.loads(val) if val else None
        except Exception as e:
            logger.warning(f"Redis get error for {key}: {e}")
            return None

    async def set_json(self, key: str, value: Any, ttl_seconds: int = 3600) -> bool:
        client = await get_redis_client()
        if not client:
            return False
        try:
            await client.setex(key, ttl_seconds, json.dumps(value))
            return True
        except Exception as e:
            logger.warning(f"Redis set error for {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        client = await get_redis_client()
        if not client:
            return False
        try:
            await client.delete(key)
            return True
        except Exception as e:
            logger.warning(f"Redis delete error for {key}: {e}")
            return False
