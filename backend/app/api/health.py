"""
Health Check Endpoints — System, LLM router, and ML model status checks.
"""

from fastapi import APIRouter
from backend.app.services.prediction_service import get_predictor
from backend.app.llm.router import get_llm_router
from backend.app.services.cache_service import get_redis_client

router = APIRouter(prefix="/api/health", tags=["Health"])


@router.get("")
async def system_health():
    """Overall system health check."""
    predictor = get_predictor()
    router = get_llm_router()
    redis_client = await get_redis_client()

    return {
        "status": "healthy",
        "components": {
            "ml_model": "loaded" if predictor.is_loaded else "unloaded",
            "llm_router": "configured" if router.providers else "unconfigured",
            "redis": "connected" if redis_client else "disconnected",
        }
    }


@router.get("/llm")
async def llm_health():
    """LLM router health & provider status."""
    router = get_llm_router()
    return router.get_health_status()


@router.get("/ml")
async def ml_health():
    """ML model status & metadata."""
    predictor = get_predictor()
    return {
        "is_loaded": predictor.is_loaded,
        "metadata": predictor.model_metadata or {},
    }
