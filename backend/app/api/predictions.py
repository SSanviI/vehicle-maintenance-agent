"""
Predictions API Endpoint — Standalone ML maintenance prediction endpoint.
Note: Prediction comes strictly from ML, never from LLM.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Dict, Any

from backend.app.schemas.prediction import PredictionRequest, PredictionResponse
from backend.app.services.prediction_service import PredictionService
from backend.app.memory.repository import MemoryRepository
from backend.app.database.session import get_db

router = APIRouter(prefix="/api/predictions", tags=["Predictions"])


@router.post("", response_model=PredictionResponse)
async def make_prediction(request: PredictionRequest):
    """
    Direct ML prediction endpoint.
    Accepts structured feature state dict and returns ML prediction model score & class.
    The LLM is NEVER called in this endpoint.
    """
    service = PredictionService()
    result = service.make_prediction(request.feature_state)

    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])

    return PredictionResponse(
        prediction=result["prediction"],
        prediction_class=result["prediction_class"],
        model_score=result["model_score"],
        risk_level=result["risk_level"],
        features_used=result["features_used"],
        missing_features=result["missing_features"],
        feature_sources=result["feature_sources"],
        confidence_note=result.get("confidence_note"),
    )


@router.get("/{conversation_id}")
async def get_predictions_history(
    conversation_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get history of predictions for a conversation."""
    repo = MemoryRepository(db)
    conv = await repo.get_or_create_conversation(conversation_id)

    preds = [
        {
            "id": str(p.id),
            "prediction_class": p.prediction_class,
            "model_score": p.model_score,
            "features_used": p.features_used,
            "missing_features": p.missing_features,
            "created_at": p.created_at,
        }
        for p in (conv.predictions or [])
    ]

    return {"conversation_id": conversation_id, "predictions": preds, "total": len(preds)}
