"""
Pydantic schemas for prediction-related data.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class PredictionRequest(BaseModel):
    """Request for ML prediction with structured feature state."""
    feature_state: Dict[str, Dict[str, Any]]
    conversation_id: Optional[str] = None


class PredictionResponse(BaseModel):
    """ML prediction result."""
    prediction_id: Optional[str] = None
    prediction: str  # "maintenance_required" or "no_maintenance_needed"
    prediction_class: int  # 0 or 1
    model_score: float
    risk_level: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    features_used: List[str]
    missing_features: List[str]
    feature_sources: Dict[str, str]
    confidence_note: Optional[str] = None
    previous_score: Optional[float] = None

    @staticmethod
    def get_risk_level(score: float) -> str:
        if score >= 0.85:
            return "CRITICAL"
        elif score >= 0.65:
            return "HIGH"
        elif score >= 0.45:
            return "MEDIUM"
        else:
            return "LOW"


class PredictionHistoryResponse(BaseModel):
    predictions: List[Dict[str, Any]]
    total: int
