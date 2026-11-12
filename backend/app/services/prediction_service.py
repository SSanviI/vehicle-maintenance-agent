"""
Prediction Service — Wraps the ML predictor and manages prediction lifecycle.
"""

from typing import Dict, Any, Optional
from backend.app.ml.predict import MaintenancePredictor
from backend.app.core.config import get_settings
from backend.app.core.logging import get_logger

logger = get_logger(__name__)

# Global singleton predictor
_predictor: Optional[MaintenancePredictor] = None


def get_predictor() -> MaintenancePredictor:
    """Get initialized ML MaintenancePredictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = MaintenancePredictor()
        settings = get_settings()
        
        # Absolute paths based on project root
        model_path = settings.project_root / settings.ml_model_path
        pipeline_path = settings.project_root / settings.ml_pipeline_path
        metadata_path = settings.project_root / "models" / "model_metadata.json"

        loaded = _predictor.load(
            str(model_path),
            str(pipeline_path),
            str(metadata_path) if metadata_path.exists() else None
        )

        if not loaded:
            logger.warning("ML model failed to load. Will attempt initialization on demand.")

    return _predictor


class PredictionService:
    """Service layer for ML maintenance predictions."""

    def __init__(self, predictor: Optional[MaintenancePredictor] = None):
        self.predictor = predictor or get_predictor()

    def make_prediction(self, feature_state: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Make prediction for given feature state."""
        return self.predictor.predict(feature_state)
