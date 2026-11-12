"""
ML Prediction module — loads trained model and makes predictions.
The LLM NEVER produces predictions. Only this module does.

CRITICAL: Every inference generates a unique prediction_id.
The prediction object is the SINGLE SOURCE OF TRUTH for that turn.
All consumers (LLM explanation, prediction card, DB, frontend) MUST use
the same prediction object. No component may call predict() independently.
"""

import joblib
import json
import uuid
from datetime import datetime, timezone
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from backend.app.ml.preprocessing import (
    MaintenancePreprocessor, SELECTED_FEATURES, 
    CATEGORICAL_FEATURES, NUMERICAL_FEATURES, FEATURE_PRIORITY
)
from backend.app.core.logging import get_logger

logger = get_logger(__name__)


class MaintenancePredictor:
    """
    ML Prediction engine.
    
    CRITICAL: This is the ONLY authority for maintenance predictions.
    The LLM receives results from this module but NEVER generates predictions itself.
    """

    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.model_metadata = None
        self.explainer = None
        self.is_loaded = False
        self._inference_count = 0  # Debug counter for tracking duplicate calls

    def load(self, model_path: str, pipeline_path: str, metadata_path: Optional[str] = None):
        """Load trained model and preprocessing pipeline."""
        try:
            model_path = Path(model_path)
            pipeline_path = Path(pipeline_path)

            if not model_path.exists():
                logger.error(f"Model file not found: {model_path}")
                return False

            if not pipeline_path.exists():
                logger.error(f"Pipeline file not found: {pipeline_path}")
                return False

            self.model = joblib.load(model_path)
            self.preprocessor = MaintenancePreprocessor.load(str(pipeline_path))

            if metadata_path:
                meta_path = Path(metadata_path)
                if meta_path.exists():
                    with open(meta_path, 'r') as f:
                        self.model_metadata = json.load(f)

            from backend.app.ml.explainer import LocalPredictionExplainer
            self.explainer = LocalPredictionExplainer(self.model, self.preprocessor)

            self.is_loaded = True
            logger.info(f"ML model loaded successfully from {model_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load ML model: {e}")
            self.is_loaded = False
            return False

    def explain(self, feature_state: Dict[str, Dict[str, Any]], prediction_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate SHAP local explanation for the prediction on this feature state."""
        if not self.is_loaded or self.explainer is None:
            from backend.app.ml.explainer import get_local_explainer
            return get_local_explainer().explain_prediction(feature_state, prediction_result)
        return self.explainer.explain_prediction(feature_state, prediction_result)


    def predict(self, feature_state: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Make a maintenance prediction from partial feature state.
        
        Args:
            feature_state: Dict mapping feature names to {value, status, source}
            
        Returns:
            Dict with prediction results including class, score, features used/missing
        """
        if not self.is_loaded:
            return {
                "error": "ML model not loaded",
                "prediction": None,
                "prediction_class": None,
                "model_score": None,
            }

        try:
            # Generate unique prediction ID — this is the SINGLE SOURCE OF TRUTH
            prediction_id = str(uuid.uuid4())
            self._inference_count += 1

            # Determine which features are known vs missing
            features_used = []
            missing_features = []
            feature_sources = {}

            for feat in SELECTED_FEATURES:
                if feat in feature_state and feature_state[feat].get("status") == "known":
                    features_used.append(feat)
                    feature_sources[feat] = feature_state[feat].get("source", "user")
                else:
                    missing_features.append(feat)

            # Transform features
            X = self.preprocessor.transform_feature_state(feature_state)
            # Audit the exact current-state model inputs before the persisted
            # training preprocessor encodes missing/categorical values.
            model_input = {
                feature: (
                    feature_state[feature]["value"]
                    if feature_state.get(feature, {}).get("status") == "known"
                    else None
                )
                for feature in SELECTED_FEATURES
            }

            # Predict — SINGLE ML CALL for this prediction_id
            raw_probs = self.model.predict_proba(X)[0]
            prediction_class = int(self.model.predict(X)[0])
            
            # Audit class mapping using model.classes_
            class_map = {int(c): idx for idx, c in enumerate(getattr(self.model, "classes_", [0, 1]))}
            maint_idx = class_map.get(1, 1)
            no_maint_idx = class_map.get(0, 0)

            maint_prob = float(raw_probs[maint_idx])
            no_maint_prob = float(raw_probs[no_maint_idx])
            model_score = maint_prob

            # Determine risk level based on probability
            risk_level = "HIGH" if model_score > 0.50 else "LOW"

            # Confidence note based on how many features are known
            confidence_note = self._get_confidence_note(len(features_used), len(SELECTED_FEATURES))

            result = {
                "prediction_id": prediction_id,
                "prediction": "maintenance_required" if prediction_class == 1 else "no_maintenance_needed",
                "prediction_class": prediction_class,
                "maintenance_probability": round(maint_prob, 4),
                "no_maintenance_probability": round(no_maint_prob, 4),
                "model_score": round(model_score, 4),
                "score_semantics": "Model-estimated probability of maintenance required (Class 1)",
                "risk_level": risk_level,
                "features_used": features_used,
                "missing_features": missing_features,
                "feature_sources": feature_sources,
                "confidence_note": confidence_note,
                "total_features": len(SELECTED_FEATURES),
                "known_features_count": len(features_used),
                "feature_values_used": model_input,
                "prediction_timestamp": datetime.now(timezone.utc).isoformat(),
                "preprocessed_feature_vector": X.iloc[0].to_dict(),
            }

            logger.info(
                f"Prediction [{prediction_id[:8]}] (call #{self._inference_count}): "
                f"class={prediction_class}, maint_prob={maint_prob:.4f}, "
                f"features_used={len(features_used)}/{len(SELECTED_FEATURES)}")
            logger.info("[PREDICTION_INPUT] id=%s values=%s", prediction_id[:8], model_input)

            return result

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                "error": str(e),
                "prediction": None,
                "prediction_class": None,
                "model_score": None,
            }

    @staticmethod
    def _get_risk_level(score: float) -> str:
        if score >= 0.85:
            return "CRITICAL"
        elif score >= 0.65:
            return "HIGH"
        elif score >= 0.45:
            return "MEDIUM"
        else:
            return "LOW"

    @staticmethod
    def _get_confidence_note(known_count: int, total_count: int) -> str:
        ratio = known_count / total_count if total_count > 0 else 0
        if ratio >= 0.75:
            return "High confidence — most relevant information provided"
        elif ratio >= 0.5:
            return "Moderate confidence — some information still missing"
        elif ratio >= 0.25:
            return "Limited confidence — more information would improve accuracy"
        elif known_count > 0:
            return "Low confidence — very few details available"
        else:
            return "No vehicle information provided — prediction based on population average"

    def get_next_useful_feature(self, feature_state: Dict[str, Dict[str, Any]], 
                                declined_features: list = None) -> Optional[str]:
        """
        Determine the most useful next feature to ask about.
        Skips features that are already known or declined by user.
        """
        declined = set(declined_features or [])
        
        for feat in FEATURE_PRIORITY:
            if feat in feature_state and feature_state[feat].get("status") == "known":
                continue
            if feat in declined:
                continue
            if feat in feature_state and feature_state[feat].get("status") == "unavailable":
                continue
            return feat

        return None  # All features are known or declined
