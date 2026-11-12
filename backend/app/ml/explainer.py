"""
ML Explainability module for AutoCare AI.
Calculates local per-prediction feature contributions using SHAP TreeExplainer for Random Forest.

IMPORTANT: get_local_explainer() preferentially reuses the already-loaded model from the
MaintenancePredictor singleton to avoid double-loading and path resolution failures.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path

from backend.app.core.logging import get_logger

logger = get_logger(__name__)

# Standardized human-readable feature display names for user-facing outputs
FEATURE_DISPLAY_NAMES_CLEAN: Dict[str, str] = {
    "Vehicle_Age": "Vehicle age",
    "Reported_Issues": "Reported issues",
    "Maintenance_History": "Maintenance history",
    "Last_Service_Date": "Last service date",
    "Service_History": "Service history",
    "Accident_History": "Accident history",
    "Brake_Condition": "Brake condition",
    "Battery_Status": "Battery status",
    "Tire_Condition": "Tire condition",
    "Mileage": "Mileage",
    "Engine_Size": "Engine size",
    "Odometer_Reading": "Odometer reading",
    "Fuel_Type": "Fuel type",
    "Transmission_Type": "Transmission type",
    "Owner_Type": "Owner history",
    "Insurance_Premium": "Insurance premium",
    "Fuel_Efficiency": "Fuel efficiency",
    "Warranty_Expiry_Date": "Warranty expiry date",
    "Vehicle_Model": "Vehicle model",
}


class LocalPredictionExplainer:
    """
    Computes local feature contributions for an individual Random Forest prediction.
    Uses SHAP TreeExplainer to determine which confirmed vehicle factors
    most influenced the maintenance risk score and in which direction.
    """

    def __init__(self, model: Any = None, preprocessor: Any = None):
        self.model = model
        self.preprocessor = preprocessor
        self._tree_explainer: Optional[Any] = None
        self._shap_version: Optional[str] = None
        self._init_explainer()

    def _init_explainer(self):
        if self.model is None:
            logger.warning("[EXPLAINER] No model provided — explainer is non-functional.")
            return
        try:
            import shap
            self._shap_version = shap.__version__
            self._tree_explainer = shap.TreeExplainer(self.model)
            logger.info(
                "[EXPLAINER] Initialized SHAP TreeExplainer. "
                "model=%s  shap_version=%s",
                type(self.model).__name__,
                self._shap_version,
            )
        except Exception:
            logger.exception(
                "[EXPLAINER] Could not initialize SHAP TreeExplainer. "
                "model_type=%s",
                type(self.model).__name__ if self.model else "None",
            )
            self._tree_explainer = None

    def explain_prediction(
        self,
        feature_state: Dict[str, Dict[str, Any]],
        prediction_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Compute local feature contributions for confirmed features in feature_state.

        Returns:
            Dict containing:
                available (bool): Whether local explainability succeeded
                risk_level (str): "HIGH" / "LOW" / "MEDIUM"
                model_score (float): Probability of class 1
                prediction_probability (float): Probability of class 1
                shap_base_value (float): Base value from TreeExplainer (class-1)
                main_factor (Dict): Top contributing feature details
                top_contributors (List[Dict]): Ranked feature contributions
                top_factors (List[Dict]): Alias for top_contributors
                positive_contributors (List[Dict]): Features pushing toward higher risk
                negative_contributors (List[Dict]): Features pushing toward lower risk
                summary_explanation (str): Natural language explanation snippet
        """
        risk = prediction_result.get("risk_level", "LOW") if prediction_result else "LOW"
        score = prediction_result.get("model_score") if prediction_result else None
        if score is None and prediction_result:
            score = prediction_result.get("maintenance_probability")

        default_fail: Dict[str, Any] = {
            "available": False,
            "reason": "Local explainability engine not initialized",
            "risk_level": risk,
            "model_score": score,
            "prediction_probability": score,
            "shap_base_value": None,
            "main_factor": None,
            "top_contributors": [],
            "top_factors": [],
            "positive_contributors": [],
            "negative_contributors": [],
            "summary_explanation": None,
        }

        # Guard: explainer not available
        if self.model is None or self.preprocessor is None:
            logger.warning("[SHAP_FAIL] model=%s preprocessor=%s — explainer non-functional",
                           self.model, self.preprocessor)
            default_fail["reason"] = "Model or preprocessor not loaded"
            return default_fail

        if self._tree_explainer is None:
            logger.warning("[SHAP_FAIL] TreeExplainer is None — attempting re-init")
            self._init_explainer()
            if self._tree_explainer is None:
                default_fail["reason"] = "SHAP TreeExplainer could not be initialized"
                return default_fail

        # ── SHAP_START ───────────────────────────────────────────────────────
        logger.info(
            "[SHAP_START] model_type=%s  shap_version=%s",
            type(self.model).__name__,
            self._shap_version,
        )

        try:
            # 1. Transform feature state to model input dataframe
            X = self.preprocessor.transform_feature_state(feature_state)

            logger.info(
                "[SHAP_INPUT_SHAPE] X.shape=%s  n_columns=%d",
                X.shape,
                len(X.columns),
            )
            logger.info("[SHAP_INPUT_FEATURE_NAMES] %s", list(X.columns))

            # Sanity-check: model expects exactly this many features
            n_model_features = getattr(self.model, "n_features_in_", None)
            if n_model_features is not None and X.shape[1] != n_model_features:
                msg = (
                    f"Feature count mismatch: X has {X.shape[1]} columns "
                    f"but model expects {n_model_features}"
                )
                logger.error("[SHAP_FAIL] %s", msg)
                default_fail["reason"] = msg
                return default_fail

            # 2. Compute local SHAP values
            sv = self._tree_explainer.shap_values(X)

            logger.info(
                "[SHAP_OUTPUT_SHAPE] type=%s  shape=%s  is_list=%s",
                type(sv).__name__,
                getattr(sv, "shape", None) if not isinstance(sv, list) else f"list[{len(sv)}]",
                isinstance(sv, list),
            )
            if isinstance(sv, list):
                for i, arr in enumerate(sv):
                    logger.info("[SHAP_OUTPUT_SHAPE] sv[%d].shape=%s", i, arr.shape)

            # 3. Resolve base expected value for class-1
            base_val: Optional[float] = None
            ev = getattr(self._tree_explainer, "expected_value", None)
            if ev is not None:
                if isinstance(ev, np.ndarray):
                    # Binary classification: ev shape = (2,)
                    base_val = float(ev[1]) if len(ev) > 1 else float(ev[0])
                elif isinstance(ev, (list, tuple)):
                    base_val = float(ev[1]) if len(ev) > 1 else float(ev[0])
                else:
                    try:
                        base_val = float(ev)
                    except (TypeError, ValueError):
                        base_val = None

            # 4. Extract class-1 SHAP values — handle all SHAP version shapes
            #
            # SHAP 0.41+ with RandomForest binary classifier may return:
            #   (a) ndarray shape (1, n_features, 2)  ← SHAP 0.44+
            #   (b) list of 2 ndarrays each (1, n_features)
            #   (c) ndarray shape (1, n_features)  ← single output
            #
            shap_class1: Optional[np.ndarray] = None

            if isinstance(sv, np.ndarray):
                if sv.ndim == 3:
                    # Shape: (n_samples, n_features, n_classes)
                    shap_class1 = sv[0, :, 1]
                    logger.info("[SHAP_VERSION_BRANCH] 3D ndarray → sv[0,:,1]")
                elif sv.ndim == 2:
                    shap_class1 = sv[0, :]
                    logger.info("[SHAP_VERSION_BRANCH] 2D ndarray → sv[0,:]")
                else:
                    shap_class1 = sv.flatten()
                    logger.info("[SHAP_VERSION_BRANCH] 1D ndarray → sv.flatten()")
            elif isinstance(sv, list):
                if len(sv) >= 2:
                    # Binary: sv[0]=class-0 contributions, sv[1]=class-1 contributions
                    arr = sv[1]
                    shap_class1 = arr[0] if arr.ndim == 2 else arr
                    logger.info("[SHAP_VERSION_BRANCH] list[2] → sv[1][0]")
                elif len(sv) == 1:
                    arr = sv[0]
                    shap_class1 = arr[0] if arr.ndim == 2 else arr
                    logger.info("[SHAP_VERSION_BRANCH] list[1] → sv[0][0]")
                else:
                    msg = "SHAP returned empty list"
                    logger.error("[SHAP_FAIL] %s", msg)
                    default_fail["reason"] = msg
                    return default_fail
            else:
                msg = f"Unexpected SHAP output structure: {type(sv)}"
                logger.error("[SHAP_FAIL] %s", msg)
                default_fail["reason"] = msg
                return default_fail

            # Ensure 1-D
            if shap_class1.ndim != 1:
                shap_class1 = shap_class1.flatten()

            logger.info(
                "[SHAP_CLASS1_SHAPE] len=%d  sum=%.6f",
                len(shap_class1),
                float(np.sum(shap_class1)),
            )

            # 5. Verify lengths match before zipping
            if len(shap_class1) != len(X.columns):
                msg = (
                    f"SHAP output length {len(shap_class1)} "
                    f"!= preprocessed feature count {len(X.columns)}"
                )
                logger.error("[SHAP_FAIL] %s", msg)
                default_fail["reason"] = msg
                return default_fail

            # 6. Aggregate: collapse `_missing` indicator columns back into their base feature
            feat_contribs: Dict[str, float] = {}
            for col_name, val in zip(X.columns, shap_class1):
                base_feat = col_name[:-8] if col_name.endswith("_missing") else col_name
                feat_contribs[base_feat] = feat_contribs.get(base_feat, 0.0) + float(val)

            # 7. Keep only features the user has confirmed as "known"
            known_contribs: List[Dict[str, Any]] = []
            for feat_name, contrib in feat_contribs.items():
                fs_entry = feature_state.get(feat_name, {})
                if fs_entry.get("status") == "known" and fs_entry.get("value") is not None:
                    user_val = fs_entry["value"]
                    disp_name = FEATURE_DISPLAY_NAMES_CLEAN.get(
                        feat_name, feat_name.replace("_", " ").capitalize()
                    )
                    direction = "higher_risk" if contrib > 0 else "lower_risk"
                    known_contribs.append({
                        "feature": feat_name,
                        "display_name": disp_name,
                        "value": user_val,
                        "contribution": round(contrib, 4),
                        "shap_value": round(contrib, 4),
                        "abs_contribution": round(abs(contrib), 4),
                        "direction": direction,
                    })

            # 8. Rank by absolute contribution strength (largest impact first)
            known_contribs.sort(key=lambda item: item["abs_contribution"], reverse=True)

            positive = [f for f in known_contribs if f["direction"] == "higher_risk"]
            negative = [f for f in known_contribs if f["direction"] == "lower_risk"]
            main_factor = known_contribs[0] if known_contribs else None

            # ── SHAP_COMPLETE ─────────────────────────────────────────────
            logger.info("[SHAP_COMPLETE] base_value=%.6f  main_factor=%s  n_contributors=%d",
                        base_val or 0.0,
                        main_factor["feature"] if main_factor else "none",
                        len(known_contribs))
            logger.info(
                "[SHAP_TOP_FEATURES] %s",
                [(f["feature"], f["contribution"], f["direction"]) for f in known_contribs],
            )

            # 9. Build natural-language summary snippet
            explanation_parts: List[str] = []
            if main_factor:
                dir_desc = "higher risk" if main_factor["direction"] == "higher_risk" else "lower risk"
                explanation_parts.append(
                    f"The strongest factor in this prediction was the {main_factor['display_name'].lower()} "
                    f"(reported as {main_factor['value']}), which pushed the assessment toward {dir_desc}."
                )
            if len(known_contribs) > 1:
                secondary = known_contribs[1]
                sec_dir = "higher risk" if secondary["direction"] == "higher_risk" else "lower risk"
                explanation_parts.append(
                    f"Additionally, {secondary['display_name'].lower()} ({secondary['value']}) "
                    f"contributed toward {sec_dir}."
                )
            summary_text = " ".join(explanation_parts)

            # SHAP audit log
            logger.info(
                "[SHAP_AUDIT] prediction=%s  prob=%s  base_val=%s  "
                "main_factor=%s  top_contributions=%s",
                risk,
                score,
                base_val,
                main_factor["feature"] if main_factor else None,
                [(f["feature"], f["contribution"], f["direction"]) for f in known_contribs],
            )

            return {
                "available": True,
                "risk_level": risk,
                "model_score": score,
                "prediction_probability": score,
                "shap_base_value": base_val,
                "main_factor": main_factor,
                "top_contributors": known_contribs,
                "top_factors": known_contribs,
                "positive_contributors": positive,
                "negative_contributors": negative,
                "summary_explanation": summary_text,
            }

        except Exception:
            logger.exception(
                "[SHAP_FAIL] Unexpected exception in explain_prediction. "
                "model_type=%s  feature_state_keys=%s",
                type(self.model).__name__ if self.model else "None",
                list(feature_state.keys()),
            )
            default_fail["reason"] = "SHAP calculation raised an unexpected exception (see logs)"
            return default_fail


# ---------------------------------------------------------------------------
# Global singleton — preferentially reuses the predictor's loaded model
# ---------------------------------------------------------------------------
_global_explainer: Optional[LocalPredictionExplainer] = None


def get_local_explainer() -> LocalPredictionExplainer:
    """
    Get or create singleton LocalPredictionExplainer.

    Priority order for model source:
    1. Already-loaded MaintenancePredictor singleton (avoids double-loading)
    2. Direct joblib load from configured paths
    3. Empty (non-functional) instance with warning
    """
    global _global_explainer
    if _global_explainer is not None:
        return _global_explainer

    # --- Option 1: reuse predictor singleton's already-loaded model --------
    try:
        from backend.app.services.prediction_service import get_predictor
        predictor = get_predictor()
        if predictor.is_loaded and predictor.model is not None and predictor.preprocessor is not None:
            logger.info(
                "[EXPLAINER] Reusing predictor singleton's model (%s) and preprocessor.",
                type(predictor.model).__name__,
            )
            # Use the explainer already attached to the predictor if it has one
            if predictor.explainer is not None and predictor.explainer._tree_explainer is not None:
                _global_explainer = predictor.explainer
                logger.info("[EXPLAINER] Reusing predictor.explainer directly.")
            else:
                _global_explainer = LocalPredictionExplainer(predictor.model, predictor.preprocessor)
            return _global_explainer
    except Exception:
        logger.exception("[EXPLAINER] Could not reuse predictor singleton — falling back to direct load.")

    # --- Option 2: load model directly from configured paths ---------------
    try:
        import joblib
        from backend.app.ml.preprocessing import MaintenancePreprocessor
        from backend.app.core.config import get_settings

        settings = get_settings()
        model_path = settings.project_root / settings.ml_model_path
        pipe_path = settings.project_root / settings.ml_pipeline_path

        if not (model_path.exists() and pipe_path.exists()):
            # Try candidate roots
            _candidate_roots = [
                Path("."),
                Path(__file__).resolve().parent.parent.parent.parent,
                Path(__file__).resolve().parent.parent.parent.parent.parent,
            ]
            for root in _candidate_roots:
                mp = root / "models" / "maintenance_model.joblib"
                pp = root / "models" / "preprocessing_pipeline.joblib"
                if mp.exists() and pp.exists():
                    model_path, pipe_path = mp, pp
                    break

        if model_path.exists() and pipe_path.exists():
            model = joblib.load(model_path)
            preprocessor = MaintenancePreprocessor.load(str(pipe_path))
            _global_explainer = LocalPredictionExplainer(model, preprocessor)
            logger.info("[EXPLAINER] Loaded model directly from %s", model_path)
        else:
            logger.error(
                "[EXPLAINER] Model files not found at %s or %s — explainer non-functional.",
                model_path,
                pipe_path,
            )
            _global_explainer = LocalPredictionExplainer()
    except Exception:
        logger.exception("[EXPLAINER] Direct model load failed — explainer will be non-functional.")
        _global_explainer = LocalPredictionExplainer()

    return _global_explainer


def reset_explainer_singleton() -> None:
    """Force reset the explainer singleton (used in tests or after model reload)."""
    global _global_explainer
    _global_explainer = None
    logger.info("[EXPLAINER] Singleton reset.")


# ---------------------------------------------------------------------------
# UI-facing helpers
# ---------------------------------------------------------------------------

def explanation_for_ui(explanation: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return a display-safe local explanation payload for API/UI consumers."""
    explanation = explanation or {}
    available = bool(explanation.get("available"))
    payload: Dict[str, Any] = {
        "available": available,
        "risk": explanation.get("risk_level"),
        "probability": explanation.get("model_score") or explanation.get("prediction_probability"),
        "shap_base_value": explanation.get("shap_base_value"),
        "top_factors": [],
    }
    if not available:
        payload["reason"] = explanation.get("reason")
        return payload

    payload["top_factors"] = [
        {
            "feature": factor.get("feature"),
            "display_name": factor["display_name"],
            "value": factor.get("value"),
            "contribution": factor["contribution"],
            "shap_value": factor.get("shap_value", factor["contribution"]),
            "direction": factor["direction"],
        }
        for factor in explanation.get("top_contributors", [])[:5]
        if factor.get("display_name") and "contribution" in factor and factor.get("direction")
    ]
    return payload


def format_factor_explanation_text(
    explanation: Optional[Dict[str, Any]],
    prediction: Optional[Dict[str, Any]] = None,
) -> str:
    """Format answer for 'What is the main factor?' using the actual strongest SHAP contributor."""
    if not explanation or not explanation.get("available") or not explanation.get("main_factor"):
        logger.warning(
            "[EXPLAINER] format_factor_explanation_text called with unavailable explanation. "
            "available=%s  reason=%s",
            (explanation or {}).get("available"),
            (explanation or {}).get("reason"),
        )
        return (
            "I don't have a reliable per-prediction factor breakdown available for this result, "
            "so I will not guess which single factor drove it."
        )

    mf = explanation["main_factor"]
    dir_desc = "higher-risk" if mf["direction"] == "higher_risk" else "lower-risk"
    val_str = f" (reported as {mf['value']})" if mf.get("value") is not None else ""
    lines = [
        f"The strongest contributor to this particular risk prediction was {mf['display_name']}{val_str}. "
        f"It pushed the model toward the {dir_desc} side of the assessment."
    ]

    top = explanation.get("top_contributors", [])
    if len(top) > 1:
        sec = top[1]
        sec_dir = "higher-risk" if sec["direction"] == "higher_risk" else "lower-risk"
        sec_val = f" ({sec['value']})" if sec.get("value") is not None else ""
        lines.append(f"Additionally, {sec['display_name']}{sec_val} contributed toward the {sec_dir} side.")

    lines.append(
        "Other confirmed factors also contributed, so this does not mean that "
        f"{mf['display_name']} is the mechanical cause of the problem."
    )
    return " ".join(lines)


def format_risk_explanation_text(
    explanation: Optional[Dict[str, Any]],
    prediction: Optional[Dict[str, Any]] = None,
) -> str:
    """Format answer for 'Why is the risk high/low?' or 'What is the reason for this output?'."""
    risk = (prediction or {}).get("risk_level", (explanation or {}).get("risk_level", "LOW"))
    score = (prediction or {}).get("model_score", (explanation or {}).get("model_score"))
    score_str = f" ({score * 100:.1f}%)" if score is not None else ""

    if not explanation or not explanation.get("available") or not explanation.get("top_contributors"):
        return (
            f"The model's {risk} maintenance risk estimate{score_str} reflects the combined pattern "
            "of all confirmed vehicle factors. This is a model-based maintenance-risk estimate, not a definitive mechanical diagnosis."
        )

    top = explanation.get("top_contributors", [])
    pos = [
        f"{f['display_name']} ({f['value']})" if f.get("value") is not None else f["display_name"]
        for f in top if f["direction"] == "higher_risk"
    ]
    neg = [
        f"{f['display_name']} ({f['value']})" if f.get("value") is not None else f["display_name"]
        for f in top if f["direction"] == "lower_risk"
    ]

    parts: List[str] = []
    if risk == "HIGH":
        if pos:
            parts.append(
                f"The assessment is on the higher-risk side mainly because {', '.join(pos[:2])} "
                f"pushed the prediction upward."
            )
            if neg:
                parts.append(f"Other factors such as {', '.join(neg[:2])} offset some of that effect.")
        else:
            parts.append("The assessment is on the higher-risk side based on the combined pattern of your vehicle's confirmed details.")
    else:
        if neg:
            parts.append(
                f"The assessment is on the lower-risk side mainly because {', '.join(neg[:2])} "
                f"kept the prediction lower."
            )
            if pos:
                parts.append(f"Factors such as {', '.join(pos[:2])} added some upward risk.")
        else:
            parts.append("The assessment is on the lower-risk side based on the confirmed vehicle details.")

    parts.append("This is a model-based maintenance-risk estimate, not a definitive mechanical diagnosis.")
    return " ".join(parts)


def format_score_factors_text(
    explanation: Optional[Dict[str, Any]],
    prediction: Optional[Dict[str, Any]] = None,
) -> str:
    """Format answer for 'What affected the score?' returning the top 2-3 contributors."""
    if not explanation or not explanation.get("available") or not explanation.get("top_contributors"):
        return (
            "I don't have a reliable per-prediction factor breakdown available for this result, "
            "so I will not guess which factors drove the score."
        )

    top = explanation.get("top_contributors", [])
    selected: List[Dict[str, Any]] = []
    higher = [f for f in top if f["direction"] == "higher_risk"]
    lower = [f for f in top if f["direction"] == "lower_risk"]

    if higher:
        selected.append(higher[0])
    if lower and lower[0] not in selected:
        selected.append(lower[0])
    for f in top:
        if f not in selected and len(selected) < 3:
            selected.append(f)

    lines = ["Here are the strongest model contributors to your maintenance risk score:"]
    for f in selected:
        dir_label = "higher risk" if f["direction"] == "higher_risk" else "lower risk"
        val_suffix = f" (reported as {f['value']})" if f.get("value") is not None else ""
        lines.append(f"• {f['display_name']}{val_suffix} — {dir_label}")

    lines.append("The model calculates the score from all confirmed details together rather than a single component.")
    return "\n".join(lines)
