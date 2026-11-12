"""Schema-bounded, deterministic feature ranking for one conversation turn."""

from __future__ import annotations

from typing import Dict, List, Optional, Iterable

from backend.app.agent.state.state import AgentState
from backend.app.ml.feature_schema import MODEL_FEATURE_SCHEMA
from backend.app.ml.preprocessing import FEATURE_PRIORITY, CATEGORICAL_FEATURES, NUMERICAL_FEATURES
from backend.app.ml.feature_schema import DATE_FEATURES
from backend.app.services.prediction_service import get_predictor
from backend.app.core.logging import get_logger

logger = get_logger(__name__)


class FeatureRankingEngine:
    """Ranks only real model features; intent changes order, never schema."""

    _INTENT_FEATURES = {
        "brake": ("Brake_Condition", "Reported_Issues", "Last_Service_Date", "Maintenance_History"),
        "battery": ("Battery_Status", "Reported_Issues", "Last_Service_Date", "Vehicle_Age"),
        "tire": ("Tire_Condition", "Reported_Issues", "Odometer_Reading", "Vehicle_Age"),
        "starting": ("Battery_Status", "Reported_Issues", "Maintenance_History", "Last_Service_Date"),
        "overheating": ("Reported_Issues", "Maintenance_History", "Last_Service_Date", "Vehicle_Age"),
        "age": ("Vehicle_Age",),
        "maintenance_history": ("Maintenance_History", "Service_History"),
        "accident": ("Accident_History",),
    }

    @classmethod
    def _feature_importances(cls) -> Dict[str, float]:
        """Aggregate RF importances from transformed columns to raw features."""
        predictor = get_predictor()
        model = predictor.model if predictor.is_loaded else None
        raw = getattr(model, "feature_importances_", None)
        if raw is None:
            return {}

        # MaintenancePreprocessor emits [feature, feature_missing] for each
        # categorical feature, followed by the same pair for numerics.
        transformed_names = []
        for feature in CATEGORICAL_FEATURES:
            transformed_names.extend((feature, f"{feature}_missing"))
        for feature in DATE_FEATURES + NUMERICAL_FEATURES:
            if feature not in CATEGORICAL_FEATURES:
                transformed_names.extend((feature, f"{feature}_missing"))
        totals = {feature: 0.0 for feature in MODEL_FEATURE_SCHEMA}
        for name, importance in zip(transformed_names, raw):
            base = name.removesuffix("_missing")
            if base in totals:
                totals[base] += float(importance)
        return totals

    @classmethod
    def rank_features(
        cls, state: AgentState, intent: Optional[str] = None, candidates: Optional[Iterable[str]] = None
    ) -> List[str]:
        """Return all model-schema candidates ordered for this conversation."""
        known = set(state.get_known_features())
        declined = set(state.declined_features)
        schema_candidates = [
            feature for feature in MODEL_FEATURE_SCHEMA
            if feature not in known and feature not in declined
            and state.feature_state.get(feature, {}).get("status") != "unavailable"
        ]
        # Grounding has already rejected unsupported hypotheses.  Intersecting
        # rather than trusting the supplied names preserves the hard firewall.
        grounding_rank: Dict[str, int] = {}
        if candidates:
            ordered_candidates = [feature for feature in candidates if feature in MODEL_FEATURE_SCHEMA]
            grounding_rank = {
                feature: len(ordered_candidates) - index
                for index, feature in enumerate(ordered_candidates)
            }
        importance = cls._feature_importances()
        boosted_order = list(cls._INTENT_FEATURES.get((intent or "").lower(), ()))
        symptom_text = " ".join(state.current_symptoms + state.raw_observations).lower()
        for token, related in cls._INTENT_FEATURES.items():
            if token in symptom_text:
                for feature in related:
                    if feature not in boosted_order:
                        boosted_order.append(feature)
        boost_rank = {feature: len(boosted_order) - index for index, feature in enumerate(boosted_order)}
        priority_index = {feature: index for index, feature in enumerate(FEATURE_PRIORITY)}

        # Intent boost is deliberately larger than importance so a symptom-led
        # conversation begins with the relevant existing schema field.
        return sorted(
            schema_candidates,
            key=lambda feature: (
                grounding_rank.get(feature, 0),
                boost_rank.get(feature, 0),
                importance.get(feature, 0.0),
                -priority_index.get(feature, len(FEATURE_PRIORITY)),
                feature,
            ),
            reverse=True,
        )

    @classmethod
    def select_feature_subset(
        cls, state: AgentState, intent: Optional[str] = None, target_size: int = 9,
        candidates: Optional[Iterable[str]] = None,
    ) -> List[str]:
        """Choose up to ``target_size`` best schema features for this conversation."""
        target_size = max(1, min(target_size, len(MODEL_FEATURE_SCHEMA)))
        ranked_unknown = cls.rank_features(state, intent, candidates)
        known = list(state.get_known_features())
        needed = target_size - len(known)
        selected_unknown = ranked_unknown[:needed] if needed > 0 else []
        return list(dict.fromkeys(known + selected_unknown))


feature_ranking_engine = FeatureRankingEngine()
