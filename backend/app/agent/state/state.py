"""
Agent State — Single authoritative state management for vehicle assessment conversations.
Tracks validated feature values, statuses, current question context,
prediction history, and checkpoints for turn-level recovery.
"""

from __future__ import annotations

import json
from typing import Dict, Any, Optional, List, Set
from copy import deepcopy

from backend.app.ml.preprocessing import (
    SELECTED_FEATURES,
    FEATURE_DISPLAY_NAMES,
    FEATURE_PRIORITY,
    CATEGORICAL_VALUES,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
)
from backend.app.ml.feature_schema import MIN_FEATURES_FOR_PREDICTION

# ---------------------------------------------------------------------------
# Checkpoint constants — progress markers for each agent turn
# ---------------------------------------------------------------------------
CHECKPOINT_INPUT_RECEIVED = "input_received"
CHECKPOINT_FEATURES_EXTRACTED = "features_extracted"
CHECKPOINT_FEATURES_VALIDATED = "features_validated"
CHECKPOINT_INTENT_EXTRACTED = "intent_extracted"
CHECKPOINT_STATE_UPDATED = "state_updated"
CHECKPOINT_FEATURES_RANKED = "features_ranked"
CHECKPOINT_ASSESSMENT_FEATURES_SELECTED = "assessment_features_selected"
CHECKPOINT_QUESTION_SELECTED = "question_selected"
CHECKPOINT_PREDICTION_GENERATED = "prediction_generated"
CHECKPOINT_RESPONSE_GENERATED = "response_generated"
CHECKPOINT_JUDGE_PASSED = "judge_passed"



class AgentState:
    """
    Authoritative state for a vehicle maintenance assessment conversation.

    Attributes:
        conversation_id: Unique conversation identifier
        feature_state: Dict mapping feature name -> {"value": Any, "status": str, "source": str}
        asked_features: Set of feature names that have been asked to the user
        declined_features: Set of feature names the user explicitly declined / didn't know
        current_question_feature: The active feature asked in the previous turn
        predictions: List of historical and current prediction objects
        compacted_summary: Rolling conversation summary for context window preservation
        turn_count: Number of completed turns
        checkpoint_log: Ordered list of checkpoints reached during the current turn
        raw_observations: List of user-mentioned raw observations / symptoms (not validated features)
    """

    def __init__(self, conversation_id: Optional[str] = None):
        self.conversation_id: Optional[str] = conversation_id
        self.feature_state: Dict[str, Dict[str, Any]] = {}
        self.asked_features: Set[str] = set()
        self.declined_features: Set[str] = set()
        self.current_question_feature: Optional[str] = None
        self.predictions: List[Dict[str, Any]] = []
        self.explanations: List[Dict[str, Any]] = []
        self.compacted_summary: Optional[str] = None
        self.turn_count: int = 0
        self.checkpoint_log: List[str] = []
        self.raw_observations: List[str] = []
        self.symptoms: Dict[str, str] = {}
        self.raw_user_inputs: Dict[str, Any] = {}
        # A conversation-specific, schema-bounded question plan.  It contains
        # names only from SELECTED_FEATURES and is never supplied by an LLM.
        self.selected_features: List[str] = []
        self.assessment_plan_initialized: bool = False
        # Prediction availability and assessment completion are deliberately
        # independent: a partial RF score never ends the interview.
        self.assessment_complete: bool = False
        self.assessment_completion_reason: Optional[str] = None
        self.remaining_relevant_features: List[str] = []
        self.latest_prediction_state_signature: Optional[str] = None
        self.latest_explanation_state_signature: Optional[str] = None
        self.current_intent: Optional[str] = None
        self.current_symptoms: List[str] = []
        self.current_hypotheses: List[Dict[str, Any]] = []
        self.grounded_hypotheses: List[Dict[str, Any]] = []

        # Assessment lifecycle mode: "ASSESSMENT", "AWAITING_SUMMARY_CONFIRMATION", "SUMMARY", "FOLLOW_UP"
        self.assessment_mode: str = "ASSESSMENT"

        # Initialize every trained-model feature as unknown.  The schema is the
        # hard boundary; conversational observations never create new entries.
        for feat in SELECTED_FEATURES:
            self.feature_state[feat] = {
                "value": None,
                "status": "unknown",
                "source": "user",
            }

    # ---------------------------------------------------------------------------
    # Checkpoint Management
    # ---------------------------------------------------------------------------

    def record_checkpoint(self, checkpoint: str):
        """Record a completed checkpoint for this turn."""
        if checkpoint not in self.checkpoint_log:
            self.checkpoint_log.append(checkpoint)

    def has_checkpoint(self, checkpoint: str) -> bool:
        """Check if a checkpoint has been reached."""
        return checkpoint in self.checkpoint_log

    def reset_turn_checkpoints(self):
        """Reset checkpoint log at the start of each new turn."""
        self.checkpoint_log = []

    # ---------------------------------------------------------------------------
    # Feature State Management
    # ---------------------------------------------------------------------------

    def update_feature(
        self,
        name: str,
        value: Any,
        status: str = "known",
        source: str = "user",
    ) -> bool:
        """
        Update a feature value in the state.
        Returns True if the value actually changed.
        Only valid SELECTED_FEATURES are permitted.
        """
        if name not in self.feature_state:
            return False

        old = self.feature_state[name].copy()
        self.feature_state[name] = {
            "value": value,
            "status": status,
            "source": source,
        }

        changed = (old.get("value") != value or old.get("status") != status)
        if changed:
            if status == "known":
                self.declined_features.discard(name)
                # If this was the current question feature, clear it
                if self.current_question_feature == name:
                    self.current_question_feature = None

        return changed

    def mark_declined(self, feature_name: str):
        """
        Mark a feature as declined / unavailable ("I don't know").
        Guarantees this feature will NEVER be asked again.
        """
        if feature_name in self.feature_state:
            self.feature_state[feature_name]["status"] = "unavailable"
            self.feature_state[feature_name]["value"] = None
            self.declined_features.add(feature_name)
            self.asked_features.add(feature_name)
            if self.current_question_feature == feature_name:
                self.current_question_feature = None

    def mark_asked(self, feature_name: str):
        """
        Record that a feature question was presented to the user.
        Sets current_question_feature to track conversational context.
        """
        if feature_name in self.feature_state:
            self.asked_features.add(feature_name)
            self.current_question_feature = feature_name

    def clear_current_question(self):
        """Clear the current active question context."""
        self.current_question_feature = None

    def begin_new_assessment(self):
        """Reset only assessment facts when the user explicitly starts over.

        The conversation, rolling-context infrastructure, and agent lifecycle
        remain intact.  This simply prevents facts and predictions from a
        completed vehicle assessment being reused for a new one.
        """
        for feature in self.feature_state:
            self.feature_state[feature] = {
                "value": None,
                "status": "unknown",
                "source": "user",
            }

        self.asked_features.clear()
        self.declined_features.clear()
        self.current_question_feature = None
        self.predictions = []
        self.explanations = []
        self.raw_observations = []
        self.symptoms = {}
        self.raw_user_inputs = {}
        self.selected_features = []
        self.assessment_plan_initialized = False
        self.assessment_complete = False
        self.assessment_completion_reason = None
        self.remaining_relevant_features = []
        self.latest_prediction_state_signature = None
        self.latest_explanation_state_signature = None
        self.current_intent = None
        self.current_symptoms = []
        self.current_hypotheses = []
        self.grounded_hypotheses = []
        self.assessment_mode = "ASSESSMENT"

    def set_feature_subset(self, features: List[str], intent: Optional[str] = None):
        """Persist the ranked, dataset-bounded feature subset for this turn."""
        known = list(self.get_known_features())
        valid = [feature for feature in features if feature in self.feature_state]
        # Preserve confirmed values in the plan even when a later intent shifts
        # ranking, then keep the deterministic ranking's order for the rest.
        self.selected_features = list(dict.fromkeys(known + valid))
        self.assessment_plan_initialized = True
        self.assessment_complete = False
        self.assessment_completion_reason = None
        self.current_intent = intent

    def get_known_features(self) -> Dict[str, Any]:
        """Return a dict of all known features: {name: value}."""
        return {
            name: state["value"]
            for name, state in self.feature_state.items()
            if state.get("status") == "known" and state.get("value") is not None
        }

    def get_unknown_features(self) -> List[str]:
        """Return list of features that are still unknown (not known and not declined)."""
        return [
            name for name, state in self.feature_state.items()
            if state.get("status") == "unknown" and name not in self.declined_features
        ]

    def get_askable_features(self) -> List[str]:
        """Return list of features that can be asked in priority order."""
        from backend.app.agent.state.feature_selector import get_askable_features
        return get_askable_features(self)

    def get_next_question_feature(self) -> Optional[str]:
        """Return the next deterministic feature to ask about."""
        from backend.app.agent.state.feature_selector import get_next_question_feature
        return get_next_question_feature(self)

    def known_feature_count(self) -> int:
        """Return the count of currently confirmed known features."""
        return len(self.get_known_features())

    def has_enough_for_prediction(self) -> bool:
        """
        Check if state satisfies the hard programmatic gate for ML prediction (>= 2 features).
        """
        return self.known_feature_count() >= MIN_FEATURES_FOR_PREDICTION

    # ---------------------------------------------------------------------------
    # Prediction & Explanation History Management
    # ---------------------------------------------------------------------------

    def add_prediction(self, prediction: Dict[str, Any]):
        """Record a newly generated ML prediction."""
        self.predictions.append(prediction)
        self.latest_prediction_state_signature = self.feature_state_signature()

    def get_latest_prediction(self) -> Optional[Dict[str, Any]]:
        """Get the most recent prediction object, or None if no prediction exists."""
        return self.predictions[-1] if self.predictions else None

    def add_explanation(self, explanation: Dict[str, Any]):
        """Record a newly calculated SHAP local explanation."""
        self.explanations.append(explanation)
        self.latest_explanation_state_signature = self.feature_state_signature()

    def get_latest_explanation(self) -> Optional[Dict[str, Any]]:
        """Get the most recent SHAP local explanation object, or None if none exists."""
        return self.explanations[-1] if self.explanations else None

    def feature_state_signature(self) -> str:
        """Stable signature used to prove a prediction/explanation matches current facts."""
        return json.dumps(self.get_known_features(), sort_keys=True, default=str)

    def has_current_prediction(self) -> bool:
        return bool(self.get_latest_prediction()) and (
            self.latest_prediction_state_signature == self.feature_state_signature()
        )

    def has_current_explanation(self) -> bool:
        return bool(self.get_latest_explanation()) and (
            self.latest_explanation_state_signature == self.feature_state_signature()
        )

    # ---------------------------------------------------------------------------
    # Serialization
    # ---------------------------------------------------------------------------

    def to_context_dict(self) -> Dict[str, Any]:
        """Export state summary for LLM context injection."""
        known = self.get_known_features()
        unknown = self.get_unknown_features()
        declined = list(self.declined_features)
        latest = self.get_latest_prediction()
        latest_expl = self.get_latest_explanation()

        return {
            "known_features": known,
            "assessed_features": known,
            "unknown_features": unknown,
            "declined_features": declined,
            "current_question_feature": self.current_question_feature,
            # Explicit public continuation contract for API/SSE consumers.
            # It is a display-safe alias of the selected schema feature.
            "next_question": self.current_question_feature,
            "raw_observations": self.raw_observations,
            "total_known": len(known),
            "total_features": len(SELECTED_FEATURES),
            "min_features_for_prediction": MIN_FEATURES_FOR_PREDICTION,
            "ready_for_prediction": self.has_enough_for_prediction(),
            "latest_prediction": {
                "prediction": latest.get("prediction"),
                "model_score": latest.get("model_score"),
                "risk_level": latest.get("risk_level"),
                "prediction_id": latest.get("prediction_id"),
            } if latest else None,
            "latest_explanation": latest_expl if self.has_current_explanation() else None,
            "turn_count": self.turn_count,
            "checkpoints": list(self.checkpoint_log),
            "selected_features": list(self.selected_features),
            "assessment_plan_initialized": self.assessment_plan_initialized,
            "assessment_complete": self.assessment_complete,
            "assessment_completion_reason": self.assessment_completion_reason,
            "assessment_mode": self.assessment_mode,
            "remaining_relevant_features": list(self.remaining_relevant_features),
            "open_questions": list(self.remaining_relevant_features),
            "latest_prediction_is_current": self.has_current_prediction(),
            "latest_explanation_is_current": self.has_current_explanation(),
            "current_intent": self.current_intent,
            "current_symptoms": list(self.current_symptoms),
            "current_hypotheses": deepcopy(self.current_hypotheses),
            "grounded_hypotheses": deepcopy(self.grounded_hypotheses),
        }

    def to_json(self) -> Dict[str, Any]:
        """Serialize complete state for database persistence."""
        return {
            "feature_state": deepcopy(self.feature_state),
            "asked_features": list(self.asked_features),
            "declined_features": list(self.declined_features),
            "current_question_feature": self.current_question_feature,
            "predictions": deepcopy(self.predictions),
            "explanations": deepcopy(self.explanations),
            "turn_count": self.turn_count,
            "compacted_summary": self.compacted_summary,
            "checkpoint_log": list(self.checkpoint_log),
            "raw_observations": list(self.raw_observations),
            "symptoms": deepcopy(self.symptoms),
            "raw_user_inputs": deepcopy(self.raw_user_inputs),
            "selected_features": list(self.selected_features),
            "assessment_plan_initialized": self.assessment_plan_initialized,
            "assessment_complete": self.assessment_complete,
            "assessment_completion_reason": self.assessment_completion_reason,
            "assessment_mode": self.assessment_mode,
            "remaining_relevant_features": list(self.remaining_relevant_features),
            "latest_prediction_state_signature": self.latest_prediction_state_signature,
            "latest_explanation_state_signature": self.latest_explanation_state_signature,
            "current_intent": self.current_intent,
            "current_symptoms": list(self.current_symptoms),
            "current_hypotheses": deepcopy(self.current_hypotheses),
            "grounded_hypotheses": deepcopy(self.grounded_hypotheses),
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any], conversation_id: Optional[str] = None) -> 'AgentState':
        """Restore state from database dictionary."""
        state = cls(conversation_id=conversation_id)
        if data:
            incoming_features = data.get("feature_state")
            if incoming_features and isinstance(incoming_features, dict):
                for feat, feat_dict in incoming_features.items():
                    if feat in state.feature_state and isinstance(feat_dict, dict):
                        state.feature_state[feat].update(feat_dict)

            state.asked_features = set(data.get("asked_features", []))
            state.declined_features = set(data.get("declined_features", []))
            state.current_question_feature = data.get("current_question_feature")
            state.predictions = data.get("predictions", [])
            state.explanations = data.get("explanations", [])
            state.turn_count = data.get("turn_count", 0)
            state.compacted_summary = data.get("compacted_summary")
            state.checkpoint_log = data.get("checkpoint_log", [])
            state.raw_observations = data.get("raw_observations", [])
            state.symptoms = data.get("symptoms", {})
            state.raw_user_inputs = data.get("raw_user_inputs", {})
            state.selected_features = [
                feature for feature in data.get("selected_features", [])
                if feature in state.feature_state
            ]
            state.assessment_plan_initialized = bool(data.get("assessment_plan_initialized", False))
            state.assessment_complete = bool(data.get("assessment_complete", False))
            state.assessment_completion_reason = data.get("assessment_completion_reason")
            state.assessment_mode = data.get("assessment_mode", "AWAITING_SUMMARY_CONFIRMATION" if state.assessment_complete else "ASSESSMENT")
            state.remaining_relevant_features = [
                feature for feature in data.get("remaining_relevant_features", [])
                if feature in state.feature_state
            ]
            state.latest_prediction_state_signature = data.get("latest_prediction_state_signature")
            state.latest_explanation_state_signature = data.get("latest_explanation_state_signature")
            state.current_intent = data.get("current_intent")
            state.current_symptoms = data.get("current_symptoms", [])
            state.current_hypotheses = data.get("current_hypotheses", [])
            state.grounded_hypotheses = data.get("grounded_hypotheses", [])
        return state

