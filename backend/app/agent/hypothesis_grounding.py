"""Dataset-grounding firewall for conversational maintenance hypotheses.

Hypotheses are useful conversational signals, never model inputs.  This module
is the only bridge from a hypothesis to a questionable model feature and it
always emits names from ``MODEL_FEATURE_SCHEMA``.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List

from backend.app.ml.feature_schema import MODEL_FEATURE_SCHEMA


# Controlled semantic metadata.  The keys are language users and an Input LLM
# commonly use; values are *existing* dataset columns, never generated names.
_SEMANTIC_MAPPINGS = {
    "brake": ("Brake_Condition", "Reported_Issues", "Maintenance_History", "Service_History", "Last_Service_Date"),
    "braking": ("Brake_Condition", "Reported_Issues", "Maintenance_History", "Service_History", "Last_Service_Date"),
    "grinding": ("Brake_Condition", "Reported_Issues", "Maintenance_History", "Service_History", "Tire_Condition"),
    "battery": ("Battery_Status", "Reported_Issues", "Maintenance_History", "Last_Service_Date", "Vehicle_Age"),
    "electrical": ("Battery_Status", "Reported_Issues", "Maintenance_History", "Last_Service_Date"),
    "won't start": ("Battery_Status", "Reported_Issues", "Maintenance_History", "Last_Service_Date"),
    "wont start": ("Battery_Status", "Reported_Issues", "Maintenance_History", "Last_Service_Date"),
    "tire": ("Tire_Condition", "Reported_Issues", "Maintenance_History", "Service_History", "Odometer_Reading"),
    "tyre": ("Tire_Condition", "Reported_Issues", "Maintenance_History", "Service_History", "Odometer_Reading"),
    "service": ("Maintenance_History", "Service_History", "Last_Service_Date"),
    "maintenance": ("Maintenance_History", "Service_History", "Last_Service_Date"),
    "problem": ("Maintenance_History", "Service_History", "Brake_Condition", "Battery_Status", "Accident_History"),
    "issue": ("Maintenance_History", "Service_History", "Brake_Condition", "Battery_Status", "Accident_History"),
    "accident": ("Accident_History", "Reported_Issues"),
    "crash": ("Accident_History", "Reported_Issues"),
}


def derive_hypotheses(text: str) -> List[Dict[str, Any]]:
    """Provide a safe fallback when the Input LLM is unavailable.

    These are deliberately broad possible causes, not assertions or values.
    """
    lower = (text or "").lower()
    hypotheses: List[Dict[str, Any]] = []
    for phrase, label in (
        ("brake", "brake-related condition issue"),
        ("braking", "brake-related condition issue"),
        ("grinding", "brake-related wear or moving-part issue"),
        ("battery", "battery-related issue"),
        ("dying", "battery-related issue"),
        ("tire", "tire-related condition issue"),
        ("tyre", "tire-related condition issue"),
        ("wheel alignment", "wheel alignment issue"),
        ("chain", "chain or moving-part issue"),
        ("problems", "maintenance-related issues"),
        ("problem", "maintenance-related issues"),
        ("issues", "maintenance-related issues"),
        ("issue", "maintenance-related issues"),
        ("maintenance", "maintenance or service history issue"),
        ("service", "maintenance or service history issue"),
    ):
        if phrase in lower and not any(item["hypothesis"] == label for item in hypotheses):
            hypotheses.append({"hypothesis": label, "confidence": 0.6})
    if any(token in lower for token in ("noise", "sound", "rattle", "squeal")):
        hypotheses.append({"hypothesis": "brake or tire-related condition issue", "confidence": 0.4})
    return hypotheses


def normalise_hypotheses(candidates: Any, fallback_text: str) -> List[Dict[str, Any]]:
    """Keep LLM hypotheses as untrusted text, with a deterministic fallback."""
    result: List[Dict[str, Any]] = []
    if isinstance(candidates, list):
        for item in candidates:
            text = item.get("hypothesis") if isinstance(item, dict) else item
            if not isinstance(text, str) or not text.strip():
                continue
            confidence = item.get("confidence", 0.5) if isinstance(item, dict) else 0.5
            try:
                confidence = max(0.0, min(1.0, float(confidence)))
            except (TypeError, ValueError):
                confidence = 0.5
            clean = text.strip()[:240]
            if clean.lower() not in {entry["hypothesis"].lower() for entry in result}:
                result.append({"hypothesis": clean, "confidence": confidence})
    return result or derive_hypotheses(fallback_text)


def ground_hypotheses(hypotheses: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Map hypotheses to actual model columns, rejecting unsupported concepts.

    A result with ``status='unsupported'`` is retained for auditability but
    cannot be selected or asked because ``dataset_feature`` is ``None``.
    """
    grounded: List[Dict[str, Any]] = []
    for item in hypotheses:
        hypothesis = str(item.get("hypothesis", "")).strip()
        lower = hypothesis.lower()
        matches: List[str] = []
        # Unsupported concepts must not leak through vague words such as
        # "wheel" or "chain".  There is no Wheel_Alignment/Chain_Condition.
        if re.search(r"\b(wheel\s+alignment|alignment|chain)\b", lower):
            phrases = ()
        else:
            phrases = (phrase for phrase in _SEMANTIC_MAPPINGS if phrase in lower)
        for phrase in phrases:
            for feature in _SEMANTIC_MAPPINGS[phrase]:
                if feature in MODEL_FEATURE_SCHEMA and feature not in matches:
                    matches.append(feature)
        if not matches:
            grounded.append({
                "hypothesis": hypothesis,
                "dataset_feature": None,
                "status": "unsupported",
                "relevance": "none",
                "confidence": item.get("confidence", 0.5),
            })
            continue
        for index, feature in enumerate(matches):
            grounded.append({
                "hypothesis": hypothesis,
                "dataset_feature": feature,
                "status": "grounded",
                "relevance": "high" if index == 0 else "medium",
                "confidence": item.get("confidence", 0.5),
            })
    return grounded


def grounded_features(grounded: Iterable[Dict[str, Any]]) -> List[str]:
    """Return unique, schema-valid features in grounding relevance order."""
    return list(dict.fromkeys(
        item["dataset_feature"] for item in grounded
        if item.get("status") == "grounded" and item.get("dataset_feature") in MODEL_FEATURE_SCHEMA
    ))
