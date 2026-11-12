"""
Emergency Engine — Explicit emergency condition evaluation.

CRITICAL DESIGN PRINCIPLES:
1. The emergency engine is SEPARATE from the ML prediction system.
2. High maintenance probability does NOT automatically mean emergency.
3. The LLM does NOT declare emergencies.
4. The ML model does NOT declare emergencies.
5. Only VALIDATED user-reported conditions matching the explicit policy trigger an emergency.
6. Only conditions supported by the actual ML feature schema are evaluated.

Emergency conditions are defined by an EXPLICIT, DOCUMENTED policy
based on the project's 8 ML features.
"""

from typing import Dict, Any, List, Optional
from backend.app.core.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Explicit Emergency Policy
# =============================================================================
# 
# These are the ONLY emergency conditions. Do NOT invent additional categories.
# All conditions reference features from the final ML model schema:
#   Vehicle_Age, Reported_Issues, Maintenance_History, Service_History,
#   Accident_History, Tire_Condition, Brake_Condition, Battery_Status
#

EMERGENCY_POLICIES = [
    {
        "id": "BRAKE_AND_BATTERY_CRITICAL",
        "severity": "CRITICAL",
        "description": "Worn brakes combined with weak battery",
        "conditions": {
            "Brake_Condition": "Worn Out",
            "Battery_Status": "Weak",
        },
    },
    {
        "id": "BRAKE_HIGH_ISSUES",
        "severity": "HIGH",
        "description": "Worn brakes with multiple reported issues",
        "conditions": {
            "Brake_Condition": "Worn Out",
        },
        "numerical_conditions": {
            "Reported_Issues": {"min": 3},
        },
    },
    {
        "id": "BATTERY_POOR_MAINTENANCE",
        "severity": "HIGH",
        "description": "Weak battery with poor maintenance history",
        "conditions": {
            "Battery_Status": "Weak",
            "Maintenance_History": "Poor",
        },
    },
    {
        "id": "ALL_WEAR_CRITICAL",
        "severity": "CRITICAL",
        "description": "All wear components in worst condition",
        "conditions": {
            "Brake_Condition": "Worn Out",
            "Battery_Status": "Weak",
            "Tire_Condition": "Worn Out",
        },
    },
]

# Severity ranking for comparison
SEVERITY_RANK = {
    "CRITICAL": 3,
    "HIGH": 2,
    "MEDIUM": 1,
    "LOW": 0,
}


class EmergencyEngine:
    """
    Evaluates validated vehicle state against explicit emergency policies.
    
    This engine:
    - Takes ONLY validated feature state from the agent
    - Checks against documented policies
    - Returns structured emergency result
    - Does NOT use ML probability
    - Does NOT accept LLM declarations
    """

    @staticmethod
    def evaluate_emergency(feature_state: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate validated vehicle feature state for emergency conditions.
        
        Args:
            feature_state: Dict mapping feature names to {value, status, source}
                           This comes from AgentState.feature_state (validated data).
        
        Returns:
            {
                "is_emergency": bool,
                "severity": str or None,  
                "matched_policies": [policy IDs],
                "conditions": [human-readable conditions],
                "symptoms": [detected symptoms from features],
            }
        """
        # Extract known feature values
        known_values = {}
        for feat_name, feat_data in feature_state.items():
            if isinstance(feat_data, dict) and feat_data.get("status") == "known":
                known_values[feat_name] = feat_data.get("value")

        matched_policies = []
        all_conditions = []
        all_symptoms = []
        highest_severity = None

        for policy in EMERGENCY_POLICIES:
            # Check categorical conditions
            cat_match = True
            for feat, required_val in policy.get("conditions", {}).items():
                actual_val = known_values.get(feat)
                if actual_val != required_val:
                    cat_match = False
                    break

            if not cat_match:
                continue

            # Check numerical conditions if any
            num_match = True
            for feat, constraint in policy.get("numerical_conditions", {}).items():
                actual_val = known_values.get(feat)
                if actual_val is None:
                    num_match = False
                    break
                try:
                    actual_num = float(actual_val)
                    if "min" in constraint and actual_num < constraint["min"]:
                        num_match = False
                        break
                    if "max" in constraint and actual_num > constraint["max"]:
                        num_match = False
                        break
                except (ValueError, TypeError):
                    num_match = False
                    break

            if not num_match:
                continue

            # Policy matched
            matched_policies.append(policy["id"])
            all_conditions.append(policy["description"])

            # Track highest severity
            sev = policy["severity"]
            if highest_severity is None or SEVERITY_RANK.get(sev, 0) > SEVERITY_RANK.get(highest_severity, 0):
                highest_severity = sev

        # Build symptoms from known dangerous values
        dangerous_values = {
            ("Brake_Condition", "Worn Out"): "Brakes are worn out",
            ("Battery_Status", "Weak"): "Battery is weak",
            ("Tire_Condition", "Worn Out"): "Tires are worn out",
            ("Maintenance_History", "Poor"): "Poor maintenance history",
        }

        for (feat, val), symptom in dangerous_values.items():
            if known_values.get(feat) == val:
                all_symptoms.append(symptom)

        # Reported issues count
        issues = known_values.get("Reported_Issues")
        if issues is not None:
            try:
                if int(issues) >= 3:
                    all_symptoms.append(f"{issues} reported issues")
            except (ValueError, TypeError):
                pass

        is_emergency = len(matched_policies) > 0

        if is_emergency:
            logger.info(
                f"Emergency detected: severity={highest_severity}, "
                f"policies={matched_policies}"
            )
        else:
            logger.debug("No emergency conditions detected")

        return {
            "is_emergency": is_emergency,
            "severity": highest_severity,
            "matched_policies": matched_policies,
            "conditions": list(set(all_conditions)),
            "symptoms": list(set(all_symptoms)),
        }

    @staticmethod
    def get_vehicle_info_from_state(feature_state: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract known vehicle info from feature state for email content.
        Only includes validated, known values. Never invents data.
        """
        info = {}
        for feat_name, feat_data in feature_state.items():
            if isinstance(feat_data, dict) and feat_data.get("status") == "known":
                info[feat_name] = feat_data.get("value")
        return info
