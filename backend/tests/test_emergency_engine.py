"""
Unit tests for the EmergencyEngine: policy evaluation, emergency separation from ML probability, and anti-hallucination.
"""

import pytest
from backend.app.emergency.emergency_engine import EmergencyEngine


def test_emergency_engine_healthy_vehicle():
    """Verify normal/healthy vehicle state produces NO emergency."""
    feature_state = {
        "Vehicle_Age": {"value": 3, "status": "known"},
        "Brake_Condition": {"value": "Good", "status": "known"},
        "Battery_Status": {"value": "Good", "status": "known"},
        "Tire_Condition": {"value": "Good", "status": "known"},
        "Reported_Issues": {"value": 0, "status": "known"},
    }

    result = EmergencyEngine.evaluate_emergency(feature_state)
    assert result["is_emergency"] is False
    assert result["severity"] is None
    assert len(result["matched_policies"]) == 0


def test_emergency_engine_critical_brakes_and_battery():
    """Verify explicit condition: Worn Brakes + Weak Battery -> CRITICAL Emergency."""
    feature_state = {
        "Brake_Condition": {"value": "Worn Out", "status": "known"},
        "Battery_Status": {"value": "Weak", "status": "known"},
    }

    result = EmergencyEngine.evaluate_emergency(feature_state)
    assert result["is_emergency"] is True
    assert result["severity"] == "CRITICAL"
    assert "BRAKE_AND_BATTERY_CRITICAL" in result["matched_policies"]
    assert len(result["conditions"]) > 0


def test_emergency_engine_high_probability_does_not_force_emergency():
    """
    CRITICAL SECURITY REQUIREMENT:
    High maintenance probability alone MUST NOT automatically trigger an emergency.
    If vehicle features do NOT match explicit emergency policy, EmergencyEngine returns is_emergency=False.
    """
    # High age (15) and average history, but no worn brakes or weak battery
    feature_state = {
        "Vehicle_Age": {"value": 15, "status": "known"},
        "Maintenance_History": {"value": "Average", "status": "known"},
        "Brake_Condition": {"value": "Good", "status": "known"},
        "Battery_Status": {"value": "Good", "status": "known"},
    }

    result = EmergencyEngine.evaluate_emergency(feature_state)
    assert result["is_emergency"] is False


def test_emergency_engine_ignores_unvalidated_unknown_features():
    """Unknown/unvalidated features must be ignored by emergency engine."""
    feature_state = {
        "Brake_Condition": {"value": "Worn Out", "status": "unknown"},  # Status is unknown!
        "Battery_Status": {"value": "Weak", "status": "unknown"},
    }

    result = EmergencyEngine.evaluate_emergency(feature_state)
    assert result["is_emergency"] is False
