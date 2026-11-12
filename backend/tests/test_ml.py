"""
Automated tests for ML model, preprocessing, and partial input handling.
"""


import numpy as np
from backend.app.services.prediction_service import get_predictor
from backend.app.ml.preprocessing import MaintenancePreprocessor


def test_predictor_loaded():
    """Verify ML predictor is loaded and functional."""
    predictor = get_predictor()
    assert predictor.is_loaded is True
    assert predictor.model is not None


def test_partial_input_prediction():
    """Verify prediction works with partial feature states."""
    predictor = get_predictor()

    # Case 1: Partial state (Vehicle_Age & Reported_Issues known)
    state_partial = {
        "Vehicle_Age": {"value": 5, "status": "known", "source": "user"},
        "Reported_Issues": {"value": 3, "status": "known", "source": "user"},
        "Brake_Condition": {"value": None, "status": "unknown", "source": "user"},
    }
    res1 = predictor.predict(state_partial)
    assert res1.get("prediction") in ["maintenance_required", "no_maintenance_needed"]
    assert "Vehicle_Age" in res1["features_used"]
    assert "Reported_Issues" in res1["features_used"]
    assert "Brake_Condition" in res1["missing_features"]

    # Case 2: Worn out brakes (high risk domain signal)
    state_worn_brakes = {
        "Vehicle_Age": {"value": 5, "status": "known", "source": "user"},
        "Reported_Issues": {"value": 3, "status": "known", "source": "user"},
        "Brake_Condition": {"value": "Worn Out", "status": "known", "source": "user"},
    }
    res2 = predictor.predict(state_worn_brakes)
    assert res2["prediction"] == "maintenance_required"
    assert res2["model_score"] > 0.85
    assert res2["risk_level"] in ["HIGH", "CRITICAL"]


def test_preprocessor_from_config():
    """Test standalone preprocessor initialization."""
    pp = MaintenancePreprocessor.from_config()
    assert pp.is_fitted is True
    assert "Maintenance_History" in pp.category_maps
