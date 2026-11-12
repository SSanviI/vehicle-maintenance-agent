import sys
from pathlib import Path

root = Path(__file__).parent.parent
sys.path.insert(0, str(root))

from backend.app.services.prediction_service import get_predictor

def main():
    print("Testing ML Predictor Engine...")
    predictor = get_predictor()
    print(f"ML Predictor loaded: {predictor.is_loaded}")
    if not predictor.is_loaded:
        print("FAIL: ML Predictor failed to load model!")
        return

    # Partial input test
    state = {
        "Vehicle_Age": {"value": 5, "status": "known", "source": "user"},
        "Reported_Issues": {"value": 3, "status": "known", "source": "user"},
    }
    result = predictor.predict(state)
    print(f"Prediction result: class={result['prediction_class']}, score={result['model_score']}, risk={result['risk_level']}")
    assert result["prediction"] in ["maintenance_required", "no_maintenance_needed"]
    print("PASS: ML Predictor Engine is loaded and working correctly!")

if __name__ == "__main__":
    main()
