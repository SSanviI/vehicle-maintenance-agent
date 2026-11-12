"""Time per-request predict() and explain() calls."""
import time
t0 = time.time()

from backend.app.services.prediction_service import get_predictor
from backend.app.ml.preprocessing import SELECTED_FEATURES

p = get_predictor()
print(f"[{time.time()-t0:.2f}s] predictor loaded")

feature_state = {
    "Brake_Condition": {"value": "Worn Out", "status": "known", "source": "user"},
    "Battery_Status":  {"value": "Weak",     "status": "known", "source": "user"},
}
for feat in SELECTED_FEATURES:
    if feat not in feature_state:
        feature_state[feat] = {"value": None, "status": "unknown", "source": "user"}

print(f"[{time.time()-t0:.2f}s] running predict()...")
pred = p.predict(feature_state)
print(f"[{time.time()-t0:.2f}s] predict done: risk={pred.get('risk_level')}")

print(f"[{time.time()-t0:.2f}s] running explain()...")
expl = p.explainer.explain_prediction(feature_state, pred)
print(f"[{time.time()-t0:.2f}s] explain done: available={expl.get('available')}")

# Time a second call (warm)
print(f"[{time.time()-t0:.2f}s] WARM: running explain() again...")
expl2 = p.explainer.explain_prediction(feature_state, pred)
print(f"[{time.time()-t0:.2f}s] WARM explain done: available={expl2.get('available')}")
