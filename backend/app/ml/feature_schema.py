"""Dataset/model schema loaded from the shipped model artifact."""

import json
from pathlib import Path

_SCHEMA_PATH = Path(__file__).resolve().parents[3] / "models" / "feature_schema.json"
with _SCHEMA_PATH.open(encoding="utf-8") as _schema_file:
    MODEL_SCHEMA = json.load(_schema_file)

MODEL_FEATURE_SCHEMA = list(MODEL_SCHEMA["selected_features"])
NUMERICAL_FEATURES = list(MODEL_SCHEMA["numerical_features"])
DATE_FEATURES = list(MODEL_SCHEMA.get("date_features", []))
CATEGORICAL_FEATURES = list(MODEL_SCHEMA["categorical_features"])
CATEGORICAL_VALUES = {
    name: [value for value in values if value != "UNKNOWN"]
    for name, values in MODEL_SCHEMA.get("categorical_values", {}).items()
}
FEATURE_PRIORITY = [feature for feature in MODEL_SCHEMA["feature_priority"] if feature in MODEL_FEATURE_SCHEMA]
FEATURE_DISPLAY_NAMES = dict(MODEL_SCHEMA.get("feature_display_names", {}))
MIN_FEATURES_FOR_PREDICTION = int(MODEL_SCHEMA.get("min_features_for_prediction", 2))

def is_valid_feature(name: str) -> bool:
    """Return True if *name* matches a feature in the model schema.
    Matching is case‑insensitive and tolerant of spaces/underscores.
    """
    if not name:
        return False
    cleaned = str(name).strip().replace(" ", "_").lower()
    for feat in MODEL_FEATURE_SCHEMA:
        if feat.lower() == cleaned or feat.lower().replace("_", "") == cleaned.replace("_", ""):
            return True
    return False
