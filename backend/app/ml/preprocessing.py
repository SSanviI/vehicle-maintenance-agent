"""
ML Preprocessing module for the backend.
Mirrors the training pipeline's MaintenancePreprocessor.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Any, Optional
from backend.app.core.logging import get_logger

logger = get_logger(__name__)

# Feature configuration (mirrored from training) – now sourced from central schema
from backend.app.ml.feature_schema import (
    MODEL_FEATURE_SCHEMA as SELECTED_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    DATE_FEATURES,
    CATEGORICAL_VALUES,
    FEATURE_PRIORITY,
    FEATURE_DISPLAY_NAMES,
)


class MaintenancePreprocessor:
    """Preprocessing pipeline that handles partial inputs."""

    def __init__(self):
        self.category_maps = {}
        self.numerical_medians = {}
        self.is_fitted = False

    def fit_from_config(self):
        """Initialize from known configuration without needing training data."""
        for col in CATEGORICAL_FEATURES:
            values = CATEGORICAL_VALUES[col]
            self.category_maps[col] = {v: i for i, v in enumerate(values)}
        self.is_fitted = True

    def transform_feature_state(self, feature_state: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Transform a partial feature state dict into model-ready input.
        """
        row = {}
        for feat in SELECTED_FEATURES:
            if feat in feature_state and feature_state[feat].get("status") == "known":
                row[feat] = feature_state[feat]["value"]
            else:
                row[feat] = np.nan

        df = pd.DataFrame([row])
        return self._transform(df)

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform dataframe with missing value handling."""
        result = pd.DataFrame(index=df.index)

        for col in CATEGORICAL_FEATURES:
            values = df[col].copy() if col in df.columns else pd.Series(np.nan, index=df.index)
            is_missing = values.isna() | (values == "UNKNOWN") | (values == "")
            mapped = values.map(self.category_maps[col]).fillna(-1)
            result[col] = mapped.astype(float)
            result[f"{col}_missing"] = is_missing.astype(int)

        for col in DATE_FEATURES:
            values = df[col].copy() if col in df.columns else pd.Series(np.nan, index=df.index)
            parsed = pd.to_datetime(values, errors="coerce")
            is_missing = parsed.isna()
            result[col] = parsed.map(lambda value: value.toordinal() if not pd.isna(value) else -1).astype(float)
            result[f"{col}_missing"] = is_missing.astype(int)

        for col in NUMERICAL_FEATURES:
            values = df[col].copy() if col in df.columns else pd.Series(np.nan, index=df.index)
            is_missing = values.isna() | (values == -1)
            result[col] = values.fillna(-1).astype(float)
            result[f"{col}_missing"] = is_missing.astype(int)

        return result

    @classmethod
    def load(cls, path: str) -> 'MaintenancePreprocessor':
        """Load preprocessor from saved pipeline."""
        data = joblib.load(path)
        pp = cls()
        pp.category_maps = data['category_maps']
        pp.numerical_medians = data.get('numerical_medians', {})
        pp.is_fitted = True
        return pp

    @classmethod
    def from_config(cls) -> 'MaintenancePreprocessor':
        """Create preprocessor from hardcoded config (no saved file needed)."""
        pp = cls()
        pp.fit_from_config()
        return pp
