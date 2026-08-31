"""Load the trained model and score new applications.

Used by the Streamlit dashboard's Delay Prediction page and by the SHAP
explainability notebook - kept separate from train.py so scoring a new
application doesn't require scikit-learn's training-time dependencies to be
re-run, just a fitted pipeline loaded from disk.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.features.engineering import FEATURE_COLUMNS


def load_model(path: str | Path = "models/delay_prediction_model.joblib"):
    return joblib.load(path)


def predict_breach_probability(model, applications: pd.DataFrame) -> pd.Series:
    """Probability of SLA breach for each row. `applications` must contain
    every column in FEATURE_COLUMNS (extra columns are ignored).
    """
    missing = set(FEATURE_COLUMNS) - set(applications.columns)
    if missing:
        raise ValueError(f"Missing required feature column(s): {sorted(missing)}")
    proba = model.predict_proba(applications[FEATURE_COLUMNS])[:, 1]
    return pd.Series(proba, index=applications.index, name="sla_breach_probability")


def risk_category(probability: float) -> str:
    if probability >= 0.6:
        return "HIGH RISK"
    if probability >= 0.3:
        return "MODERATE RISK"
    return "LOW RISK"
