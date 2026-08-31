"""SHAP-based explainability for the SLA-breach model.

Shared by the notebook (global explanations, plots) and the Streamlit
dashboard (per-application "why is this flagged?" explanation) so the two
never diverge - all SHAP logic lives here, nowhere else.

The trained model is a `Pipeline(preprocessor, model)`. SHAP's explainers
work on the underlying estimator over its *transformed* (one-hot encoded,
scaled) input, not the raw pipeline - `transform_features` does that step
and keeps the human-readable feature names alongside it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.features.engineering import CATEGORICAL_FEATURES, FEATURE_COLUMNS


def transform_features(pipeline: Pipeline, X: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    preprocessor = pipeline.named_steps["preprocessor"]
    X_transformed = preprocessor.transform(X[FEATURE_COLUMNS])
    feature_names = list(preprocessor.get_feature_names_out())
    return X_transformed, feature_names


def build_explainer(pipeline: Pipeline, background_data: pd.DataFrame | None = None):
    """Tree models get the fast, exact `TreeExplainer`. Logistic Regression
    (the only linear model among the three trained) gets `LinearExplainer`,
    which needs a background sample to estimate the reference distribution.
    """
    model = pipeline.named_steps["model"]
    if isinstance(model, LogisticRegression):
        if background_data is None:
            raise ValueError("LogisticRegression requires background_data for LinearExplainer.")
        X_background, _ = transform_features(pipeline, background_data)
        return shap.LinearExplainer(model, X_background)
    return shap.TreeExplainer(model)


def compute_shap_values(pipeline: Pipeline, explainer, X: pd.DataFrame) -> shap.Explanation:
    """SHAP values for every row in X, in log-odds space, for the positive
    (breach) class. `base_value + shap_values.sum(axis=1)` reconstructs the
    model's raw margin prediction for each row - ranking/direction (not the
    raw magnitude) is what's used for human-readable explanations below.
    """
    X_transformed, feature_names = transform_features(pipeline, X)
    explanation = explainer(X_transformed)
    explanation.feature_names = feature_names
    return explanation


def global_feature_importance(explanation: shap.Explanation) -> pd.DataFrame:
    """Mean absolute SHAP value per feature, ranked - the standard SHAP
    "global importance" view.
    """
    mean_abs_shap = np.abs(explanation.values).mean(axis=0)
    df = pd.DataFrame({"feature": explanation.feature_names, "mean_abs_shap": mean_abs_shap})
    df["feature_label"] = df["feature"].apply(humanize_feature_name)
    return df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def humanize_feature_name(raw_name: str) -> str:
    """"categorical__customer_segment_Corporate" -> "Customer segment = Corporate"
    "numeric__risk_score" -> "Risk score"
    """
    name = raw_name.split("__", 1)[-1]
    for cat in CATEGORICAL_FEATURES:
        prefix = f"{cat}_"
        if name.startswith(prefix):
            value = name[len(prefix):]
            label = cat.replace("_", " ").capitalize()
            return f"{label} = {value}"
    return name.replace("_", " ").capitalize()


def explain_single_prediction(
    pipeline: Pipeline, explainer, application: pd.DataFrame, top_n: int = 4
) -> dict:
    """Human-readable explanation for one application: predicted
    probability and its top contributing factors, direction included.

    `application` must be a single-row DataFrame with the FEATURE_COLUMNS.
    """
    if len(application) != 1:
        raise ValueError("explain_single_prediction expects exactly one row.")

    probability = float(pipeline.predict_proba(application[FEATURE_COLUMNS])[0, 1])
    explanation = compute_shap_values(pipeline, explainer, application)

    shap_row = explanation.values[0]
    order = np.argsort(-np.abs(shap_row))[:top_n]

    factors = []
    for i in order:
        factors.append(
            {
                "feature": explanation.feature_names[i],
                "label": humanize_feature_name(explanation.feature_names[i]),
                "shap_value": float(shap_row[i]),
                "direction": "increases" if shap_row[i] > 0 else "decreases",
            }
        )

    return {"probability": probability, "top_factors": factors}
