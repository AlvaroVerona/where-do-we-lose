"""Tests for SHAP-based model explainability."""

import numpy as np
import pandas as pd
import pytest

from src.data.generate_data import generate_dataset
from src.features.engineering import build_feature_frame, time_based_train_val_test_split
from src.models.explain import (
    build_explainer,
    compute_shap_values,
    explain_single_prediction,
    global_feature_importance,
    humanize_feature_name,
    transform_features,
)
from src.models.train import train_models


@pytest.fixture(scope="module")
def split_data():
    process_df, _ = generate_dataset(n_processes=3000, seed=31, start_date="2024-01-01", end_date="2024-12-31")
    features = build_feature_frame(process_df)
    return time_based_train_val_test_split(features)


@pytest.fixture(scope="module")
def gb_model(split_data):
    train_df, _, _ = split_data
    return train_models(train_df, train_df["sla_breached"])["gradient_boosting"]


@pytest.fixture(scope="module")
def lr_model(split_data):
    train_df, _, _ = split_data
    return train_models(train_df, train_df["sla_breached"])["logistic_regression"]


def test_humanize_feature_name_categorical():
    assert humanize_feature_name("categorical__customer_segment_Corporate") == "Customer segment = Corporate"
    assert humanize_feature_name("categorical__priority_High") == "Priority = High"


def test_humanize_feature_name_numeric():
    assert humanize_feature_name("numeric__risk_score") == "Risk score"
    assert humanize_feature_name("numeric__current_queue_size") == "Current queue size"


def test_transform_features_shape_matches_onehot_expansion(split_data, gb_model):
    _, val_df, _ = split_data
    X_transformed, feature_names = transform_features(gb_model, val_df)
    assert X_transformed.shape[0] == len(val_df)
    assert X_transformed.shape[1] == len(feature_names)


def test_tree_explainer_shap_values_reconstruct_margin(split_data, gb_model):
    _, val_df, _ = split_data
    sample = val_df.sample(100, random_state=1)
    explainer = build_explainer(gb_model)
    explanation = compute_shap_values(gb_model, explainer, sample)

    reconstructed_margin = explanation.base_values + explanation.values.sum(axis=1)
    reconstructed_proba = 1 / (1 + np.exp(-reconstructed_margin))
    actual_proba = gb_model.named_steps["model"].predict_proba(
        gb_model.named_steps["preprocessor"].transform(sample)
    )[:, 1]
    np.testing.assert_allclose(reconstructed_proba, actual_proba, atol=1e-6)


def test_linear_explainer_works_for_logistic_regression(split_data, lr_model):
    train_df, val_df, _ = split_data
    explainer = build_explainer(lr_model, background_data=train_df.sample(200, random_state=1))
    explanation = compute_shap_values(lr_model, explainer, val_df.sample(20, random_state=1))
    assert explanation.values.shape[0] == 20


def test_global_feature_importance_sorted_and_nonnegative(split_data, gb_model):
    _, val_df, _ = split_data
    explainer = build_explainer(gb_model)
    explanation = compute_shap_values(gb_model, explainer, val_df.sample(200, random_state=1))
    importance = global_feature_importance(explanation)
    assert (importance["mean_abs_shap"] >= 0).all()
    assert importance["mean_abs_shap"].is_monotonic_decreasing
    assert "feature_label" in importance.columns


def test_explain_single_prediction_matches_model_probability(split_data, gb_model):
    _, val_df, _ = split_data
    explainer = build_explainer(gb_model)
    row = val_df.sample(1, random_state=3)
    result = explain_single_prediction(gb_model, explainer, row, top_n=4)

    from src.features.engineering import FEATURE_COLUMNS
    expected_proba = gb_model.predict_proba(row[FEATURE_COLUMNS])[0, 1]

    assert result["probability"] == pytest.approx(expected_proba, abs=1e-6)
    assert len(result["top_factors"]) == 4
    for factor in result["top_factors"]:
        assert factor["direction"] in ("increases", "decreases")
        assert isinstance(factor["label"], str) and len(factor["label"]) > 0


def test_explain_single_prediction_rejects_multi_row(split_data, gb_model):
    _, val_df, _ = split_data
    explainer = build_explainer(gb_model)
    with pytest.raises(ValueError):
        explain_single_prediction(gb_model, explainer, val_df.head(2))
