"""Tests for model training, evaluation, and prediction."""

import numpy as np
import pandas as pd
import pytest

from src.data.generate_data import generate_dataset
from src.features.engineering import FEATURE_COLUMNS, TARGET_COLUMN, build_feature_frame, time_based_train_val_test_split
from src.models.evaluate import compute_classification_metrics
from src.models.predict import predict_breach_probability, risk_category
from src.models.train import evaluate_models, get_model_specs, save_model, select_best_model, train_models


@pytest.fixture(scope="module")
def split_data():
    process_df, _ = generate_dataset(n_processes=4000, seed=21, start_date="2024-01-01", end_date="2024-12-31")
    features = build_feature_frame(process_df)
    return time_based_train_val_test_split(features)


@pytest.fixture(scope="module")
def fitted_models(split_data):
    train_df, _, _ = split_data
    return train_models(train_df, train_df[TARGET_COLUMN])


def test_get_model_specs_returns_three_models():
    specs = get_model_specs()
    assert len(specs) == 3
    assert {"logistic_regression", "random_forest", "gradient_boosting"} == set(specs.keys())


def test_train_models_fits_all_three(fitted_models):
    assert len(fitted_models) == 3
    for name, pipeline in fitted_models.items():
        assert hasattr(pipeline, "predict_proba"), f"{name} pipeline is not fitted"


def test_predictions_have_correct_shape(fitted_models, split_data):
    _, val_df, _ = split_data
    for pipeline in fitted_models.values():
        preds = pipeline.predict(val_df[FEATURE_COLUMNS])
        assert preds.shape == (len(val_df),)


def test_predict_proba_output_is_valid_probability(fitted_models, split_data):
    _, val_df, _ = split_data
    for pipeline in fitted_models.values():
        proba = pipeline.predict_proba(val_df[FEATURE_COLUMNS])
        assert proba.shape == (len(val_df), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert ((proba >= 0) & (proba <= 1)).all()


def test_compute_classification_metrics_bounded():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=200)
    y_proba = rng.random(200)
    y_pred = (y_proba >= 0.5).astype(int)
    metrics = compute_classification_metrics(y_true, y_pred, y_proba)
    for key in ["roc_auc", "pr_auc", "precision", "recall", "f1", "accuracy"]:
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0


def test_evaluate_models_ranks_by_roc_auc_descending(fitted_models, split_data):
    _, val_df, _ = split_data
    comparison = evaluate_models(fitted_models, val_df, val_df[TARGET_COLUMN])
    assert list(comparison.columns[:1]) == ["model"]
    assert comparison["roc_auc"].is_monotonic_decreasing


def test_select_best_model_matches_top_of_comparison(fitted_models, split_data):
    _, val_df, _ = split_data
    comparison = evaluate_models(fitted_models, val_df, val_df[TARGET_COLUMN])
    best_name, best_model = select_best_model(fitted_models, comparison)
    assert best_name == comparison.iloc[0]["model"]
    assert best_model is fitted_models[best_name]


def test_models_beat_random_guessing(fitted_models, split_data):
    _, val_df, _ = split_data
    comparison = evaluate_models(fitted_models, val_df, val_df[TARGET_COLUMN])
    # a non-informative model scores ~0.5 ROC-AUC; every trained model
    # should meaningfully beat that given informative, non-leaked features
    assert (comparison["roc_auc"] > 0.55).all()


def test_save_and_load_model_roundtrip(fitted_models, split_data, tmp_path):
    from src.models.predict import load_model

    _, val_df, _ = split_data
    model = fitted_models["random_forest"]
    path = tmp_path / "model.joblib"
    save_model(model, path)

    loaded = load_model(path)
    original_proba = model.predict_proba(val_df[FEATURE_COLUMNS])
    loaded_proba = loaded.predict_proba(val_df[FEATURE_COLUMNS])
    # RandomForest uses n_jobs=-1 (parallel tree averaging), which can differ
    # in floating-point summation order between calls at the ~1e-15 level -
    # allclose, not exact equality, is the correct check here.
    np.testing.assert_allclose(original_proba, loaded_proba, atol=1e-8)


def test_predict_breach_probability_matches_pipeline_output(fitted_models, split_data):
    _, val_df, _ = split_data
    model = fitted_models["gradient_boosting"]
    proba_series = predict_breach_probability(model, val_df)
    expected = model.predict_proba(val_df[FEATURE_COLUMNS])[:, 1]
    np.testing.assert_allclose(proba_series.to_numpy(), expected)


def test_predict_breach_probability_requires_feature_columns(fitted_models, split_data):
    _, val_df, _ = split_data
    model = fitted_models["gradient_boosting"]
    incomplete = val_df.drop(columns=["risk_score"])
    with pytest.raises(ValueError):
        predict_breach_probability(model, incomplete)


@pytest.mark.parametrize(
    "probability,expected",
    [(0.1, "LOW RISK"), (0.45, "MODERATE RISK"), (0.85, "HIGH RISK")],
)
def test_risk_category_thresholds(probability, expected):
    assert risk_category(probability) == expected
