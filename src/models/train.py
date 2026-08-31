"""Train and compare SLA-breach prediction models.

Three models, in increasing complexity, per the project spec:
    1. Logistic Regression (baseline)
    2. Random Forest
    3. HistGradientBoostingClassifier (spec's approved substitute for
       XGBoost - it ships with scikit-learn, so it avoids an extra
       dependency for no real benefit at this dataset size)

All three share the same preprocessing (one-hot encode categoricals, scale
numerics) and are trained with `class_weight="balanced"` since SLA breach is
a minority class (~20-25% of cases, see data/README.md).

Usage:
    python -m src.models.train
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.engineering import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    build_feature_frame,
    check_no_leakage,
    time_based_train_val_test_split,
)
from src.models.evaluate import compute_classification_metrics
from src.process_mining.event_log import load_process_instances

RANDOM_STATE = 42


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ]
    )


def get_model_specs() -> dict[str, object]:
    """Model name -> unfitted estimator. Pipelines are assembled in `train_models`."""
    return {
        "logistic_regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=20,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "gradient_boosting": HistGradientBoostingClassifier(
            max_iter=300, max_depth=6, learning_rate=0.05,
            class_weight="balanced", random_state=RANDOM_STATE,
        ),
    }


def train_models(X_train: pd.DataFrame, y_train: pd.Series) -> dict[str, Pipeline]:
    """Fit one Pipeline(preprocessor, model) per model spec."""
    fitted = {}
    for name, estimator in get_model_specs().items():
        pipeline = Pipeline([("preprocessor", build_preprocessor()), ("model", estimator)])
        pipeline.fit(X_train[FEATURE_COLUMNS], y_train)
        fitted[name] = pipeline
    return fitted


def evaluate_models(fitted_models: dict[str, Pipeline], X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    rows = []
    for name, pipeline in fitted_models.items():
        y_proba = pipeline.predict_proba(X[FEATURE_COLUMNS])[:, 1]
        y_pred = pipeline.predict(X[FEATURE_COLUMNS])
        metrics = compute_classification_metrics(y, y_pred, y_proba)
        rows.append({"model": name, **metrics})
    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)


def select_best_model(fitted_models: dict[str, Pipeline], comparison: pd.DataFrame) -> tuple[str, Pipeline]:
    """Best model by validation ROC-AUC (the primary ranking metric for this
    imbalanced binary problem - see evaluate.py for why accuracy is not used).
    """
    best_name = comparison.iloc[0]["model"]
    return best_name, fitted_models[best_name]


def save_model(model: Pipeline, path: str | Path = "models/delay_prediction_model.joblib") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and compare SLA-breach prediction models.")
    parser.add_argument("--data-dir", type=str, default="data/raw")
    parser.add_argument("--output", type=str, default="models/delay_prediction_model.joblib")
    args = parser.parse_args()

    check_no_leakage(FEATURE_COLUMNS)

    process_instances = load_process_instances(f"{args.data_dir}/process_instances.csv")
    features = build_feature_frame(process_instances)
    train_df, val_df, test_df = time_based_train_val_test_split(features)

    print(f"train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")
    print(f"breach rate  train={train_df[TARGET_COLUMN].mean():.1%}  "
          f"val={val_df[TARGET_COLUMN].mean():.1%}  test={test_df[TARGET_COLUMN].mean():.1%}")

    fitted_models = train_models(train_df, train_df[TARGET_COLUMN])

    val_comparison = evaluate_models(fitted_models, val_df, val_df[TARGET_COLUMN])
    print("\nValidation set performance:")
    print(val_comparison.round(4).to_string(index=False))

    best_name, best_model = select_best_model(fitted_models, val_comparison)
    print(f"\nBest model on validation ROC-AUC: {best_name}")

    test_metrics = compute_classification_metrics(
        test_df[TARGET_COLUMN],
        best_model.predict(test_df[FEATURE_COLUMNS]),
        best_model.predict_proba(test_df[FEATURE_COLUMNS])[:, 1],
    )
    print("\nBest model, held-out test set performance:")
    for k, v in test_metrics.items():
        print(f"  {k:12s} {v:.4f}")

    save_model(best_model, args.output)
    print(f"\nSaved best model ({best_name}) to {args.output}")


if __name__ == "__main__":
    main()
