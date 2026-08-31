"""Tests for feature engineering (SLA-breach model inputs)."""

import pandas as pd
import pytest

from src.data.generate_data import generate_dataset
from src.features.engineering import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    POST_OUTCOME_COLUMNS,
    TARGET_COLUMN,
    add_current_queue_size,
    add_temporal_features,
    build_feature_frame,
    check_no_leakage,
    time_based_train_val_test_split,
)


@pytest.fixture(scope="module")
def process_instances():
    process_df, _ = generate_dataset(n_processes=2000, seed=11, start_date="2024-01-01", end_date="2024-06-30")
    return process_df


def test_build_feature_frame_has_expected_columns(process_instances):
    feat = build_feature_frame(process_instances)
    assert set(FEATURE_COLUMNS) <= set(feat.columns)
    assert TARGET_COLUMN in feat.columns
    assert len(feat) == len(process_instances)


def test_build_feature_frame_no_missing_values(process_instances):
    feat = build_feature_frame(process_instances)
    assert feat[FEATURE_COLUMNS].isna().sum().sum() == 0


def test_temporal_features_are_derived_correctly(process_instances):
    df = add_temporal_features(process_instances)
    assert (df["day_of_week"] == df["application_date"].dt.dayofweek).all()
    assert (df["hour_of_day"] == df["application_date"].dt.hour).all()
    assert df["day_of_week"].between(0, 6).all()
    assert df["hour_of_day"].between(0, 23).all()


def test_current_queue_size_is_nonnegative_and_excludes_self(process_instances):
    df = add_current_queue_size(process_instances)
    assert (df["current_queue_size"] >= 0).all()
    # the very first arrival in the dataset has no one ahead of it
    first_row = df.sort_values("application_date").iloc[0]
    assert first_row["current_queue_size"] == 0


def test_current_queue_size_matches_manual_count_for_one_row(process_instances):
    df = add_current_queue_size(process_instances).sort_values("application_date").reset_index(drop=True)
    row = df.iloc[100]
    window_start = row["application_date"] - pd.Timedelta(days=1)
    manual_count = ((df["application_date"] >= window_start) & (df["application_date"] < row["application_date"])).sum()
    assert row["current_queue_size"] == manual_count


def test_no_leakage_passes_for_real_feature_list():
    check_no_leakage(FEATURE_COLUMNS)  # should not raise


def test_no_leakage_detects_post_outcome_columns():
    for leaky_col in POST_OUTCOME_COLUMNS:
        with pytest.raises(ValueError):
            check_no_leakage(FEATURE_COLUMNS + [leaky_col])


def test_feature_lists_are_disjoint_and_cover_all_features():
    assert set(CATEGORICAL_FEATURES).isdisjoint(NUMERIC_FEATURES)
    assert set(CATEGORICAL_FEATURES) | set(NUMERIC_FEATURES) == set(FEATURE_COLUMNS)


def test_time_based_split_is_chronological_and_nonoverlapping(process_instances):
    feat = build_feature_frame(process_instances)
    train, val, test = time_based_train_val_test_split(feat)

    assert len(train) + len(val) + len(test) == len(feat)
    assert train["application_date"].max() <= val["application_date"].min()
    assert val["application_date"].max() <= test["application_date"].min()

    # roughly 70/15/15
    n = len(feat)
    assert len(train) == pytest.approx(n * 0.70, abs=2)
    assert len(val) == pytest.approx(n * 0.15, abs=2)


def test_time_based_split_uses_all_original_rows(process_instances):
    feat = build_feature_frame(process_instances)
    train, val, test = time_based_train_val_test_split(feat)
    combined_ids = set(train["process_id"]) | set(val["process_id"]) | set(test["process_id"])
    assert combined_ids == set(feat["process_id"])
