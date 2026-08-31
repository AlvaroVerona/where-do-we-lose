"""Tests for synthetic data generation and validation."""

import numpy as np
import pandas as pd
import pytest

from src.data.generate_data import STAGE_ORDER, generate_dataset
from src.data.validation import (
    REQUIRED_EVENT_COLUMNS,
    REQUIRED_PROCESS_COLUMNS,
    check_event_ordering,
    check_process_id_uniqueness,
    validate_dataset,
)


@pytest.fixture(scope="module")
def small_dataset():
    process_df, event_df = generate_dataset(n_processes=500, seed=123, start_date="2024-01-01", end_date="2024-03-31")
    return process_df, event_df


def test_generate_dataset_row_counts(small_dataset):
    process_df, event_df = small_dataset
    assert len(process_df) == 500
    assert len(event_df) > len(process_df)  # every process has >= 2 events


def test_generate_dataset_reproducible():
    p1, e1 = generate_dataset(n_processes=200, seed=7, start_date="2024-01-01", end_date="2024-02-29")
    p2, e2 = generate_dataset(n_processes=200, seed=7, start_date="2024-01-01", end_date="2024-02-29")
    pd.testing.assert_frame_equal(p1, p2)
    pd.testing.assert_frame_equal(e1, e2)


def test_generate_dataset_different_seed_differs():
    p1, _ = generate_dataset(n_processes=200, seed=1, start_date="2024-01-01", end_date="2024-02-29")
    p2, _ = generate_dataset(n_processes=200, seed=2, start_date="2024-01-01", end_date="2024-02-29")
    assert not p1["total_processing_time_hours"].equals(p2["total_processing_time_hours"])


def test_required_columns_present(small_dataset):
    process_df, event_df = small_dataset
    assert set(REQUIRED_PROCESS_COLUMNS).issubset(process_df.columns)
    assert set(REQUIRED_EVENT_COLUMNS).issubset(event_df.columns)


def test_no_duplicate_process_ids(small_dataset):
    process_df, _ = small_dataset
    assert process_df["process_id"].is_unique
    assert check_process_id_uniqueness(process_df).passed


def test_no_duplicate_event_ids(small_dataset):
    _, event_df = small_dataset
    assert event_df["event_id"].is_unique


def test_valid_timestamps(small_dataset):
    _, event_df = small_dataset
    ts = pd.to_datetime(event_df["timestamp"])
    assert ts.notna().all()
    assert (ts >= pd.Timestamp("2000-01-01")).all()
    assert (ts <= pd.Timestamp("2100-01-01")).all()


def test_events_ordered_within_process(small_dataset):
    _, event_df = small_dataset
    assert check_event_ordering(event_df).passed


def test_every_process_has_start_event(small_dataset):
    process_df, event_df = small_dataset
    started = set(event_df.loc[event_df["activity"] == "Application Submitted", "process_id"])
    assert started == set(process_df["process_id"])


def test_completed_processes_have_completed_event(small_dataset):
    process_df, event_df = small_dataset
    completed_ids = set(process_df.loc[process_df["final_status"] == "Completed", "process_id"])
    has_completed_event = set(event_df.loc[event_df["activity"] == "Completed", "process_id"])
    assert completed_ids == has_completed_event


def test_rejected_processes_stop_before_disbursement(small_dataset):
    process_df, event_df = small_dataset
    rejected_ids = set(process_df.loc[process_df["final_status"] == "Rejected", "process_id"])
    has_completed_event = set(event_df.loc[event_df["activity"] == "Completed", "process_id"])
    assert rejected_ids.isdisjoint(has_completed_event)


def test_no_negative_durations(small_dataset):
    process_df, _ = small_dataset
    assert (process_df["total_processing_time_hours"] >= 0).all()


def test_activities_are_from_expected_set(small_dataset):
    _, event_df = small_dataset
    expected = {"Application Submitted", "Completed", *STAGE_ORDER}
    rework_activities = {a for a in event_df["activity"].unique() if a.endswith("Rework")}
    assert set(event_df["activity"].unique()) <= expected | rework_activities


def test_numeric_ranges_are_plausible(small_dataset):
    process_df, _ = small_dataset
    assert process_df["risk_score"].between(0, 100).all()
    assert process_df["complexity_score"].between(0, 100).all()
    assert (process_df["loan_amount"] > 0).all()


def test_full_validation_suite_passes(small_dataset):
    process_df, event_df = small_dataset
    report = validate_dataset(process_df, event_df)
    assert report.passed, report.summary()


def test_validation_catches_duplicate_process_id(small_dataset):
    process_df, event_df = small_dataset
    corrupted = process_df.copy()
    corrupted.loc[0, "process_id"] = corrupted.loc[1, "process_id"]
    report = validate_dataset(corrupted, event_df)
    assert not report.passed
    assert any(i.check == "process_id_uniqueness" for i in report.errors)


def test_validation_catches_negative_duration(small_dataset):
    process_df, event_df = small_dataset
    corrupted = process_df.copy()
    corrupted.loc[0, "total_processing_time_hours"] = -1.0
    report = validate_dataset(corrupted, event_df)
    assert not report.passed
    assert any(i.check == "negative_duration" for i in report.errors)


def test_validation_catches_out_of_order_events(small_dataset):
    process_df, event_df = small_dataset
    corrupted = event_df.copy()
    pid = corrupted.loc[10, "process_id"]
    idxs = corrupted[corrupted["process_id"] == pid].index
    a, b = idxs[0], idxs[1]
    corrupted.loc[a, "timestamp"], corrupted.loc[b, "timestamp"] = (
        corrupted.loc[b, "timestamp"],
        corrupted.loc[a, "timestamp"],
    )
    report = check_event_ordering(corrupted)
    assert not report.passed
