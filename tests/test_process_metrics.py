"""Tests for process-mining metric computations."""

import pandas as pd
import pytest

from src.data.generate_data import STAGE_ORDER, generate_dataset
from src.process_mining.bottleneck import (
    build_bottleneck_table,
    calculate_bottleneck_score,
    calculate_processing_times,
    calculate_resource_utilization,
    calculate_rework_rate,
    calculate_stage_metrics,
    calculate_waiting_times,
    rank_bottlenecks,
)
from src.process_mining.event_log import build_traces, compute_variants
from src.process_mining.process_metrics import (
    compute_activity_frequency,
    compute_cycle_time_stats,
    compute_event_gaps,
    compute_rework_rate_by_stage,
    compute_sla_breach_rate,
    compute_stage_durations,
    compute_transitions,
    estimate_processing_and_waiting_time,
)


@pytest.fixture(scope="module")
def dataset():
    return generate_dataset(n_processes=1500, seed=99, start_date="2024-01-01", end_date="2024-04-30")


@pytest.fixture(scope="module")
def event_log(dataset):
    return dataset[1]


@pytest.fixture(scope="module")
def process_instances(dataset):
    return dataset[0]


def test_event_gaps_first_event_has_no_gap(event_log):
    # NB: groupby(...).first() skips NaN per column independently, so it
    # would not actually return each group's first row here - use head(1).
    gaps = compute_event_gaps(event_log)
    first_events = gaps.groupby("process_id", sort=False).head(1)
    assert (first_events["activity"] == "Application Submitted").all()
    assert first_events["gap_hours"].isna().all()


def test_event_gaps_are_nonnegative(event_log):
    gaps = compute_event_gaps(event_log)
    assert (gaps["gap_hours"].dropna() >= 0).all()


def test_stage_durations_only_contains_main_stages(event_log):
    sd = compute_stage_durations(event_log)
    assert set(sd["activity"].unique()) <= set(STAGE_ORDER)
    assert sd["gap_hours"].notna().all()


def test_estimate_processing_and_waiting_sums_to_gap(event_log):
    sd = compute_stage_durations(event_log)
    sd = estimate_processing_and_waiting_time(sd)
    total = sd["estimated_processing_hours"] + sd["estimated_waiting_hours"]
    pd.testing.assert_series_equal(total, sd["gap_hours"], check_names=False)
    assert (sd["estimated_processing_hours"] >= 0).all()
    assert (sd["estimated_waiting_hours"] >= 0).all()


def test_activity_frequency_counts_match_event_log(event_log):
    freq = compute_activity_frequency(event_log)
    assert freq["count"].sum() == len(event_log)
    assert set(freq["activity"]) == set(event_log["activity"].unique())


def test_transitions_reference_valid_activities(event_log):
    transitions = compute_transitions(event_log)
    valid_activities = set(event_log["activity"].unique())
    assert set(transitions["from_activity"]) <= valid_activities
    assert set(transitions["to_activity"]) <= valid_activities
    assert (transitions["count"] > 0).all()


def test_rework_rate_attributed_to_correct_stage(event_log):
    # Document Validation is the stage with by far the most rework in this
    # dataset (missing documents drive most rework loops) - regression test
    # for a naming-mismatch bug where "Document Rework" -> "Document" failed
    # to map back onto the "Document Validation" stage.
    rates = compute_rework_rate_by_stage(event_log).set_index("stage")
    assert rates.loc["Document Validation", "rework_count"] > 0
    assert rates.loc["Credit Assessment", "rework_count"] > 0
    assert rates.loc["Risk Review", "rework_count"] > 0
    assert rates["rework_rate"].between(0, 1).all()


def test_variants_cover_all_processes(event_log):
    variants = compute_variants(event_log)
    assert variants["count"].sum() == event_log["process_id"].nunique()
    assert variants["pct"].sum() == pytest.approx(1.0)


def test_build_traces_start_with_application_submitted(event_log):
    traces = build_traces(event_log)
    assert all(t[0] == "Application Submitted" for t in traces)


def test_cycle_time_stats_shape(process_instances):
    overall = compute_cycle_time_stats(process_instances)
    assert overall.loc[0, "n"] == len(process_instances)
    by_type = compute_cycle_time_stats(process_instances, group_by="application_type")
    assert set(by_type["application_type"]) == set(process_instances["application_type"].unique())


def test_sla_breach_rate_matches_manual_calc(process_instances):
    overall = compute_sla_breach_rate(process_instances)
    assert overall.loc[0, "sla_breach_rate"] == pytest.approx(process_instances["sla_breached"].mean())


# --- bottleneck detection --------------------------------------------------


def test_calculate_waiting_and_processing_times_cover_all_stages(event_log):
    waiting = calculate_waiting_times(event_log)
    processing = calculate_processing_times(event_log)
    assert list(waiting["stage"]) == STAGE_ORDER
    assert list(processing["stage"]) == STAGE_ORDER
    assert (waiting["avg_waiting_time"] >= 0).all()
    assert (processing["avg_processing_time"] >= 0).all()


def test_calculate_rework_rate_bounded(event_log):
    rework = calculate_rework_rate(event_log)
    assert rework["rework_rate"].between(0, 1).all()


def test_calculate_resource_utilization_bounded(event_log):
    util = calculate_resource_utilization(event_log)
    # bounded near [0, ~1] by construction (average vs. own P95 peak day);
    # allow a little headroom since a handful of days can exceed the P95 peak
    assert util["resource_utilization"].between(0, 1.5).all()
    assert (util["n_employees"] > 0).all()


def test_calculate_stage_metrics_shape(event_log):
    metrics = calculate_stage_metrics(event_log)
    assert list(metrics["stage"]) == STAGE_ORDER
    expected_cols = {
        "stage", "volume", "avg_processing_time", "median_processing_time", "p90_processing_time",
        "avg_waiting_time", "median_waiting_time", "p90_waiting_time", "rework_rate",
        "n_employees", "resource_utilization",
    }
    assert expected_cols <= set(metrics.columns)
    assert (metrics["volume"] > 0).all()


def test_bottleneck_score_is_bounded_and_varies(event_log):
    metrics = calculate_stage_metrics(event_log)
    scored = calculate_bottleneck_score(metrics)
    assert scored["bottleneck_score"].between(0, 1).all()
    assert scored["bottleneck_score"].nunique() > 1


def test_rank_bottlenecks_orders_by_score_descending(event_log):
    metrics = calculate_stage_metrics(event_log)
    scored = calculate_bottleneck_score(metrics)
    ranked = rank_bottlenecks(scored)
    assert list(ranked["rank"]) == list(range(1, len(ranked) + 1))
    assert ranked["bottleneck_score"].is_monotonic_decreasing


def test_build_bottleneck_table_identifies_credit_assessment_as_top_bottleneck(event_log):
    # Regression/sanity check: given this dataset's documented calibration
    # (data/README.md), Credit Assessment should consistently surface as the
    # top-ranked bottleneck - it has the longest processing time and among
    # the longest waiting time of any stage.
    table = build_bottleneck_table(event_log)
    assert table.loc[table["rank"] == 1, "stage"].iloc[0] == "Credit Assessment"
