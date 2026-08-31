"""Tests for resource allocation optimization."""

import pytest

from src.data.generate_data import STAGE_CONFIG, STAGE_ORDER, generate_dataset
from src.optimization.resource_allocation import (
    bottleneck_focused_staffing,
    compute_avg_daily_volume,
    current_staffing,
    expected_wait_hours,
    optimize_allocation,
    simulate_scenario,
    summarize_scenario,
    temporary_staffing,
)


@pytest.fixture(scope="module")
def small_dataset():
    return generate_dataset(n_processes=2000, seed=17, start_date="2024-01-01", end_date="2024-06-30")


@pytest.fixture(scope="module")
def avg_daily_volume(small_dataset):
    _, event_log = small_dataset
    return compute_avg_daily_volume(event_log)


def test_current_staffing_matches_stage_config():
    staffing = current_staffing()
    for stage in STAGE_ORDER:
        assert staffing[stage] == STAGE_CONFIG[stage]["n_employees"]


def test_temporary_staffing_restores_original_values_on_success():
    original = STAGE_CONFIG["Credit Assessment"]["n_employees"]
    with temporary_staffing({"Credit Assessment": original + 50}):
        assert STAGE_CONFIG["Credit Assessment"]["n_employees"] == original + 50
    assert STAGE_CONFIG["Credit Assessment"]["n_employees"] == original


def test_temporary_staffing_restores_original_values_on_exception():
    original = STAGE_CONFIG["Credit Assessment"]["n_employees"]
    with pytest.raises(RuntimeError):
        with temporary_staffing({"Credit Assessment": original + 50}):
            raise RuntimeError("boom")
    assert STAGE_CONFIG["Credit Assessment"]["n_employees"] == original


def test_expected_wait_hours_increases_with_utilization():
    low_wait = expected_wait_hours("Credit Assessment", n_employees=300, avg_daily_volume=100)
    high_wait = expected_wait_hours("Credit Assessment", n_employees=50, avg_daily_volume=100)
    assert high_wait > low_wait
    assert low_wait >= 0


def test_expected_wait_hours_bounded():
    wait = expected_wait_hours("Credit Assessment", n_employees=1, avg_daily_volume=1000)
    assert 0 <= wait <= 120


def test_compute_avg_daily_volume_covers_all_stages(avg_daily_volume):
    assert set(avg_daily_volume.keys()) == set(STAGE_ORDER)
    assert all(v >= 0 for v in avg_daily_volume.values())


def test_bottleneck_focused_staffing_only_changes_target_stage():
    baseline = current_staffing()
    scenario = bottleneck_focused_staffing("Credit Assessment", increase_fraction=0.2, baseline_staffing=baseline)
    for stage in STAGE_ORDER:
        if stage == "Credit Assessment":
            assert scenario[stage] == int(round(baseline[stage] * 1.2))
        else:
            assert scenario[stage] == baseline[stage]


def test_optimize_allocation_respects_budget_constraint(avg_daily_volume):
    baseline = current_staffing()
    result = optimize_allocation(avg_daily_volume, baseline, budget_growth_fraction=0.05)
    assert result.status in ("OPTIMAL", "FEASIBLE")
    total_current = sum(baseline.values())
    total_optimized = sum(result.staffing.values())
    assert total_optimized <= total_current * 1.05 + 1e-6


def test_optimize_allocation_respects_staffing_bounds(avg_daily_volume):
    baseline = current_staffing()
    result = optimize_allocation(
        avg_daily_volume, baseline, min_staff_fraction=0.75, max_staff_fraction=1.3, budget_growth_fraction=0.05
    )
    for stage in STAGE_ORDER:
        assert result.staffing[stage] >= 1
        assert result.staffing[stage] <= baseline[stage] * 1.3 + 1e-6


def test_optimize_allocation_all_stages_present(avg_daily_volume):
    result = optimize_allocation(avg_daily_volume, current_staffing())
    assert set(result.staffing.keys()) == set(STAGE_ORDER)


def test_simulate_scenario_applies_custom_staffing_and_restores_config():
    baseline = current_staffing()
    custom = dict(baseline)
    custom["Credit Assessment"] = baseline["Credit Assessment"] + 100

    process_df, event_log = simulate_scenario(custom, n_processes=500, seed=5, start_date="2024-01-01", end_date="2024-02-29")
    assert len(process_df) == 500
    assert STAGE_CONFIG["Credit Assessment"]["n_employees"] == baseline["Credit Assessment"]


def test_more_staff_at_bottleneck_reduces_average_processing_time():
    baseline = current_staffing()
    heavy_staffing = dict(baseline)
    heavy_staffing["Credit Assessment"] = baseline["Credit Assessment"] * 3

    kwargs = dict(n_processes=3000, seed=8, start_date="2024-01-01", end_date="2024-12-31")
    pi_baseline, _ = simulate_scenario(baseline, **kwargs)
    pi_heavy, _ = simulate_scenario(heavy_staffing, **kwargs)

    assert pi_heavy["total_processing_time_hours"].mean() < pi_baseline["total_processing_time_hours"].mean()
    assert pi_heavy["sla_breached"].mean() <= pi_baseline["sla_breached"].mean()


def test_summarize_scenario_shape(small_dataset):
    process_df, event_log = small_dataset
    summary = summarize_scenario("test", current_staffing(), process_df, event_log)
    expected_keys = {
        "scenario", "total_employees", "avg_processing_time_hours", "p90_processing_time_hours",
        "sla_breach_rate", "completion_rate", "throughput_per_day", "avg_resource_utilization",
    }
    assert expected_keys <= set(summary.keys())
    assert 0 <= summary["sla_breach_rate"] <= 1
    assert 0 <= summary["completion_rate"] <= 1
