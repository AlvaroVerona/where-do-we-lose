"""Resource allocation optimization: how should staff be redistributed
across process stages to reduce cycle time and SLA breaches?

Two distinct models are used here, deliberately kept separate:

1. An **analytical queueing proxy** (`expected_wait_hours`), a deterministic
   version of `generate_data.py`'s own `compute_wait_hours` formula (same
   business-hours queueing curve, expected noise instead of sampled noise).
   This is cheap to evaluate thousands of times, which is what OR-Tools
   needs while searching the staffing space - but it is only ever used
   *inside the optimizer* to pick a candidate allocation.

2. The **actual discrete-event simulation** (`src/data/generate_data.py`),
   re-run with each scenario's staffing plugged in via `temporary_staffing`.
   Every reported scenario metric (SLA breach rate, P90 processing time,
   throughput, ...) in this module and in the scenario-simulation notebook
   comes from *this* - real simulated data, not the analytical proxy. Per
   the project's rules ("do not invent these exact values"), the proxy
   never appears in a reported result, only in the solver's objective.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

from src.data.generate_data import BUSINESS_HOURS_PER_DAY, STAGE_CONFIG, STAGE_ORDER, generate_dataset
from src.process_mining.bottleneck import calculate_resource_utilization

# E[lognormal(mu=0, sigma)] = exp(sigma^2 / 2) - the expected value of the
# multiplicative noise terms `compute_wait_hours`/`compute_service_hours`
# sample from in the real simulator.
_WAIT_NOISE_MEAN = float(np.exp(0.25**2 / 2))


def current_staffing() -> dict[str, int]:
    return {stage: STAGE_CONFIG[stage]["n_employees"] for stage in STAGE_ORDER}


@contextlib.contextmanager
def temporary_staffing(overrides: dict[str, int]):
    """Temporarily mutate STAGE_CONFIG's headcount for the duration of the
    `with` block (e.g. while generating a scenario dataset), then restore
    the original values - even if an exception is raised.
    """
    originals = {stage: STAGE_CONFIG[stage]["n_employees"] for stage in overrides}
    try:
        for stage, n in overrides.items():
            STAGE_CONFIG[stage]["n_employees"] = n
        yield
    finally:
        for stage, n in originals.items():
            STAGE_CONFIG[stage]["n_employees"] = n


def expected_wait_hours(stage: str, n_employees: int, avg_daily_volume: float) -> float:
    """Deterministic version of `generate_data.compute_wait_hours`: expected
    queueing delay for `stage` given `n_employees` and average business-day
    arrival volume. Used only by the optimizer's search - see module
    docstring.
    """
    cfg = STAGE_CONFIG[stage]
    capacity = n_employees * BUSINESS_HOURS_PER_DAY / cfg["base_hours"] if n_employees > 0 else 0.0
    utilization = min(avg_daily_volume / capacity, 0.97) if capacity > 0 else 0.97
    wait = cfg["base_wait"] + cfg["wait_scale"] * (utilization / max(1 - utilization, 0.03)) * _WAIT_NOISE_MEAN
    return float(min(max(wait, 0.0), 120.0))


def compute_avg_daily_volume(event_log: pd.DataFrame, stages: list[str] | None = None) -> dict[str, float]:
    """Average business-day (Mon-Fri) occurrence count per stage, from the
    real event log - the same demand figure the optimizer weighs staffing
    decisions against.
    """
    stages = stages or STAGE_ORDER
    df = event_log[event_log["activity"].isin(stages)].copy()
    df["day"] = df["timestamp"].dt.normalize()
    df = df[df["day"].dt.dayofweek < 5]
    full_business_days = pd.date_range(df["day"].min(), df["day"].max(), freq="B")
    return {
        stage: df[df["activity"] == stage].groupby("day").size().reindex(full_business_days, fill_value=0).mean()
        for stage in stages
    }


@dataclass
class OptimizationResult:
    staffing: dict[str, int]
    status: str
    objective_value: float


def optimize_allocation(
    avg_daily_volume: dict[str, float],
    baseline_staffing: dict[str, int] | None = None,
    min_staff_fraction: float = 0.75,
    max_staff_fraction: float = 1.3,
    max_utilization: float = 0.95,
    budget_growth_fraction: float = 0.05,
    cost_per_employee: float = 1.0,
) -> OptimizationResult:
    """Redistribute (and modestly grow) headcount across stages to minimize
    a combined cost of expected queueing delay and employee cost, subject to
    staffing and capacity constraints.

    Objective (mirrors the project spec's `total_processing_time + lambda *
    SLA_breaches + employee_cost`, collapsed to two terms - see module
    docstring for why a separate SLA-breach term would be redundant here):

        minimize  sum_stage( volume_stage * expected_wait_hours(x_stage) )
                  + cost_per_employee * sum_stage( x_stage )

    The first term is a volume-weighted proxy for total cycle time impact;
    because SLA targets are fixed per application, less time in queue is
    monotonically fewer breaches, so a single term captures both without
    double-counting the same signal with two tuning weights.

    Constraints:
        - `min_staff_fraction`/`max_staff_fraction` of each stage's current
          headcount ("minimum staffing requirements" / a sanity ceiling).
        - Each stage must have enough staff to keep utilization under
          `max_utilization` ("capacity constraints").
        - Total headcount across stages may grow by at most
          `budget_growth_fraction` over the current total ("maximum
          available employees") - this is what forces genuine redistribution
          rather than the trivial "hire more everywhere" answer.
    """
    baseline_staffing = baseline_staffing or current_staffing()
    model = cp_model.CpModel()

    x: dict[str, cp_model.IntVar] = {}
    bounds: dict[str, tuple[int, int]] = {}
    for stage in STAGE_ORDER:
        cfg = STAGE_CONFIG[stage]
        current = baseline_staffing[stage]
        capacity_floor = int(np.ceil(avg_daily_volume[stage] * cfg["base_hours"] / (BUSINESS_HOURS_PER_DAY * max_utilization)))
        lo = max(1, int(current * min_staff_fraction), capacity_floor)
        hi = max(lo, int(current * max_staff_fraction))
        bounds[stage] = (lo, hi)
        x[stage] = model.new_int_var(lo, hi, f"x_{stage}")

    total_budget = int(sum(baseline_staffing.values()) * (1 + budget_growth_fraction))
    model.add(sum(x[stage] for stage in STAGE_ORDER) <= total_budget)

    # Precompute expected_wait_hours(n) for every feasible n per stage and
    # wire it in via an element (table-lookup) constraint, since the
    # queueing curve is nonlinear in n and CP-SAT needs linear/table terms.
    wait_cost_terms = []
    for stage in STAGE_ORDER:
        lo, hi = bounds[stage]
        volume = avg_daily_volume[stage]
        table = [int(round(expected_wait_hours(stage, n, volume) * volume * 100)) for n in range(lo, hi + 1)]
        index = model.new_int_var(0, hi - lo, f"idx_{stage}")
        model.add(index == x[stage] - lo)
        cost_var = model.new_int_var(min(table), max(table), f"cost_{stage}")
        model.add_element(index, table, cost_var)
        wait_cost_terms.append(cost_var)

    employee_cost_scaled = int(round(cost_per_employee * 100))
    model.minimize(sum(wait_cost_terms) + employee_cost_scaled * sum(x[stage] for stage in STAGE_ORDER))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    status = solver.solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"Optimization did not find a feasible solution (status={solver.status_name(status)})")

    staffing = {stage: solver.value(x[stage]) for stage in STAGE_ORDER}
    return OptimizationResult(staffing=staffing, status=solver.status_name(status), objective_value=solver.objective_value / 100)


def bottleneck_focused_staffing(
    bottleneck_stage: str, increase_fraction: float = 0.20, baseline_staffing: dict[str, int] | None = None
) -> dict[str, int]:
    """Scenario A: add resources only to the single largest bottleneck,
    leaving every other stage unchanged - the simplest possible response to
    "the bottleneck is X", for comparison against the full optimization.
    """
    baseline_staffing = dict(baseline_staffing or current_staffing())
    baseline_staffing[bottleneck_stage] = int(round(baseline_staffing[bottleneck_stage] * (1 + increase_fraction)))
    return baseline_staffing


def simulate_scenario(
    staffing: dict[str, int], n_processes: int, seed: int, start_date: str, end_date: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the real generator with `staffing` plugged in - this is what
    produces every reported scenario metric (see module docstring).
    """
    with temporary_staffing(staffing):
        return generate_dataset(n_processes, seed, start_date, end_date)


def summarize_scenario(name: str, staffing: dict[str, int], process_instances: pd.DataFrame, event_log: pd.DataFrame) -> dict:
    """The comparison metrics the spec asks for: average/P90 processing
    time, SLA breach rate, throughput, resource utilization, cost.
    """
    n_days = (process_instances["application_date"].max() - process_instances["application_date"].min()).days + 1
    total_cost = sum(staffing.values())
    avg_utilization = calculate_resource_utilization(event_log)["resource_utilization"].mean()
    return {
        "scenario": name,
        "total_employees": total_cost,
        "avg_processing_time_hours": process_instances["total_processing_time_hours"].mean(),
        "p90_processing_time_hours": process_instances["total_processing_time_hours"].quantile(0.9),
        "sla_breach_rate": process_instances["sla_breached"].mean(),
        "completion_rate": (process_instances["final_status"] == "Completed").mean(),
        "throughput_per_day": len(process_instances) / n_days,
        "avg_resource_utilization": avg_utilization,
    }


def run_scenario_comparison(
    n_processes: int = 50_000,
    seed: int = 42,
    start_date: str = "2024-01-01",
    end_date: str = "2025-06-30",
    data_dir: str = "data/raw",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Baseline (the already-generated dataset) vs. Scenario A (add to the
    single largest bottleneck) vs. Scenario B (OR-Tools optimized
    redistribution) - each run through the real simulation, not the proxy.

    Returns (scenario_comparison, staffing_by_stage) - both used directly by
    the Streamlit dashboard's Optimization page, precomputed here because
    each scenario takes ~10-15s to simulate at n_processes=50,000 and a
    dashboard page must not re-run that on every load.
    """
    from src.process_mining.bottleneck import build_bottleneck_table
    from src.process_mining.event_log import load_event_log, load_process_instances

    baseline_pi = load_process_instances(f"{data_dir}/process_instances.csv")
    baseline_el = load_event_log(f"{data_dir}/event_log.csv")
    baseline_staffing = current_staffing()

    avg_daily_volume = compute_avg_daily_volume(baseline_el)
    top_bottleneck = build_bottleneck_table(baseline_el).iloc[0]["stage"]

    scenario_a_staffing = bottleneck_focused_staffing(top_bottleneck, baseline_staffing=baseline_staffing)
    opt_result = optimize_allocation(avg_daily_volume, baseline_staffing)

    summaries = [summarize_scenario("Baseline", baseline_staffing, baseline_pi, baseline_el)]

    pi_a, el_a = simulate_scenario(scenario_a_staffing, n_processes, seed, start_date, end_date)
    summaries.append(summarize_scenario(f"A: Add to bottleneck ({top_bottleneck})", scenario_a_staffing, pi_a, el_a))

    pi_b, el_b = simulate_scenario(opt_result.staffing, n_processes, seed, start_date, end_date)
    summaries.append(summarize_scenario("B: Optimized", opt_result.staffing, pi_b, el_b))

    comparison = pd.DataFrame(summaries)

    staffing_rows = []
    for stage in STAGE_ORDER:
        staffing_rows.append(
            {
                "stage": stage,
                "current": baseline_staffing[stage],
                "scenario_a": scenario_a_staffing[stage],
                "optimized": opt_result.staffing[stage],
            }
        )
    staffing_by_stage = pd.DataFrame(staffing_rows)

    return comparison, staffing_by_stage


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run and save the baseline/bottleneck/optimized scenario comparison.")
    parser.add_argument("--n-processes", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-date", type=str, default="2024-01-01")
    parser.add_argument("--end-date", type=str, default="2025-06-30")
    parser.add_argument("--data-dir", type=str, default="data/raw")
    parser.add_argument("--output-dir", type=str, default="reports")
    args = parser.parse_args()

    comparison, staffing_by_stage = run_scenario_comparison(
        args.n_processes, args.seed, args.start_date, args.end_date, args.data_dir
    )

    from pathlib import Path

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_dir / "scenario_comparison.csv", index=False)
    staffing_by_stage.to_csv(output_dir / "optimized_staffing.csv", index=False)

    print(comparison.round(3).to_string(index=False))
    print()
    print(staffing_by_stage.to_string(index=False))
    print(f"\nSaved to {output_dir}/scenario_comparison.csv and {output_dir}/optimized_staffing.csv")


if __name__ == "__main__":
    main()
