"""Bottleneck detection: turn per-stage metrics into a single ranked score.

Builds the final bottleneck table (one row per process stage) from the
event log, combining stage duration, waiting time, volume, rework rate and
resource utilization into a documented `bottleneck_score`.

Usage:
    from src.process_mining.bottleneck import build_bottleneck_table
    table = build_bottleneck_table(event_log)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.generate_data import STAGE_ORDER
from src.process_mining.process_metrics import (
    compute_rework_rate_by_stage,
    compute_stage_durations,
    estimate_processing_and_waiting_time,
)

DEFAULT_WEIGHTS = {
    "processing_time": 0.30,
    "waiting_time": 0.30,
    "volume": 0.20,
    "rework_rate": 0.10,
    "utilization": 0.10,
}


def calculate_waiting_times(event_log: pd.DataFrame, stages: list[str] | None = None) -> pd.DataFrame:
    """Average/median/P90 estimated waiting time per stage.

    See `src/process_mining/process_metrics.py` for why waiting time is an
    *estimate* (derived from a per-activity low-percentile floor), not a
    direct measurement - the event log has only one timestamp per activity.
    """
    stages = stages or STAGE_ORDER
    sd = estimate_processing_and_waiting_time(compute_stage_durations(event_log, stages))
    out = sd.groupby("activity")["estimated_waiting_hours"].agg(
        avg_waiting_time="mean", median_waiting_time="median", p90_waiting_time=lambda s: s.quantile(0.9)
    )
    return out.reindex(stages).reset_index().rename(columns={"activity": "stage"})


def calculate_processing_times(event_log: pd.DataFrame, stages: list[str] | None = None) -> pd.DataFrame:
    """Average/median/P90 estimated processing (service) time per stage."""
    stages = stages or STAGE_ORDER
    sd = estimate_processing_and_waiting_time(compute_stage_durations(event_log, stages))
    out = sd.groupby("activity")["estimated_processing_hours"].agg(
        avg_processing_time="mean", median_processing_time="median", p90_processing_time=lambda s: s.quantile(0.9)
    )
    return out.reindex(stages).reset_index().rename(columns={"activity": "stage"})


def calculate_rework_rate(event_log: pd.DataFrame, stages: list[str] | None = None) -> pd.DataFrame:
    """Share of each stage's occurrences that triggered a rework loop."""
    stages = stages or STAGE_ORDER
    rework = compute_rework_rate_by_stage(event_log, stages)
    return rework[["stage", "rework_rate"]]


def calculate_resource_utilization(
    event_log: pd.DataFrame, stages: list[str] | None = None, peak_percentile: float = 0.95
) -> pd.DataFrame:
    """Estimated resource utilization per stage, as a share of each
    department's own demonstrated peak throughput.

    Note on method: an earlier version of this function tried to compute
    "hours worked" / "hours available" using the estimated processing-time
    split from `process_metrics.py`. That split is measured in *elapsed
    calendar hours* between two consecutive events (see that module's
    docstring), which are inflated relative to actual business-hours effort
    by weekends and off-hours (roughly 3x, per `data/README.md`). Dividing a
    calendar-hour quantity by a business-hours capacity is a unit mismatch,
    and it produced utilization figures above 100% for the busiest stage -
    a red flag caught by inspecting the output, not a plausible result.

    This function avoids that mismatch entirely by working in **volume**,
    not hours: for each stage, it computes each business day's throughput
    per employee (`stage occurrences that day / n_employees`), and expresses
    the average of that against the department's own `peak_percentile`
    (default P95) daily throughput per employee - i.e. "how close to this
    department's own best demonstrated pace does it run, on average". This
    is bounded, self-calibrating per department, and needs no assumption
    about how many hours a single case actually takes.

    Department headcount (`n_employees`) is inferred from the event log
    itself (distinct `employee_id` values seen working that stage), not
    read from the data generator's internal staffing config.
    """
    stages = stages or STAGE_ORDER
    df = event_log[event_log["activity"].isin(stages)].copy()
    df["day"] = df["timestamp"].dt.normalize()
    df = df[df["day"].dt.dayofweek < 5]  # the department only operates Mon-Fri
    full_business_days = pd.date_range(df["day"].min(), df["day"].max(), freq="B")

    rows = []
    for stage in stages:
        stage_df = df[df["activity"] == stage]
        n_employees = stage_df["employee_id"].nunique()
        daily_volume = stage_df.groupby("day").size().reindex(full_business_days, fill_value=0)
        per_employee_daily = daily_volume / n_employees if n_employees else daily_volume * 0.0
        peak_capacity = per_employee_daily.quantile(peak_percentile)
        utilization = per_employee_daily.mean() / peak_capacity if peak_capacity > 0 else np.nan
        rows.append(
            {
                "stage": stage,
                "n_employees": n_employees,
                "avg_daily_volume_per_employee": per_employee_daily.mean(),
                "peak_daily_volume_per_employee": peak_capacity,
                "resource_utilization": utilization,
            }
        )
    return pd.DataFrame(rows)


def calculate_stage_metrics(event_log: pd.DataFrame, stages: list[str] | None = None) -> pd.DataFrame:
    """Assemble the core per-stage metrics table: volume, processing time,
    waiting time, rework rate and resource utilization. Does not compute the
    bottleneck score - see `calculate_bottleneck_score`.
    """
    stages = stages or STAGE_ORDER
    volume = event_log[event_log["activity"].isin(stages)]["activity"].value_counts()
    volume = volume.reindex(stages).rename("volume").rename_axis("stage").reset_index()

    processing = calculate_processing_times(event_log, stages)
    waiting = calculate_waiting_times(event_log, stages)
    rework = calculate_rework_rate(event_log, stages)
    utilization = calculate_resource_utilization(event_log, stages)[["stage", "n_employees", "resource_utilization"]]

    metrics = volume.merge(processing, on="stage").merge(waiting, on="stage").merge(rework, on="stage").merge(
        utilization, on="stage"
    )
    return metrics


def _min_max_normalize(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(0.0, index=series.index)
    return (series - lo) / (hi - lo)


def calculate_bottleneck_score(stage_metrics: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.DataFrame:
    """Add a `bottleneck_score` column (0-1) to a stage metrics table.

    Score = weighted sum of min-max-normalized:
        avg_processing_time, avg_waiting_time, volume, rework_rate, resource_utilization

    Design note: the spec's illustrative formula multiplies the three
    normalized factors together. A pure product is fragile here - since
    every stage's `rework_rate` is small (a few percent) its normalized
    value is close to 0 for the least-rework stage, and multiplying would
    let that one factor crush the whole score even for a stage that is
    genuinely slow and overloaded. A weighted **sum** of normalized factors
    is the standard, more robust alternative (used in most operational
    scorecards) and is used here instead; weights are configurable via the
    `weights` argument and default to `DEFAULT_WEIGHTS`, chosen so that
    processing and waiting time (the direct cycle-time drivers) dominate,
    volume scales the score by how many applications are affected, and
    rework rate / utilization contribute as secondary quality/capacity
    signals.
    """
    weights = weights or DEFAULT_WEIGHTS
    df = stage_metrics.copy()
    norm = pd.DataFrame(
        {
            "processing_time": _min_max_normalize(df["avg_processing_time"]),
            "waiting_time": _min_max_normalize(df["avg_waiting_time"]),
            "volume": _min_max_normalize(df["volume"]),
            "rework_rate": _min_max_normalize(df["rework_rate"]),
            "utilization": _min_max_normalize(df["resource_utilization"]),
        }
    )
    df["bottleneck_score"] = sum(norm[k] * w for k, w in weights.items())
    return df


def rank_bottlenecks(stage_metrics_with_score: pd.DataFrame) -> pd.DataFrame:
    """Sort by bottleneck_score descending and assign rank 1..N."""
    df = stage_metrics_with_score.sort_values("bottleneck_score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df


def build_bottleneck_table(
    event_log: pd.DataFrame, stages: list[str] | None = None, weights: dict[str, float] | None = None
) -> pd.DataFrame:
    """Convenience one-call pipeline: metrics -> score -> rank."""
    metrics = calculate_stage_metrics(event_log, stages)
    scored = calculate_bottleneck_score(metrics, weights)
    return rank_bottlenecks(scored)
