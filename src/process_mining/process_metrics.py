"""Reusable process-mining metrics computed from the event log.

The generated event log records one timestamp per activity occurrence (its
completion), not separate start/complete lifecycle transitions. That means
"time between consecutive activities" (the gap) is directly observable, but
it mixes together queueing/waiting time and actual processing time - a
single completion timestamp cannot tell them apart on its own. Real
process-mining projects hit this same limitation whenever a source system
doesn't log activity start times.

`estimate_processing_and_waiting_time` documents and implements the standard
workaround: use a low percentile of each activity's observed duration as a
proxy for its "pure" processing time (the closest thing to a no-queue best
case actually observed), and attribute the remainder of each occurrence's
duration to waiting. This is an estimate, not a measurement - see
data/README.md and the Business Interpretation notes in the notebooks for
how it's used and its limits.
"""

from __future__ import annotations

import pandas as pd

from src.data.generate_data import REWORK_ACTIVITY, STAGE_ORDER
from src.process_mining.event_log import is_rework_activity

REWORK_TO_STAGE = {v: k for k, v in REWORK_ACTIVITY.items()}

DEFAULT_PROCESSING_FLOOR_PERCENTILE = 0.10


def compute_event_gaps(event_log: pd.DataFrame) -> pd.DataFrame:
    """One row per event with the time elapsed since the previous event in
    the same process (NaN for each process's first event). This gap is
    attributed to the *current* activity - e.g. the gap ending at a
    "Credit Assessment" event is that occurrence's wait + processing time.
    """
    df = event_log.sort_values(["process_id", "event_id"], kind="stable").copy()
    grouped = df.groupby("process_id", sort=False)
    df["prev_activity"] = grouped["activity"].shift(1)
    df["prev_timestamp"] = grouped["timestamp"].shift(1)
    df["gap_hours"] = (df["timestamp"] - df["prev_timestamp"]).dt.total_seconds() / 3600.0
    return df


def compute_activity_frequency(event_log: pd.DataFrame) -> pd.DataFrame:
    """How often each activity occurs, and in what share of processes."""
    counts = event_log["activity"].value_counts().reset_index()
    counts.columns = ["activity", "count"]
    counts["pct_of_events"] = counts["count"] / counts["count"].sum()
    n_processes = event_log["process_id"].nunique()
    processes_touched = event_log.groupby("activity")["process_id"].nunique()
    counts["pct_of_processes"] = counts["activity"].map(processes_touched) / n_processes
    return counts.sort_values("count", ascending=False).reset_index(drop=True)


def compute_transitions(event_log: pd.DataFrame) -> pd.DataFrame:
    """Frequency and timing of each observed (from_activity -> to_activity)
    transition - the edges of the process flow graph.
    """
    gaps = compute_event_gaps(event_log).dropna(subset=["prev_activity"])
    agg = gaps.groupby(["prev_activity", "activity"])["gap_hours"].agg(
        count="count", avg_hours="mean", median_hours="median"
    )
    agg["p90_hours"] = gaps.groupby(["prev_activity", "activity"])["gap_hours"].quantile(0.9)
    agg = agg.reset_index().rename(columns={"prev_activity": "from_activity", "activity": "to_activity"})
    return agg.sort_values("count", ascending=False).reset_index(drop=True)


def compute_stage_durations(event_log: pd.DataFrame, stages: list[str] | None = None) -> pd.DataFrame:
    """One row per occurrence of a main process stage (including repeat
    occurrences after rework), with the elapsed time since the previous
    event. Excludes the boundary "Application Submitted"/"Completed" events
    and rework activities themselves.
    """
    stages = stages or STAGE_ORDER
    gaps = compute_event_gaps(event_log)
    return gaps[gaps["activity"].isin(stages)].reset_index(drop=True)


def estimate_processing_and_waiting_time(
    stage_durations: pd.DataFrame, floor_percentile: float = DEFAULT_PROCESSING_FLOOR_PERCENTILE
) -> pd.DataFrame:
    """Add `estimated_processing_hours` and `estimated_waiting_hours` columns.

    `estimated_processing_hours` is capped at each activity's `floor_percentile`
    duration (a proxy for the fastest realistic no-queue case); everything
    above that floor is attributed to `estimated_waiting_hours`. See the
    module docstring for why this is an estimate rather than a direct
    measurement.
    """
    df = stage_durations.copy()
    floors = df.groupby("activity")["gap_hours"].quantile(floor_percentile)
    df["processing_floor_hours"] = df["activity"].map(floors)
    df["estimated_processing_hours"] = df[["gap_hours", "processing_floor_hours"]].min(axis=1)
    df["estimated_waiting_hours"] = (df["gap_hours"] - df["estimated_processing_hours"]).clip(lower=0)
    return df


def compute_rework_events(event_log: pd.DataFrame) -> pd.DataFrame:
    """One row per rework occurrence, with the stage it belongs to."""
    rework = event_log[event_log["activity"].apply(is_rework_activity)].copy()
    rework["stage"] = rework["activity"].map(REWORK_TO_STAGE)
    return rework


def compute_rework_rate_by_stage(event_log: pd.DataFrame, stages: list[str] | None = None) -> pd.DataFrame:
    """Share of each stage's occurrences that were followed by a rework loop."""
    stages = stages or STAGE_ORDER
    stage_counts = event_log[event_log["activity"].isin(stages)]["activity"].value_counts()
    rework_counts = compute_rework_events(event_log)["stage"].value_counts()
    out = pd.DataFrame({"stage_occurrences": stage_counts, "rework_count": rework_counts}).fillna(0)
    out["rework_count"] = out["rework_count"].astype(int)
    out["rework_rate"] = out["rework_count"] / out["stage_occurrences"]
    return out.reindex(stages).reset_index().rename(columns={"index": "stage"})


def compute_cycle_time_stats(process_instances: pd.DataFrame, group_by: str | list[str] | None = None) -> pd.DataFrame:
    """Mean/median/P90/P95 total processing time, optionally grouped."""
    col = "total_processing_time_hours"
    if group_by is None:
        return pd.DataFrame(
            {
                "n": [len(process_instances)],
                "mean_hours": [process_instances[col].mean()],
                "median_hours": [process_instances[col].median()],
                "p90_hours": [process_instances[col].quantile(0.9)],
                "p95_hours": [process_instances[col].quantile(0.95)],
            }
        )
    grouped = process_instances.groupby(group_by)[col]
    return grouped.agg(
        n="count", mean_hours="mean", median_hours="median",
        p90_hours=lambda s: s.quantile(0.9), p95_hours=lambda s: s.quantile(0.95),
    ).reset_index()


def compute_sla_breach_rate(process_instances: pd.DataFrame, group_by: str | list[str] | None = None) -> pd.DataFrame:
    if group_by is None:
        return pd.DataFrame(
            {"n": [len(process_instances)], "sla_breach_rate": [process_instances["sla_breached"].mean()]}
        )
    grouped = process_instances.groupby(group_by)["sla_breached"]
    return grouped.agg(n="count", sla_breach_rate="mean").reset_index()
