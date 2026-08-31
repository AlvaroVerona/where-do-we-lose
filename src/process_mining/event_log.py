"""Loading and trace-level views of the event log.

These are the low-level building blocks (load data, build per-process
traces, compute process variants) that both the EDA/process-mining notebooks
and the bottleneck detection module build on top of.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_event_log(path: str | Path = "data/raw/event_log.csv") -> pd.DataFrame:
    """Load the event log, parse timestamps, and sort into recorded order
    (process_id, event_id) so downstream sequence-based logic can rely on it.
    """
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df.sort_values(["process_id", "event_id"], kind="stable").reset_index(drop=True)


def load_process_instances(path: str | Path = "data/raw/process_instances.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["application_date"])
    return df


def is_rework_activity(activity: str) -> bool:
    return activity.endswith("Rework")


def build_traces(event_log: pd.DataFrame) -> pd.Series:
    """One row per process_id -> tuple of activities in recorded order."""
    return event_log.groupby("process_id", sort=False)["activity"].apply(tuple)


def compute_variants(event_log: pd.DataFrame) -> pd.DataFrame:
    """Process variants: distinct activity sequences, ranked by frequency.

    A "variant" is the exact ordered sequence of activities a process went
    through (e.g. every rework loop makes a distinct variant). This is the
    standard process-mining view of "how many different ways does this
    process actually get executed".
    """
    traces = build_traces(event_log)
    variant_str = traces.apply(lambda t: " -> ".join(t))
    counts = variant_str.value_counts().reset_index()
    counts.columns = ["variant", "count"]
    counts["pct"] = counts["count"] / counts["count"].sum()
    counts["cumulative_pct"] = counts["pct"].cumsum()
    counts["n_steps"] = counts["variant"].str.count(" -> ") + 1
    return counts.reset_index(drop=True)
