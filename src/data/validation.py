"""Automated data validation for the process instance table and event log.

Every check returns a list of human-readable issues instead of raising, so a
full report can be produced in one pass. `validate_dataset` runs the whole
suite and aggregates the result into a `ValidationReport`.

Usage:
    python -m src.data.validation --data-dir data/raw
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.data.generate_data import REWORK_ACTIVITY, STAGE_ORDER

ALL_ACTIVITIES = ["Application Submitted", *STAGE_ORDER, "Completed", *REWORK_ACTIVITY.values()]
TERMINAL_ACTIVITIES = {"Completed", "Rejected"}

REQUIRED_PROCESS_COLUMNS = [
    "process_id",
    "application_date",
    "customer_segment",
    "application_type",
    "loan_amount",
    "risk_score",
    "complexity_score",
    "channel",
    "region",
    "priority",
    "final_status",
    "total_processing_time_hours",
    "sla_target_hours",
    "sla_breached",
]

REQUIRED_EVENT_COLUMNS = [
    "process_id",
    "event_id",
    "activity",
    "timestamp",
    "department",
    "employee_id",
    "status",
]

# (column, min, max) - inclusive plausibility bounds, not hard business limits
NUMERIC_RANGES = [
    ("loan_amount", 0, 5_000_000),
    ("risk_score", 0, 100),
    ("complexity_score", 0, 100),
    ("total_processing_time_hours", 0, 10_000),
    ("sla_target_hours", 0, 5_000),
]

MIN_PLAUSIBLE_DATE = pd.Timestamp("2000-01-01")
MAX_PLAUSIBLE_DATE = pd.Timestamp("2100-01-01")


@dataclass
class ValidationIssue:
    check: str
    message: str
    severity: str = "error"  # "error" or "warning"


@dataclass
class ValidationReport:
    issues: list[ValidationIssue] = field(default_factory=list)

    def add(self, check: str, message: str, severity: str = "error") -> None:
        self.issues.append(ValidationIssue(check, message, severity))

    def extend(self, other: "ValidationReport") -> None:
        self.issues.extend(other.issues)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def passed(self) -> bool:
        return len(self.errors) == 0

    def summary(self) -> str:
        if not self.issues:
            return "All checks passed."
        lines = []
        for issue in self.issues:
            tag = "ERROR" if issue.severity == "error" else "WARNING"
            lines.append(f"[{tag}] {issue.check}: {issue.message}")
        status = "PASSED" if self.passed else "FAILED"
        lines.append(f"\nValidation {status} — {len(self.errors)} error(s), {len(self.warnings)} warning(s).")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# process_instances checks
# ---------------------------------------------------------------------------


def check_required_columns(df: pd.DataFrame, required: list[str], label: str) -> ValidationReport:
    report = ValidationReport()
    missing = [c for c in required if c not in df.columns]
    if missing:
        report.add("required_columns", f"{label} is missing required columns: {missing}")
    return report


def check_process_id_uniqueness(df: pd.DataFrame) -> ValidationReport:
    report = ValidationReport()
    if "process_id" not in df.columns:
        return report
    n_missing = df["process_id"].isna().sum()
    if n_missing:
        report.add("process_id_missing", f"{n_missing} row(s) have a missing process_id.")
    dupes = df["process_id"].dropna()
    dupe_ids = dupes[dupes.duplicated()].unique()
    if len(dupe_ids):
        report.add(
            "process_id_uniqueness",
            f"{len(dupe_ids)} duplicated process_id value(s), e.g. {list(dupe_ids[:5])}.",
        )
    return report


def check_numeric_ranges(df: pd.DataFrame) -> ValidationReport:
    report = ValidationReport()
    for col, lo, hi in NUMERIC_RANGES:
        if col not in df.columns:
            continue
        out_of_range = df[(df[col] < lo) | (df[col] > hi)]
        if len(out_of_range):
            report.add(
                "numeric_range",
                f"{len(out_of_range)} row(s) have {col} outside the plausible range [{lo}, {hi}] "
                f"(min={df[col].min()}, max={df[col].max()}).",
            )
        n_null = df[col].isna().sum()
        if n_null:
            report.add("numeric_range", f"{n_null} row(s) have a missing {col}.", severity="warning")
    return report


def check_no_negative_durations(df: pd.DataFrame) -> ValidationReport:
    report = ValidationReport()
    if "total_processing_time_hours" not in df.columns:
        return report
    negative = df[df["total_processing_time_hours"] < 0]
    if len(negative):
        report.add(
            "negative_duration",
            f"{len(negative)} process(es) have a negative total_processing_time_hours, e.g. "
            f"{list(negative['process_id'].head(5))}.",
        )
    return report


def check_final_status_values(df: pd.DataFrame) -> ValidationReport:
    report = ValidationReport()
    if "final_status" not in df.columns:
        return report
    valid = {"Completed", "Rejected"}
    invalid = df[~df["final_status"].isin(valid)]
    if len(invalid):
        report.add(
            "final_status_values",
            f"{len(invalid)} process(es) have an unexpected final_status: "
            f"{sorted(invalid['final_status'].unique())}.",
        )
    return report


def validate_process_instances(df: pd.DataFrame) -> ValidationReport:
    report = ValidationReport()
    report.extend(check_required_columns(df, REQUIRED_PROCESS_COLUMNS, "process_instances"))
    report.extend(check_process_id_uniqueness(df))
    report.extend(check_numeric_ranges(df))
    report.extend(check_no_negative_durations(df))
    report.extend(check_final_status_values(df))
    return report


# ---------------------------------------------------------------------------
# event_log checks
# ---------------------------------------------------------------------------


def check_event_timestamps_valid(df: pd.DataFrame) -> ValidationReport:
    report = ValidationReport()
    if "timestamp" not in df.columns:
        return report
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    n_invalid = ts.isna().sum() - df["timestamp"].isna().sum()
    if n_invalid:
        report.add("timestamp_validity", f"{n_invalid} event(s) have an unparseable timestamp.")
    n_missing = df["timestamp"].isna().sum()
    if n_missing:
        report.add("timestamp_validity", f"{n_missing} event(s) have a missing timestamp.")
    impossible = ts[(ts.notna()) & ((ts < MIN_PLAUSIBLE_DATE) | (ts > MAX_PLAUSIBLE_DATE))]
    if len(impossible):
        report.add(
            "impossible_timestamp",
            f"{len(impossible)} event(s) fall outside the plausible date range "
            f"[{MIN_PLAUSIBLE_DATE.date()}, {MAX_PLAUSIBLE_DATE.date()}].",
        )
    return report


def check_expected_activities(df: pd.DataFrame) -> ValidationReport:
    report = ValidationReport()
    if "activity" not in df.columns:
        return report
    unexpected = set(df["activity"].dropna().unique()) - set(ALL_ACTIVITIES)
    if unexpected:
        report.add("expected_activities", f"Unexpected activity value(s) found: {sorted(unexpected)}.")
    missing = set(STAGE_ORDER) - set(df["activity"].unique())
    if missing:
        report.add("expected_activities", f"Expected stage(s) never appear in the event log: {sorted(missing)}.")
    return report


def check_event_ordering(df: pd.DataFrame) -> ValidationReport:
    """Timestamps must be non-decreasing within a process, and the sequence of
    main-stage activities (ignoring rework/boundary events) must never move
    backward through STAGE_ORDER.

    Deliberately does NOT sort by timestamp first: the point of this check is
    to verify that the event log is already recorded in chronological order
    (as it should be, e.g. sorted by event_id), so it evaluates rows in the
    order they appear in the file/frame.
    """
    report = ValidationReport()
    if not {"process_id", "activity", "timestamp"}.issubset(df.columns):
        return report

    stage_index = {stage: i for i, stage in enumerate(STAGE_ORDER)}
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if "event_id" in df.columns:
        df = df.sort_values(["process_id", "event_id"], kind="stable")
    else:
        df = df.sort_values("process_id", kind="stable")

    non_monotonic = 0
    out_of_order = 0
    for _, g in df.groupby("process_id", sort=False):
        ts = g["timestamp"].to_numpy()
        if (ts[1:] < ts[:-1]).any():
            non_monotonic += 1
        stage_seq = [stage_index[a] for a in g["activity"] if a in stage_index]
        if any(b < a for a, b in zip(stage_seq, stage_seq[1:])):
            out_of_order += 1

    if non_monotonic:
        report.add("event_ordering", f"{non_monotonic} process(es) have non-monotonic event timestamps.")
    if out_of_order:
        report.add("event_ordering", f"{out_of_order} process(es) visit stages out of the expected process order.")
    return report


def validate_event_log(df: pd.DataFrame) -> ValidationReport:
    report = ValidationReport()
    report.extend(check_required_columns(df, REQUIRED_EVENT_COLUMNS, "event_log"))
    report.extend(check_event_timestamps_valid(df))
    report.extend(check_expected_activities(df))
    report.extend(check_event_ordering(df))
    n_missing_event_id = df["event_id"].isna().sum() if "event_id" in df.columns else 0
    if n_missing_event_id:
        report.add("event_id_missing", f"{n_missing_event_id} row(s) have a missing event_id.")
    if "event_id" in df.columns:
        dupes = df["event_id"].dropna()
        dupe_ids = dupes[dupes.duplicated()].unique()
        if len(dupe_ids):
            report.add("event_id_uniqueness", f"{len(dupe_ids)} duplicated event_id value(s).")
    return report


# ---------------------------------------------------------------------------
# Cross-dataset checks
# ---------------------------------------------------------------------------


def check_every_process_has_start_event(process_df: pd.DataFrame, event_df: pd.DataFrame) -> ValidationReport:
    report = ValidationReport()
    if "process_id" not in process_df.columns or not {"process_id", "activity"}.issubset(event_df.columns):
        return report
    started = set(event_df.loc[event_df["activity"] == "Application Submitted", "process_id"])
    missing = set(process_df["process_id"]) - started
    if missing:
        report.add(
            "missing_start_event",
            f"{len(missing)} process(es) have no 'Application Submitted' event, e.g. {list(missing)[:5]}.",
        )
    return report


def check_completed_processes_have_final_event(process_df: pd.DataFrame, event_df: pd.DataFrame) -> ValidationReport:
    report = ValidationReport()
    required_cols = {"process_id", "final_status"}
    if not required_cols.issubset(process_df.columns) or not {"process_id", "activity"}.issubset(event_df.columns):
        return report
    completed_ids = set(process_df.loc[process_df["final_status"] == "Completed", "process_id"])
    has_completed_event = set(event_df.loc[event_df["activity"] == "Completed", "process_id"])
    missing = completed_ids - has_completed_event
    if missing:
        report.add(
            "missing_final_event",
            f"{len(missing)} process(es) marked Completed have no 'Completed' event, e.g. {list(missing)[:5]}.",
        )
    rejected_ids = set(process_df.loc[process_df["final_status"] == "Rejected", "process_id"])
    unexpected_completed_event = rejected_ids & has_completed_event
    if unexpected_completed_event:
        report.add(
            "unexpected_final_event",
            f"{len(unexpected_completed_event)} process(es) marked Rejected still have a 'Completed' event.",
        )
    return report


def check_process_ids_consistent(process_df: pd.DataFrame, event_df: pd.DataFrame) -> ValidationReport:
    report = ValidationReport()
    if "process_id" not in process_df.columns or "process_id" not in event_df.columns:
        return report
    orphan_events = set(event_df["process_id"]) - set(process_df["process_id"])
    if orphan_events:
        report.add(
            "orphan_events",
            f"{len(orphan_events)} process_id(s) appear in event_log but not in process_instances, "
            f"e.g. {list(orphan_events)[:5]}.",
        )
    return report


def validate_dataset(process_df: pd.DataFrame, event_df: pd.DataFrame) -> ValidationReport:
    """Run the full validation suite on both tables plus cross-table checks."""
    report = ValidationReport()
    report.extend(validate_process_instances(process_df))
    report.extend(validate_event_log(event_df))
    report.extend(check_every_process_has_start_event(process_df, event_df))
    report.extend(check_completed_processes_have_final_event(process_df, event_df))
    report.extend(check_process_ids_consistent(process_df, event_df))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the generated process data.")
    parser.add_argument("--data-dir", type=str, default="data/raw")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    process_df = pd.read_csv(data_dir / "process_instances.csv")
    event_df = pd.read_csv(data_dir / "event_log.csv")

    report = validate_dataset(process_df, event_df)
    print(report.summary())
    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
