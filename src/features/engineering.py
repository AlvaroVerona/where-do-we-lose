"""Feature engineering for the SLA-breach prediction model.

The model answers one specific, time-bound question: **at the moment an
application is submitted, before any stage has processed it, what is the
probability it will end up breaching its SLA?** That prediction point is
deliberate and constrains every feature below - see "Target leakage" at the
bottom of this module.

Feature set (matches the project spec, section 15):
    customer_segment, application_type, loan_amount, risk_score,
    complexity_score, channel, region, priority, day_of_week, hour_of_day,
    current_queue_size

`previous_stage_duration` and `number_of_reworks` (also mentioned in the
spec as *potential* features) are deliberately **not** used: neither exists
yet at submission time - the process hasn't reached any stage, let alone
been reworked. Using them would require moving the prediction point
mid-process, which is a different (and more complex) product than "flag
risk on arrival", and the project's dashboard mock-up (spec section 21,
page 4) confirms arrival-time is the intended prediction point: its input
form has no fields for stage duration or rework count.
"""

from __future__ import annotations

import pandas as pd

TARGET_COLUMN = "sla_breached"

CATEGORICAL_FEATURES = ["customer_segment", "application_type", "channel", "region", "priority"]
NUMERIC_FEATURES = ["loan_amount", "risk_score", "complexity_score", "day_of_week", "hour_of_day", "current_queue_size"]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# Columns that only exist *after* the outcome is known - if any of these
# ever end up in the feature set, that's target leakage.
POST_OUTCOME_COLUMNS = [
    "final_status",
    "rejection_stage",
    "total_reworks",
    "total_processing_time_hours",
    "sla_breached",
]


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """day_of_week (0=Monday) and hour_of_day, both known the instant the
    application is submitted.
    """
    df = df.copy()
    df["day_of_week"] = df["application_date"].dt.dayofweek
    df["hour_of_day"] = df["application_date"].dt.hour
    return df


def add_current_queue_size(df: pd.DataFrame, window: str = "1D") -> pd.DataFrame:
    """Number of applications submitted in the trailing `window` immediately
    before this one - a proxy for how busy the intake system is right now,
    fully derivable from `application_date` alone (no future information).

    Uses a time-indexed rolling count with `closed="left"`, i.e. the window
    counts strictly-prior arrivals and excludes the application itself.
    """
    df = df.sort_values("application_date").copy()
    arrivals = df.set_index("application_date")["process_id"]
    rolling_counts = arrivals.rolling(window, closed="left").count()
    df["current_queue_size"] = rolling_counts.to_numpy()
    df["current_queue_size"] = df["current_queue_size"].fillna(0).astype(int)
    return df


def build_feature_frame(process_instances: pd.DataFrame, queue_window: str = "1D") -> pd.DataFrame:
    """Build the full modeling frame: process_id, every feature in
    FEATURE_COLUMNS, and the target column. Categorical columns are left as
    plain strings - encoding is a modeling-pipeline concern (see
    `src/models/train.py`), not a feature-engineering one.
    """
    df = process_instances.copy()
    df = add_temporal_features(df)
    df = add_current_queue_size(df, window=queue_window)
    df = df.sort_values("application_date").reset_index(drop=True)
    return df[["process_id", "application_date", *FEATURE_COLUMNS, TARGET_COLUMN]]


def check_no_leakage(feature_columns: list[str]) -> None:
    """Raise if any post-outcome column has ended up among the model's
    input features. Call this before training, not just at feature-build
    time, so a future refactor can't silently reintroduce leakage.
    """
    leaked = set(feature_columns) & set(POST_OUTCOME_COLUMNS)
    if leaked:
        raise ValueError(f"Target leakage detected: {sorted(leaked)} must not be used as model features.")


def time_based_train_val_test_split(
    df: pd.DataFrame, date_col: str = "application_date", train_frac: float = 0.70, val_frac: float = 0.15
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological 70/15/15 split (train/val/test), ordered by `date_col`.

    A random split would let the model "see the future" during training
    (e.g. training on an application submitted in June while validating on
    one from March) and would let queue-size/volume patterns leak across
    the split in unrealistic ways. In production this model would only ever
    be trained on the past and evaluated on the future, so the offline
    evaluation should mirror that: train on the earliest period, validate
    on the next, test on the most recent - the only split that measures
    what the model would actually do once deployed.
    """
    df = df.sort_values(date_col).reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    return train_df, val_df, test_df
