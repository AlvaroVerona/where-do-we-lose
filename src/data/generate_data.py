"""Synthetic data generator for the Process Bottleneck Detection & Optimization System.

Simulates a customer loan application process end to end (application ->
document validation -> credit assessment -> risk review -> approval ->
contract generation -> disbursement -> completed / rejected), producing:

- data/raw/process_instances.csv  (one row per application)
- data/raw/event_log.csv          (one row per activity occurrence)

The simulation is not purely random: every stage's duration and queueing
delay is driven by explicit, documented business rules (risk, complexity,
loan size, missing documents, staffing levels, day-of-week arrival load,
weekend capacity). See data/README.md for the full modeling rationale.

Usage:
    python -m src.data.generate_data --n-processes 50000 --seed 42
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Business hours calendar
# ---------------------------------------------------------------------------

BUSINESS_START_HOUR = 8
BUSINESS_END_HOUR = 18
BUSINESS_HOURS_PER_DAY = BUSINESS_END_HOUR - BUSINESS_START_HOUR  # 10


def _next_business_day_start(ts: pd.Timestamp) -> pd.Timestamp:
    nxt = ts.normalize() + pd.Timedelta(days=1)
    while nxt.weekday() >= 5:  # Sat=5, Sun=6
        nxt += pd.Timedelta(days=1)
    return nxt + pd.Timedelta(hours=BUSINESS_START_HOUR)


def _snap_to_business_window(ts: pd.Timestamp) -> pd.Timestamp:
    """Push a timestamp forward to the next moment work is actually happening."""
    if ts.weekday() >= 5:
        return _next_business_day_start(ts)
    start = ts.normalize() + pd.Timedelta(hours=BUSINESS_START_HOUR)
    end = ts.normalize() + pd.Timedelta(hours=BUSINESS_END_HOUR)
    if ts < start:
        return start
    if ts >= end:
        return _next_business_day_start(ts)
    return ts


def add_business_hours(start: pd.Timestamp, hours: float) -> pd.Timestamp:
    """Advance `start` by `hours` of effort, consumed only during business hours
    (Mon-Fri, 08:00-18:00). This is what makes weekends and after-hours arrivals
    push work into the next business day, and lets long queues span multiple days.
    """
    current = _snap_to_business_window(start)
    remaining = float(hours)
    while remaining > 1e-9:
        day_end = current.normalize() + pd.Timedelta(hours=BUSINESS_END_HOUR)
        available = (day_end - current).total_seconds() / 3600.0
        if remaining <= available:
            current = current + pd.Timedelta(hours=remaining)
            remaining = 0.0
        else:
            remaining -= available
            current = _next_business_day_start(current)
    return current


# ---------------------------------------------------------------------------
# Process configuration
# ---------------------------------------------------------------------------

STAGE_ORDER = [
    "Document Validation",
    "Credit Assessment",
    "Risk Review",
    "Approval",
    "Contract Generation",
    "Disbursement",
]

# Per-stage operational configuration. `n_employees` and `base_hours` jointly
# determine daily capacity (n_employees * BUSINESS_HOURS_PER_DAY / base_hours).
# Business-hours effort only accrues Mon-Fri 08:00-18:00 (add_business_hours),
# so 1 hour of "effort" costs ~3.4 elapsed calendar hours on average once
# weekends and off-hours are accounted for - staffing below is calibrated
# against that amplification and against the dataset's average *business-day*
# arrival volume (~110-140/day) so that total cycle times land in a realistic
# range. Credit Assessment and Risk Review are staffed to run at higher
# utilization (~0.45-0.50) than the other stages (~0.25-0.30): combined with
# their longer base_hours, they emerge as the process bottlenecks from volume
# + duration + queueing, not because the score is hardcoded. `wait_scale` sets
# how sharply queueing delay grows with utilization (an M/M/1-style u/(1-u)
# curve); it is higher for Credit Assessment/Risk Review so the same
# utilization produces more waiting than in the other departments.
STAGE_CONFIG = {
    "Document Validation": {"department": "Documentation", "n_employees": 119, "base_hours": 3.0, "base_wait": 0.5, "wait_scale": 1.5},
    "Credit Assessment":   {"department": "Credit",        "n_employees": 155, "base_hours": 6.5, "base_wait": 1.0, "wait_scale": 3.0},
    "Risk Review":         {"department": "Risk",          "n_employees": 142, "base_hours": 5.0, "base_wait": 1.0, "wait_scale": 2.5},
    "Approval":            {"department": "Approvals",     "n_employees": 95, "base_hours": 2.0, "base_wait": 0.3, "wait_scale": 1.0},
    "Contract Generation": {"department": "Contracts",     "n_employees": 139, "base_hours": 3.5, "base_wait": 0.3, "wait_scale": 1.0},
    "Disbursement":        {"department": "Disbursement",  "n_employees": 119, "base_hours": 2.5, "base_wait": 0.3, "wait_scale": 0.8},
}

REWORK_ACTIVITY = {
    "Document Validation": "Document Rework",
    "Credit Assessment": "Credit Rework",
    "Risk Review": "Risk Rework",
    "Approval": "Approval Rework",
    "Contract Generation": "Contract Rework",
    "Disbursement": "Disbursement Rework",
}

MAX_REWORKS_PER_STAGE = 2

CUSTOMER_SEGMENTS = ["Retail", "SME", "Premium", "Corporate"]
CUSTOMER_SEGMENT_WEIGHTS = [0.55, 0.25, 0.15, 0.05]

APPLICATION_TYPES = ["Personal Loan", "Auto Loan", "Mortgage", "Business Loan"]
APPLICATION_TYPE_WEIGHTS = [0.40, 0.25, 0.20, 0.15]

CHANNELS = ["Branch", "Online", "Mobile App", "Broker"]
CHANNEL_WEIGHTS = [0.30, 0.30, 0.25, 0.15]

REGIONS = ["North", "South", "East", "West", "Central"]
REGION_WEIGHTS = [0.22, 0.18, 0.20, 0.15, 0.25]

# lognormal(mu, sigma) parameters of loan_amount by application type, plus clip bounds
LOAN_AMOUNT_PARAMS = {
    "Personal Loan": {"mu": np.log(8_000), "sigma": 0.5, "min": 1_000, "max": 40_000},
    "Auto Loan":      {"mu": np.log(25_000), "sigma": 0.45, "min": 5_000, "max": 90_000},
    "Mortgage":        {"mu": np.log(280_000), "sigma": 0.4, "min": 60_000, "max": 900_000},
    "Business Loan":   {"mu": np.log(150_000), "sigma": 0.6, "min": 10_000, "max": 2_000_000},
}

# base SLA target hours (calendar hours) by application type, before
# priority/complexity adjustment. Calibrated against the simulated process
# (see data/README.md) so that a realistic minority of cases breach SLA.
SLA_BASE_HOURS = {
    "Personal Loan": 156,
    "Auto Loan": 186,
    "Mortgage": 288,
    "Business Loan": 246,
}

# relative weekday arrival weights (Mon..Sun) - Monday/Friday spikes create backlog
WEEKDAY_ARRIVAL_WEIGHTS = np.array([1.30, 1.00, 0.95, 1.00, 1.15, 0.20, 0.15])

# inherent per-type handling multiplier applied to every stage's service time,
# on top of the risk/complexity-driven multipliers - mortgages and business
# loans involve more paperwork/underwriting regardless of any one application's
# individual complexity score.
TYPE_SERVICE_MULT = {
    "Personal Loan": 0.85,
    "Auto Loan": 0.95,
    "Mortgage": 1.35,
    "Business Loan": 1.25,
}


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
# Employee pools
# ---------------------------------------------------------------------------


def build_employee_pools(rng: np.random.Generator) -> dict[str, list[tuple[str, float]]]:
    """One pool of (employee_id, speed_multiplier) per stage's department.

    speed_multiplier > 1 means the employee is faster than average; this is
    what creates "some employees process applications faster than others".
    """
    pools: dict[str, list[tuple[str, float]]] = {}
    for stage, cfg in STAGE_CONFIG.items():
        dept_code = cfg["department"][:4].upper()
        n = cfg["n_employees"]
        speeds = np.clip(rng.normal(1.0, 0.15, size=n), 0.65, 1.55)
        pools[stage] = [(f"EMP_{dept_code}_{i:03d}", float(speeds[i])) for i in range(n)]
    return pools


# ---------------------------------------------------------------------------
# Application-level (process instance) metadata generation
# ---------------------------------------------------------------------------


def sample_arrival_timestamps(n: int, start_date: pd.Timestamp, end_date: pd.Timestamp, channels: np.ndarray, rng: np.random.Generator) -> pd.DatetimeIndex:
    """Sample submission timestamps with weekday-weighted volume and
    channel-dependent hour-of-day patterns (branch = business hours,
    digital channels = broader/evening skew).
    """
    n_days = (end_date - start_date).days + 1
    dates = pd.date_range(start_date, periods=n_days, freq="D")
    day_weights = WEEKDAY_ARRIVAL_WEIGHTS[dates.weekday]
    day_weights = day_weights / day_weights.sum()
    day_idx = rng.choice(n_days, size=n, p=day_weights)
    chosen_dates = dates[day_idx]

    hours = np.empty(n)
    is_branch = channels == "Branch"
    hours[is_branch] = rng.uniform(BUSINESS_START_HOUR, BUSINESS_END_HOUR, size=is_branch.sum())
    n_digital = (~is_branch).sum()
    # evening-skewed hour distribution for digital/broker channels (24/7 submission)
    digital_hours = rng.normal(16, 5, size=n_digital)
    hours[~is_branch] = np.clip(digital_hours, 0, 23.99)

    minutes = rng.integers(0, 60, size=n)
    offsets = pd.to_timedelta(hours, unit="h") + pd.to_timedelta(minutes, unit="m")
    timestamps = pd.Series(pd.DatetimeIndex(chosen_dates) + offsets)

    # hours (up to 23.99) + minutes (up to 59) can sum to just over 24h,
    # rolling an application assigned to the last day into the next calendar
    # day - clip so every timestamp stays within [start_date, end_date].
    day_end_cap = end_date.normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59)
    timestamps = timestamps.clip(upper=day_end_cap)
    return pd.DatetimeIndex(timestamps)


def sample_process_metadata(n: int, start_date: pd.Timestamp, end_date: pd.Timestamp, rng: np.random.Generator) -> pd.DataFrame:
    segment = rng.choice(CUSTOMER_SEGMENTS, size=n, p=CUSTOMER_SEGMENT_WEIGHTS)
    app_type = rng.choice(APPLICATION_TYPES, size=n, p=APPLICATION_TYPE_WEIGHTS)
    channel = rng.choice(CHANNELS, size=n, p=CHANNEL_WEIGHTS)
    region = rng.choice(REGIONS, size=n, p=REGION_WEIGHTS)

    application_date = sample_arrival_timestamps(n, start_date, end_date, channel, rng)

    # priority: premium/corporate customers are far more likely to be fast-tracked
    high_priority_base = np.select(
        [segment == "Corporate", segment == "Premium", segment == "SME"],
        [0.60, 0.45, 0.20],
        default=0.12,
    )
    priority = np.where(rng.random(n) < high_priority_base, "High", "Standard")

    # loan amount, lognormal per application type
    loan_amount = np.empty(n)
    for t, params in LOAN_AMOUNT_PARAMS.items():
        mask = app_type == t
        cnt = mask.sum()
        vals = rng.lognormal(params["mu"], params["sigma"], size=cnt)
        loan_amount[mask] = np.clip(vals, params["min"], params["max"])

    # risk score: beta-distributed (skewed low-risk), shifted up for business
    # loans, corporate segment and large loan amounts
    risk_base = rng.beta(2.0, 5.0, size=n) * 100
    type_risk_shift = np.select(
        [app_type == "Business Loan", app_type == "Mortgage"],
        [12.0, 4.0],
        default=0.0,
    )
    segment_risk_shift = np.select(
        [segment == "Corporate", segment == "Retail"], [8.0, 2.0], default=0.0
    )
    # relative loan size within its own type also nudges risk up
    type_amount_pct = np.zeros(n)
    for t in APPLICATION_TYPES:
        mask = app_type == t
        type_amount_pct[mask] = pd.Series(loan_amount[mask]).rank(pct=True).to_numpy()
    amount_risk_shift = type_amount_pct * 10.0
    risk_score = np.clip(risk_base + type_risk_shift + segment_risk_shift + amount_risk_shift, 1, 99)

    # missing documents: more common on digital, self-service channels
    missing_doc_prob = np.select(
        [channel == "Online", channel == "Mobile App", channel == "Broker"],
        [0.30, 0.32, 0.20],
        default=0.10,
    )
    missing_documents = rng.random(n) < missing_doc_prob

    # complexity score: larger/riskier/less-documented applications are more complex
    complexity_base = rng.normal(35, 12, size=n)
    type_complexity_shift = np.select(
        [app_type == "Mortgage", app_type == "Business Loan"], [20.0, 25.0], default=0.0
    )
    complexity_score = np.clip(
        complexity_base
        + type_complexity_shift
        + type_amount_pct * 25.0
        + missing_documents * 12.0
        + (risk_score / 100) * 10.0,
        1,
        99,
    )

    # SLA target: base by type, tightened for priority, loosened for complex cases
    sla_target = np.array([SLA_BASE_HOURS[t] for t in app_type], dtype=float)
    sla_target = np.where(priority == "High", sla_target * 0.85, sla_target)
    sla_target = np.where(complexity_score > 70, sla_target * 1.15, sla_target)

    df = pd.DataFrame(
        {
            "application_date": application_date,
            "customer_segment": segment,
            "application_type": app_type,
            "loan_amount": loan_amount.round(2),
            "risk_score": risk_score.round(1),
            "complexity_score": complexity_score.round(1),
            "channel": channel,
            "region": region,
            "priority": priority,
            "missing_documents": missing_documents,
            "sla_target_hours": sla_target.round(1),
        }
    )
    # flag applications in the top decile of loan amount for their own type
    # ("large application amounts should require additional review")
    df["large_amount_flag"] = type_amount_pct >= 0.90
    return df


# ---------------------------------------------------------------------------
# Rejection assignment
# ---------------------------------------------------------------------------


def assign_rejections(df: pd.DataFrame, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Decide up front which applications will ultimately be rejected, and at
    which stage, based on risk, missing documents and loan size.
    """
    risk_z = (df["risk_score"] / 100).to_numpy()
    reject_logit = (
        -2.7
        + 3.2 * risk_z**1.5
        + 0.7 * df["missing_documents"].to_numpy()
        + 0.5 * df["large_amount_flag"].to_numpy()
        + 0.3 * (df["complexity_score"] / 100).to_numpy()
    )
    reject_prob = sigmoid(reject_logit)
    will_reject = rng.random(len(df)) < reject_prob

    stage_choices = np.array(["Document Validation", "Credit Assessment", "Risk Review"])
    reject_stage = np.full(len(df), "", dtype=object)
    for i in np.where(will_reject)[0]:
        w = np.array(
            [
                0.35 if df["missing_documents"].iat[i] else 0.05,
                0.45,
                0.20 + 0.50 * risk_z[i],
            ]
        )
        w = w / w.sum()
        reject_stage[i] = rng.choice(stage_choices, p=w)
    return will_reject, reject_stage


# ---------------------------------------------------------------------------
# Per-instance stage simulation
# ---------------------------------------------------------------------------


def compute_wait_hours(stage: str, arrival_ts: pd.Timestamp, row: pd.Series, daily_volume: dict, capacity: dict, rng: np.random.Generator) -> float:
    cfg = STAGE_CONFIG[stage]
    day = arrival_ts.normalize()
    volume = daily_volume.get(day, 0)
    cap = capacity[stage]
    utilization = min(volume / cap, 0.97) if cap > 0 else 0.97
    noise = rng.lognormal(0.0, 0.25)
    wait = cfg["base_wait"] + cfg["wait_scale"] * (utilization / max(1 - utilization, 0.03)) * noise
    # High-priority applications are expedited (queue-jumped) by the department
    if row["priority"] == "High":
        wait *= 0.4
    return float(min(max(wait, 0.0), 120.0))


def compute_service_hours(stage: str, row: pd.Series, rng: np.random.Generator) -> float:
    cfg = STAGE_CONFIG[stage]
    mult = 1.0
    if stage in ("Document Validation", "Credit Assessment", "Contract Generation"):
        mult *= 1 + 0.9 * (row["complexity_score"] / 100)
    if stage in ("Credit Assessment", "Risk Review"):
        mult *= 1 + 1.1 * (row["risk_score"] / 100)
    if stage in ("Risk Review", "Approval") and row["large_amount_flag"]:
        mult *= 1.4
    if stage == "Document Validation" and row["missing_documents"]:
        mult *= 1.7
    mult *= TYPE_SERVICE_MULT[row["application_type"]]
    noise = rng.lognormal(0.0, 0.20)
    return cfg["base_hours"] * mult * noise


def rework_probability(stage: str, row: pd.Series, attempt: int) -> float:
    base = {
        "Document Validation": 0.08,
        "Credit Assessment": 0.05,
        "Risk Review": 0.04,
        "Approval": 0.02,
        "Contract Generation": 0.02,
        "Disbursement": 0.015,
    }[stage]
    if stage == "Document Validation" and row["missing_documents"]:
        base += 0.35
    if stage == "Risk Review" and row["large_amount_flag"]:
        base += 0.10
    return base * (0.5 ** (attempt - 1))


@dataclass
class SimResult:
    events: list[dict]
    final_status: str
    rejection_stage: str
    total_reworks: int
    completion_ts: pd.Timestamp


def simulate_instance(
    process_id: str,
    row: pd.Series,
    reject_stage: str,
    employee_pools: dict,
    capacity: dict,
    daily_volume: dict,
    rng: np.random.Generator,
) -> SimResult:
    events: list[dict] = []
    current_ts = row["application_date"]

    events.append(
        {
            "process_id": process_id,
            "activity": "Application Submitted",
            "timestamp": current_ts,
            "department": "",
            "employee_id": "",
            "status": "Completed",
        }
    )

    total_reworks = 0
    final_status = "Completed"
    rejection_stage = ""

    for stage in STAGE_ORDER:
        cfg = STAGE_CONFIG[stage]
        is_reject_stage = stage == reject_stage
        attempt = 0
        while True:
            attempt += 1
            wait = compute_wait_hours(stage, current_ts, row, daily_volume, capacity, rng)
            service = compute_service_hours(stage, row, rng)
            pool = employee_pools[stage]
            emp_id, speed = pool[rng.integers(0, len(pool))]
            service = service / speed
            current_ts = add_business_hours(current_ts, wait + service)

            will_rework = attempt <= MAX_REWORKS_PER_STAGE and rng.random() < rework_probability(stage, row, attempt)
            stage_status = "Rejected" if (is_reject_stage and (not will_rework)) else "Completed"

            events.append(
                {
                    "process_id": process_id,
                    "activity": stage,
                    "timestamp": current_ts,
                    "department": cfg["department"],
                    "employee_id": emp_id,
                    "status": stage_status,
                }
            )

            if will_rework:
                total_reworks += 1
                rework_service = cfg["base_hours"] * 0.4 * rng.lognormal(0.0, 0.2) / speed
                current_ts = add_business_hours(current_ts, rework_service)
                events.append(
                    {
                        "process_id": process_id,
                        "activity": REWORK_ACTIVITY[stage],
                        "timestamp": current_ts,
                        "department": cfg["department"],
                        "employee_id": emp_id,
                        "status": "Completed",
                    }
                )
                continue

            break

        if is_reject_stage:
            final_status = "Rejected"
            rejection_stage = stage
            break

    if final_status == "Completed":
        events.append(
            {
                "process_id": process_id,
                "activity": "Completed",
                "timestamp": current_ts,
                "department": "",
                "employee_id": "",
                "status": "Completed",
            }
        )

    return SimResult(events, final_status, rejection_stage, total_reworks, current_ts)


# ---------------------------------------------------------------------------
# Top-level generation
# ---------------------------------------------------------------------------


def generate_dataset(
    n_processes: int, seed: int, start_date: str, end_date: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    meta = sample_process_metadata(n_processes, start_ts, end_ts, rng)
    meta.insert(0, "process_id", [f"PROC_{i:07d}" for i in range(n_processes)])

    will_reject, reject_stage = assign_rejections(meta, rng)

    daily_volume = meta["application_date"].dt.normalize().value_counts().to_dict()
    capacity = {
        stage: cfg["n_employees"] * BUSINESS_HOURS_PER_DAY / cfg["base_hours"]
        for stage, cfg in STAGE_CONFIG.items()
    }
    employee_pools = build_employee_pools(rng)

    all_events: list[dict] = []
    final_status_list: list[str] = []
    rejection_stage_list: list[str] = []
    total_reworks_list: list[int] = []
    total_processing_hours_list: list[float] = []

    for i in range(n_processes):
        row = meta.iloc[i]
        result = simulate_instance(
            row["process_id"],
            row,
            reject_stage[i] if will_reject[i] else "",
            employee_pools,
            capacity,
            daily_volume,
            rng,
        )
        all_events.extend(result.events)
        final_status_list.append(result.final_status)
        rejection_stage_list.append(result.rejection_stage)
        total_reworks_list.append(result.total_reworks)
        hours = (result.completion_ts - row["application_date"]).total_seconds() / 3600.0
        total_processing_hours_list.append(round(hours, 2))

    meta["final_status"] = final_status_list
    meta["rejection_stage"] = rejection_stage_list
    meta["total_reworks"] = total_reworks_list
    meta["total_processing_time_hours"] = total_processing_hours_list
    meta["sla_breached"] = meta["total_processing_time_hours"] > meta["sla_target_hours"]
    meta = meta.drop(columns=["large_amount_flag"])
    meta["missing_documents"] = meta["missing_documents"].astype(int)
    meta["sla_breached"] = meta["sla_breached"].astype(int)

    event_log = pd.DataFrame(all_events)
    event_log = event_log.sort_values(["process_id", "timestamp"]).reset_index(drop=True)
    event_log.insert(1, "event_id", [f"EVT_{i:08d}" for i in range(len(event_log))])

    process_instances = meta[
        [
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
            "missing_documents",
            "final_status",
            "rejection_stage",
            "total_reworks",
            "total_processing_time_hours",
            "sla_target_hours",
            "sla_breached",
        ]
    ]

    return process_instances, event_log


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic loan process data.")
    parser.add_argument("--n-processes", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-date", type=str, default="2024-01-01")
    parser.add_argument("--end-date", type=str, default="2025-06-30")
    parser.add_argument("--output-dir", type=str, default="data/raw")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    process_instances, event_log = generate_dataset(
        args.n_processes, args.seed, args.start_date, args.end_date
    )

    process_instances.to_csv(output_dir / "process_instances.csv", index=False)
    event_log.to_csv(output_dir / "event_log.csv", index=False)

    completion_rate = (process_instances["final_status"] == "Completed").mean()
    sla_breach_rate = process_instances["sla_breached"].mean()
    avg_hours = process_instances["total_processing_time_hours"].mean()

    print(f"Generated {len(process_instances):,} process instances and {len(event_log):,} events")
    print(f"  Date range:        {args.start_date} to {args.end_date}")
    print(f"  Completion rate:   {completion_rate:.1%}")
    print(f"  Rejection rate:    {1 - completion_rate:.1%}")
    print(f"  SLA breach rate:   {sla_breach_rate:.1%}")
    print(f"  Avg processing:    {avg_hours:.1f} hours")
    print(f"  Avg events/process:{len(event_log) / len(process_instances):.2f}")
    print(f"  Saved to:          {output_dir}/process_instances.csv")
    print(f"                     {output_dir}/event_log.csv")


if __name__ == "__main__":
    main()
