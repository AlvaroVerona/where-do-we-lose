"""Cached data/model loading for the dashboard.

Every function here is a thin, `st.cache_data`/`st.cache_resource` wrapper
around a `src/` function - no business logic lives in this file, only
Streamlit's caching concern. This is what keeps the dashboard responsive:
the underlying CSVs (50k/385k rows) and the trained model are loaded once
per session, not on every widget interaction.
"""

from __future__ import annotations

from pathlib import Path

import subprocess
import sys

import pandas as pd
import streamlit as st

from src.data.generate_data import STAGE_ORDER
from src.models.explain import build_explainer
from src.models.predict import load_model
from src.process_mining.bottleneck import build_bottleneck_table
from src.process_mining.event_log import load_event_log, load_process_instances

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "delay_prediction_model.joblib"
REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"
ROOT = Path(__file__).resolve().parents[2]

# (artifact that must exist, module that produces it). Same commands as `make
# generate-data / train / optimize`, with their default arguments (50,000 applications, seed 42).
_ARTIFACT_STEPS = [
    (DATA_DIR / "event_log.csv", "src.data.generate_data"),
    (MODEL_PATH, "src.models.train"),
    (REPORTS_DIR / "scenario_comparison.csv", "src.optimization.resource_allocation"),
]


@st.cache_resource(show_spinner="First run: generating the dataset and training the model (about a minute)...")
def ensure_artifacts() -> None:
    """Build any missing generated artifact (data, model, scenario reports).

    The data and model are gitignored, so a fresh clone or a hosted deployment has
    none of them. Running the pipeline stages on first use makes the dashboard work
    out of the box; with everything already in place this does nothing.
    """
    for artifact, module in _ARTIFACT_STEPS:
        if not artifact.exists():
            subprocess.run([sys.executable, "-m", module], cwd=ROOT, check=True)


@st.cache_data(show_spinner="Loading process data...")
def get_process_instances() -> pd.DataFrame:
    ensure_artifacts()
    return load_process_instances(DATA_DIR / "process_instances.csv")


@st.cache_data(show_spinner="Loading event log...")
def get_event_log() -> pd.DataFrame:
    ensure_artifacts()
    return load_event_log(DATA_DIR / "event_log.csv")


@st.cache_data(show_spinner="Ranking bottlenecks...")
def get_bottleneck_table() -> pd.DataFrame:
    return build_bottleneck_table(get_event_log())


@st.cache_resource(show_spinner="Loading prediction model...")
def get_model():
    ensure_artifacts()
    return load_model(MODEL_PATH)


@st.cache_resource(show_spinner="Preparing explainer...")
def get_explainer():
    return build_explainer(get_model())


@st.cache_data(show_spinner=False)
def get_scenario_comparison() -> pd.DataFrame | None:
    ensure_artifacts()
    path = REPORTS_DIR / "scenario_comparison.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def get_optimized_staffing() -> pd.DataFrame | None:
    ensure_artifacts()
    path = REPORTS_DIR / "optimized_staffing.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def apply_filters(
    df: pd.DataFrame,
    date_range: tuple | None = None,
    region: list[str] | None = None,
    channel: list[str] | None = None,
    application_type: list[str] | None = None,
    customer_segment: list[str] | None = None,
) -> pd.DataFrame:
    """Shared filter logic for the process-analysis page - applied to a
    process_instances-shaped frame with an `application_date` column.
    """
    out = df
    if date_range and len(date_range) == 2:
        start, end = date_range
        out = out[(out["application_date"].dt.date >= start) & (out["application_date"].dt.date <= end)]
    if region:
        out = out[out["region"].isin(region)]
    if channel:
        out = out[out["channel"].isin(channel)]
    if application_type:
        out = out[out["application_type"].isin(application_type)]
    if customer_segment:
        out = out[out["customer_segment"].isin(customer_segment)]
    return out


__all__ = [
    "STAGE_ORDER",
    "get_process_instances",
    "get_event_log",
    "get_bottleneck_table",
    "get_model",
    "get_explainer",
    "get_scenario_comparison",
    "get_optimized_staffing",
    "apply_filters",
]
