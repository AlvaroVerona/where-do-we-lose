"""Page 1 - Executive Overview.

Entry point of the multi-page Streamlit app (`streamlit run app/app.py`).
Streamlit auto-discovers the other four pages in `app/pages/`.
"""

from __future__ import annotations

import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_APP_DIR.parent))  # project root, for `src.*`
sys.path.insert(0, str(_APP_DIR))  # app dir, for `components.*`

import plotly.express as px
import streamlit as st

from components.data import get_process_instances
from components.style import STATUS, inject_css, kpi_card, register_plotly_template

st.set_page_config(page_title="Where do We Lose? — Executive Overview", page_icon="⏱️", layout="wide")
register_plotly_template()
inject_css()

st.title("Where do We Lose? (time and money)")
st.caption("Process Bottleneck Detection & Optimization System — Customer Loan Application Process")

pi = get_process_instances()

st.markdown("## Executive Overview")
st.markdown(
    '<p class="section-note">Portfolio-level view of how the loan application process is performing right now. '
    "All figures below come directly from the generated process data — see the sidebar pages for the "
    "process-mining, bottleneck, prediction and optimization detail behind them.</p>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# KPI cards
# ---------------------------------------------------------------------------

total_applications = len(pi)
avg_processing = pi["total_processing_time_hours"].mean()
p90_processing = pi["total_processing_time_hours"].quantile(0.9)
sla_breach_rate = pi["sla_breached"].mean()
rework_rate = (pi["total_reworks"] > 0).mean()

cols = st.columns(5)
with cols[0]:
    st.markdown(kpi_card("Total Applications", f"{total_applications:,}"), unsafe_allow_html=True)
with cols[1]:
    st.markdown(kpi_card("Avg. Processing Time", f"{avg_processing:.0f}h", f"{avg_processing / 24:.1f} days"), unsafe_allow_html=True)
with cols[2]:
    st.markdown(kpi_card("P90 Processing Time", f"{p90_processing:.0f}h", f"{p90_processing / 24:.1f} days"), unsafe_allow_html=True)
with cols[3]:
    st.markdown(kpi_card("SLA Breach Rate", f"{sla_breach_rate:.1%}"), unsafe_allow_html=True)
with cols[4]:
    st.markdown(kpi_card("Rework Rate", f"{rework_rate:.1%}", "of applications"), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Trends (weekly aggregation - daily counts are too noisy to read as a trend)
# ---------------------------------------------------------------------------

st.markdown("### Trends")
st.markdown(
    '<p class="section-note">Aggregated weekly for readability — the underlying data is daily.</p>',
    unsafe_allow_html=True,
)

weekly = pi.set_index("application_date").resample("W").agg(
    applications=("process_id", "count"),
    avg_processing_time_hours=("total_processing_time_hours", "mean"),
    sla_breach_rate=("sla_breached", "mean"),
)

tab1, tab2, tab3 = st.tabs(["Applications", "Processing Time", "SLA Breach Rate"])

with tab1:
    fig = px.line(weekly, y="applications", title="Applications Submitted per Week")
    fig.update_traces(line_color="#2a78d6")
    fig.update_layout(yaxis_title="applications", xaxis_title=None)
    st.plotly_chart(fig, width="stretch", theme=None)

with tab2:
    fig = px.line(weekly, y="avg_processing_time_hours", title="Average Processing Time per Week (hours)")
    fig.update_traces(line_color="#2a78d6")
    fig.update_layout(yaxis_title="hours", xaxis_title=None)
    st.plotly_chart(fig, width="stretch", theme=None)

with tab3:
    fig = px.line(weekly, y="sla_breach_rate", title="SLA Breach Rate per Week")
    fig.update_traces(line_color=STATUS["critical"])
    fig.update_layout(yaxis_title="breach rate", xaxis_title=None, yaxis_tickformat=".0%")
    st.plotly_chart(fig, width="stretch", theme=None)

st.markdown("---")
st.markdown(
    '<p class="section-note">Use the pages in the sidebar to drill into process flow and bottlenecks, '
    "assess an individual application's delay risk, or compare current vs. optimized staffing.</p>",
    unsafe_allow_html=True,
)
