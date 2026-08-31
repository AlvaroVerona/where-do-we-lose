"""Page 2 - Process Analysis: flow, activity frequency, duration, waiting
time and rework, filterable by date/region/channel/type/segment.
"""

from __future__ import annotations

import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_APP_DIR.parent))  # project root, for `src.*`
sys.path.insert(0, str(_APP_DIR))  # app dir, for `components.*`

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from components.data import apply_filters, get_event_log, get_process_instances
from components.style import CATEGORICAL, STATUS, inject_css, register_plotly_template
from src.data.generate_data import STAGE_ORDER
from src.process_mining.process_metrics import (
    compute_activity_frequency,
    compute_rework_rate_by_stage,
    compute_stage_durations,
    compute_transitions,
    estimate_processing_and_waiting_time,
)

st.set_page_config(page_title="Process Analysis — Where do We Lose?", page_icon="🔀", layout="wide")
register_plotly_template()
inject_css()

st.title("Process Analysis")
st.markdown(
    '<p class="section-note">Process flow, activity frequency, stage duration, waiting time and rework — '
    "filter below to see how any segment of the business moves through the process.</p>",
    unsafe_allow_html=True,
)

pi = get_process_instances()
el = get_event_log()

# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Filters")
    min_date, max_date = pi["application_date"].min().date(), pi["application_date"].max().date()
    date_range = st.date_input("Application date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    region = st.multiselect("Region", sorted(pi["region"].unique()))
    channel = st.multiselect("Channel", sorted(pi["channel"].unique()))
    application_type = st.multiselect("Application type", sorted(pi["application_type"].unique()))
    customer_segment = st.multiselect("Customer segment", sorted(pi["customer_segment"].unique()))

filtered_pi = apply_filters(pi, date_range, region, channel, application_type, customer_segment)
filtered_el = el[el["process_id"].isin(filtered_pi["process_id"])]

if len(filtered_pi) == 0:
    st.warning("No applications match the current filters.")
    st.stop()

st.markdown(f'<p class="section-note">{len(filtered_pi):,} of {len(pi):,} applications match the current filters.</p>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Process flow (Sankey)
# ---------------------------------------------------------------------------

st.markdown("### Process Flow")

node_list = ["Application Submitted", *STAGE_ORDER, "Completed", "Rejected"]
node_index = {name: i for i, name in enumerate(node_list)}
node_colors = (
    [CATEGORICAL[0]] * (len(STAGE_ORDER) + 1) + [STATUS["good"], STATUS["critical"]]
)

transitions = compute_transitions(filtered_el)
main_sequence = ["Application Submitted", *STAGE_ORDER, "Completed"]

sources, targets, values, link_labels = [], [], [], []
for a, b in zip(main_sequence[:-1], main_sequence[1:]):
    row = transitions[(transitions["from_activity"] == a) & (transitions["to_activity"] == b)]
    if len(row):
        sources.append(node_index[a])
        targets.append(node_index[b])
        values.append(int(row["count"].iloc[0]))
        link_labels.append(f"{int(row['count'].iloc[0]):,} applications, {row['avg_hours'].iloc[0]:.0f}h avg")

rejection_counts = filtered_pi.loc[filtered_pi["final_status"] == "Rejected", "rejection_stage"].value_counts()
for stage, count in rejection_counts.items():
    if stage in node_index:
        sources.append(node_index[stage])
        targets.append(node_index["Rejected"])
        values.append(int(count))
        link_labels.append(f"{int(count):,} rejected")

fig = go.Figure(
    go.Sankey(
        node=dict(label=node_list, color=node_colors, pad=20, thickness=18, line=dict(color="white", width=1)),
        link=dict(source=sources, target=targets, value=values, label=link_labels, color="rgba(42,120,214,0.25)"),
    )
)
fig.update_layout(title="Application Flow (link width = volume)", height=420, font_size=12)
st.plotly_chart(fig, width="stretch", theme=None)
st.markdown(
    '<p class="section-note">Green = completed successfully, red = rejected. Rework loops are excluded here '
    "for readability — see the Rework section below.</p>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Activity frequency / stage duration / waiting time / rework
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs(["Activity Frequency", "Stage Duration", "Waiting Time", "Rework"])

with tab1:
    freq = compute_activity_frequency(filtered_el)
    fig = px.bar(
        freq.sort_values("count"), x="count", y="activity", orientation="h",
        title="Activity Frequency (event occurrences)", color_discrete_sequence=[CATEGORICAL[0]],
    )
    fig.update_layout(yaxis_title=None)
    st.plotly_chart(fig, width="stretch", theme=None)

with tab2:
    stage_durations = compute_stage_durations(filtered_el)
    fig = px.box(
        stage_durations, x="activity", y="gap_hours", category_orders={"activity": STAGE_ORDER},
        title="Stage Duration Distribution (hours, wait + processing combined)",
        color_discrete_sequence=[CATEGORICAL[0]],
    )
    fig.update_layout(xaxis_title=None, yaxis_title="hours")
    st.plotly_chart(fig, width="stretch", theme=None)

with tab3:
    estimated = estimate_processing_and_waiting_time(stage_durations)
    by_stage = estimated.groupby("activity")[["estimated_processing_hours", "estimated_waiting_hours"]].mean().reindex(STAGE_ORDER)
    plot_df = by_stage.reset_index().melt(id_vars="activity", var_name="component", value_name="hours")
    plot_df["component"] = plot_df["component"].map(
        {"estimated_processing_hours": "Processing (estimated)", "estimated_waiting_hours": "Waiting (estimated)"}
    )
    fig = px.bar(
        plot_df, x="activity", y="hours", color="component", barmode="stack",
        title="Estimated Waiting vs. Processing Time by Stage",
        category_orders={"activity": STAGE_ORDER}, color_discrete_sequence=[CATEGORICAL[0], CATEGORICAL[1]],
    )
    fig.update_layout(xaxis_title=None, legend_title=None)
    st.plotly_chart(fig, width="stretch", theme=None)
    st.markdown(
        '<p class="section-note">The event log records one timestamp per activity, so the processing/waiting '
        "split is an estimate (see notebook 02) — the combined stage duration above is exact.</p>",
        unsafe_allow_html=True,
    )

with tab4:
    rework = compute_rework_rate_by_stage(filtered_el)
    fig = px.bar(
        rework, x="stage", y="rework_rate", title="Rework Rate by Stage",
        category_orders={"stage": STAGE_ORDER}, color_discrete_sequence=[CATEGORICAL[1]],
    )
    fig.update_layout(xaxis_title=None, yaxis_tickformat=".0%", yaxis_title="rework rate")
    st.plotly_chart(fig, width="stretch", theme=None)
