"""Page 5 - Optimization: current vs. optimized staffing and the resulting
scenario comparison, both precomputed by
`python -m src.optimization.resource_allocation` (see reports/).
"""

from __future__ import annotations

import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_APP_DIR.parent))  # project root, for `src.*`
sys.path.insert(0, str(_APP_DIR))  # app dir, for `components.*`

import plotly.express as px
import streamlit as st

from components.data import get_optimized_staffing, get_scenario_comparison
from components.style import CATEGORICAL, STATUS, inject_css, kpi_card, register_plotly_template

st.set_page_config(page_title="Optimization — Where do We Lose?", page_icon="🧮", layout="wide")
register_plotly_template()
inject_css()

st.title("Resource Allocation Optimization")
st.markdown(
    '<p class="section-note">How should staff be redistributed across process stages to reduce SLA breaches and '
    "cycle time? Compares the current allocation against two alternatives: adding headcount only to the single "
    "largest bottleneck, and an OR-Tools-optimized redistribution across all stages. Every number below comes "
    "from re-running the actual process simulation with each staffing plan — see "
    "<code>src/optimization/resource_allocation.py</code>.</p>",
    unsafe_allow_html=True,
)

comparison = get_scenario_comparison()
staffing = get_optimized_staffing()

if comparison is None or staffing is None:
    st.warning(
        "Scenario comparison not found. Run `python -m src.optimization.resource_allocation` "
        "(or `make optimize`) to generate `reports/scenario_comparison.csv` and `reports/optimized_staffing.csv`."
    )
    st.stop()

baseline = comparison.iloc[0]
optimized = comparison.iloc[-1]

st.markdown("### Baseline vs. Optimized")
cols = st.columns(5)
metrics = [
    ("SLA Breach Rate", "sla_breach_rate", "{:.1%}", True),
    ("Avg. Processing Time", "avg_processing_time_hours", "{:.0f}h", True),
    ("P90 Processing Time", "p90_processing_time_hours", "{:.0f}h", True),
    ("Resource Utilization", "avg_resource_utilization", "{:.1%}", False),
    ("Total Employees", "total_employees", "{:,.0f}", None),
]
for col, (label, key, fmt, lower_is_better) in zip(cols, metrics):
    before, after = baseline[key], optimized[key]
    delta_pct = (after - before) / before * 100 if before else 0
    if lower_is_better is None:
        delta_html = f'<div class="kpi-sub">{delta_pct:+.1f}% vs. baseline</div>'
    else:
        improved = (delta_pct < 0) if lower_is_better else (delta_pct > 0)
        css = "kpi-delta-good" if improved else "kpi-delta-bad"
        delta_html = f'<div class="{css}">{delta_pct:+.1f}% vs. baseline</div>'
    with col:
        st.markdown(
            f"""<div class="kpi-card"><div class="kpi-label">{label}</div>
            <div class="kpi-value">{fmt.format(after)}</div>{delta_html}</div>""",
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### All Scenarios")
display = comparison.rename(
    columns={
        "scenario": "Scenario", "total_employees": "Employees",
        "avg_processing_time_hours": "Avg Processing (h)", "p90_processing_time_hours": "P90 Processing (h)",
        "sla_breach_rate": "SLA Breach Rate", "completion_rate": "Completion Rate",
        "throughput_per_day": "Throughput/day", "avg_resource_utilization": "Utilization",
    }
)
st.dataframe(
    display.style.format(
        {
            "Avg Processing (h)": "{:.1f}", "P90 Processing (h)": "{:.1f}", "SLA Breach Rate": "{:.1%}",
            "Completion Rate": "{:.1%}", "Throughput/day": "{:.1f}", "Utilization": "{:.1%}", "Employees": "{:,.0f}",
        }
    ),
    width="stretch", hide_index=True,
)

st.markdown("---")
tab1, tab2 = st.tabs(["Estimated Improvement", "Staffing by Stage"])

with tab1:
    metric_choice = st.radio(
        "Metric", ["SLA breach rate", "Avg. processing time", "P90 processing time"], horizontal=True
    )
    metric_map = {
        "SLA breach rate": ("sla_breach_rate", ".1%"),
        "Avg. processing time": ("avg_processing_time_hours", ".0f"),
        "P90 processing time": ("p90_processing_time_hours", ".0f"),
    }
    key, fmt = metric_map[metric_choice]
    fig = px.bar(
        comparison, x="scenario", y=key, title=f"{metric_choice} by Scenario",
        color_discrete_sequence=[CATEGORICAL[0]], text_auto=fmt,
    )
    fig.update_layout(xaxis_title=None, yaxis_title=metric_choice)
    if key == "sla_breach_rate":
        fig.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(fig, width="stretch", theme=None)

with tab2:
    plot_df = staffing.melt(id_vars="stage", value_vars=["current", "scenario_a", "optimized"], var_name="plan", value_name="employees")
    plot_df["plan"] = plot_df["plan"].map({"current": "Current", "scenario_a": "A: Add to bottleneck", "optimized": "B: Optimized"})
    fig = px.bar(
        plot_df, x="stage", y="employees", color="plan", barmode="group",
        title="Staffing by Stage — Current vs. Scenario A vs. Optimized",
        color_discrete_sequence=[CATEGORICAL[6], CATEGORICAL[1], CATEGORICAL[0]],
    )
    fig.update_layout(xaxis_title=None, legend_title=None)
    st.plotly_chart(fig, width="stretch", theme=None)

    st.dataframe(
        staffing.rename(columns={"stage": "Stage", "current": "Current", "scenario_a": "Scenario A", "optimized": "Optimized"}),
        width="stretch", hide_index=True,
    )
