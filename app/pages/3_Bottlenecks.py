"""Page 3 - Bottlenecks: the ranked bottleneck table and its components."""

from __future__ import annotations

import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_APP_DIR.parent))  # project root, for `src.*`
sys.path.insert(0, str(_APP_DIR))  # app dir, for `components.*`

import pandas as pd
import plotly.express as px
import streamlit as st

from components.data import get_bottleneck_table
from components.style import CATEGORICAL, inject_css, register_plotly_template
from src.data.generate_data import STAGE_ORDER
from src.process_mining.bottleneck import DEFAULT_WEIGHTS

st.set_page_config(page_title="Bottlenecks — Where do We Lose?", page_icon="🚦", layout="wide")
register_plotly_template()
inject_css()

st.title("Bottleneck Ranking")
st.markdown(
    '<p class="section-note">Every process stage, ranked by a single bottleneck score combining processing time, '
    "waiting time, volume, rework rate and resource utilization. See "
    "<code>src/process_mining/bottleneck.py</code> for the exact, documented formula.</p>",
    unsafe_allow_html=True,
)

table = get_bottleneck_table()

st.markdown("### Ranked Bottleneck Table")
display_cols = {
    "rank": "Rank", "stage": "Stage", "bottleneck_score": "Score", "volume": "Volume",
    "avg_processing_time": "Avg Processing (h)", "avg_waiting_time": "Avg Waiting (h)",
    "rework_rate": "Rework Rate", "resource_utilization": "Utilization",
}
display_df = table[list(display_cols.keys())].rename(columns=display_cols)
st.dataframe(
    display_df.style.format(
        {
            "Score": "{:.3f}", "Volume": "{:,}", "Avg Processing (h)": "{:.1f}",
            "Avg Waiting (h)": "{:.1f}", "Rework Rate": "{:.1%}", "Utilization": "{:.1%}",
        }
    ).background_gradient(subset=["Score"], cmap="Blues"),
    width="stretch", hide_index=True,
)

top3 = table.head(3)
st.markdown(
    f'<p class="section-note"><b>Top 3 bottlenecks:</b> {top3.iloc[0]["stage"]}, {top3.iloc[1]["stage"]}, '
    f'and {top3.iloc[2]["stage"]} — see the Process Analysis page for why, stage by stage.</p>',
    unsafe_allow_html=True,
)

st.markdown("---")
tab1, tab2, tab3, tab4 = st.tabs(["Bottleneck Score", "Stage Duration", "Waiting Time", "Resource Utilization"])

with tab1:
    fig = px.bar(
        table.sort_values("bottleneck_score"), x="bottleneck_score", y="stage", orientation="h",
        title="Bottleneck Score by Stage", color_discrete_sequence=[CATEGORICAL[0]], text_auto=".2f",
    )
    fig.update_layout(yaxis_title=None, xaxis_title="score (0-1)")
    st.plotly_chart(fig, width="stretch", theme=None)

    st.markdown("**Score weights** (see `bottleneck.py` for the rationale behind a weighted sum):")
    weight_df = pd.DataFrame({"component": list(DEFAULT_WEIGHTS.keys()), "weight": list(DEFAULT_WEIGHTS.values())})
    st.dataframe(weight_df.style.format({"weight": "{:.0%}"}), width="stretch", hide_index=True)

with tab2:
    fig = px.bar(
        table.sort_values("avg_processing_time"), x="avg_processing_time", y="stage", orientation="h",
        title="Average Processing Time by Stage (hours)", color_discrete_sequence=[CATEGORICAL[0]],
    )
    fig.update_layout(yaxis_title=None, xaxis_title="hours")
    st.plotly_chart(fig, width="stretch", theme=None)

with tab3:
    fig = px.bar(
        table.sort_values("avg_waiting_time"), x="avg_waiting_time", y="stage", orientation="h",
        title="Average Waiting Time by Stage (hours, estimated)", color_discrete_sequence=[CATEGORICAL[1]],
    )
    fig.update_layout(yaxis_title=None, xaxis_title="hours")
    st.plotly_chart(fig, width="stretch", theme=None)

with tab4:
    fig = px.bar(
        table.sort_values("resource_utilization"), x="resource_utilization", y="stage", orientation="h",
        title="Resource Utilization by Stage (share of each department's own peak throughput)",
        color_discrete_sequence=[CATEGORICAL[2]],
    )
    fig.update_layout(yaxis_title=None, xaxis_title="utilization", xaxis_tickformat=".0%")
    st.plotly_chart(fig, width="stretch", theme=None)
    st.markdown(
        '<p class="section-note">Utilization here is relative to each department\'s own best-demonstrated day, '
        "not an absolute hours-worked ratio — see the notebook 03 note on why that method was chosen.</p>",
        unsafe_allow_html=True,
    )
