"""Shared visual language for every dashboard page: a validated,
colorblind-safe categorical palette, a status palette for risk levels, and a
Plotly template built from the same chart chrome - so every chart on every
page reads as one system rather than five separately-styled pages.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# Fixed-order categorical palette (validated for colorblind-safe adjacent
# contrast) - always assign by this order, never reassign per filter.
CATEGORICAL = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]

# Single-hue sequential ramp (blue), for magnitude encodings.
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

# Status palette - reserved for state, never reused as a categorical series.
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}

# Chart chrome
SURFACE = "#fcfcfb"
PAGE_PLANE = "#f9f9f7"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

FONT_FAMILY = "system-ui, -apple-system, 'Segoe UI', sans-serif"

TEMPLATE_NAME = "wdwl"


def register_plotly_template() -> None:
    """Build and register the shared Plotly template once per session."""
    if TEMPLATE_NAME in pio.templates:
        pio.templates.default = TEMPLATE_NAME
        return

    template = go.layout.Template()
    template.layout = go.Layout(
        colorway=CATEGORICAL,
        font=dict(family=FONT_FAMILY, color=INK_PRIMARY, size=13),
        title=dict(font=dict(size=15, color=INK_PRIMARY)),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        xaxis=dict(gridcolor=GRIDLINE, linecolor=BASELINE, tickfont=dict(color=INK_MUTED), zerolinecolor=BASELINE),
        yaxis=dict(gridcolor=GRIDLINE, linecolor=BASELINE, tickfont=dict(color=INK_MUTED), zerolinecolor=BASELINE),
        legend=dict(font=dict(color=INK_SECONDARY)),
        margin=dict(l=70, r=30, t=50, b=50),
    )
    pio.templates[TEMPLATE_NAME] = template
    pio.templates.default = TEMPLATE_NAME


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{ background-color: {PAGE_PLANE}; }}
        [data-testid="stSidebar"] {{ background-color: {SURFACE}; }}

        .kpi-card {{
            background-color: {SURFACE};
            border: 1px solid {GRIDLINE};
            border-radius: 10px;
            padding: 16px 18px;
            height: 100%;
        }}
        .kpi-label {{
            font-size: 0.80rem;
            color: {INK_SECONDARY};
            font-weight: 500;
            margin-bottom: 4px;
        }}
        .kpi-value {{
            font-size: 1.65rem;
            color: {INK_PRIMARY};
            font-weight: 700;
            line-height: 1.15;
        }}
        .kpi-delta-good {{ color: {STATUS['good']}; font-size: 0.85rem; font-weight: 600; }}
        .kpi-delta-bad {{ color: {STATUS['critical']}; font-size: 0.85rem; font-weight: 600; }}
        .kpi-sub {{ color: {INK_MUTED}; font-size: 0.78rem; margin-top: 2px; }}

        .status-badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.95rem;
            letter-spacing: 0.02em;
        }}
        .status-badge-good {{ background-color: rgba(12,163,12,0.12); color: {STATUS['good']}; }}
        .status-badge-warning {{ background-color: rgba(250,178,25,0.18); color: #8a5a00; }}
        .status-badge-critical {{ background-color: rgba(208,59,59,0.12); color: {STATUS['critical']}; }}

        .section-note {{ color: {INK_SECONDARY}; font-size: 0.92rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, sub: str | None = None) -> str:
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {sub_html}
    </div>
    """


def risk_badge_html(risk_category: str) -> str:
    mapping = {
        "LOW RISK": ("status-badge-good", "🟢"),
        "MODERATE RISK": ("status-badge-warning", "🟡"),
        "HIGH RISK": ("status-badge-critical", "🔴"),
    }
    css_class, icon = mapping.get(risk_category, ("status-badge-warning", "⚪"))
    return f'<span class="status-badge {css_class}">{icon} {risk_category}</span>'
