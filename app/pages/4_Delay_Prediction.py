"""Page 4 - Delay Prediction: score a hypothetical application and explain
why, using the trained model and SHAP explainer from `src/models/`.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_APP_DIR.parent))  # project root, for `src.*`
sys.path.insert(0, str(_APP_DIR))  # app dir, for `components.*`

import pandas as pd
import streamlit as st

from components.data import get_explainer, get_model, get_process_instances
from components.style import inject_css, register_plotly_template, risk_badge_html
from src.features.engineering import CATEGORICAL_FEATURES, FEATURE_COLUMNS
from src.models.explain import explain_single_prediction
from src.models.predict import risk_category

st.set_page_config(page_title="Delay Prediction — Where do We Lose?", page_icon="🔮", layout="wide")
register_plotly_template()
inject_css()

st.title("Delay (SLA-Breach) Prediction")
st.markdown(
    '<p class="section-note">Enter a hypothetical application as it would look the moment it is submitted — '
    "before any stage has processed it — and get its predicted SLA-breach probability, risk category, and the "
    "main factors driving that prediction.</p>",
    unsafe_allow_html=True,
)

pi = get_process_instances()
model = get_model()
explainer = get_explainer()

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        customer_segment = st.selectbox("Customer Segment", sorted(pi["customer_segment"].unique()))
        application_type = st.selectbox("Application Type", sorted(pi["application_type"].unique()))
        loan_amount = st.number_input("Loan Amount ($)", min_value=1000, max_value=2_000_000, value=25000, step=1000)
    with col2:
        risk_score = st.slider("Risk Score", 0, 100, 50)
        complexity_score = st.slider("Complexity Score", 0, 100, 40)
        priority = st.selectbox("Priority", sorted(pi["priority"].unique()))
    with col3:
        channel = st.selectbox("Channel", sorted(pi["channel"].unique()))
        region = st.selectbox("Region", sorted(pi["region"].unique()))
        current_queue_size = st.number_input(
            "Current Queue (applications in the last 24h)", min_value=0, max_value=300,
            value=int(pi["application_date"].dt.date.value_counts().median()),
        )
    submitted = st.form_submit_button("Predict SLA Breach Risk", type="primary", width="stretch")

if submitted:
    now = datetime.now()
    application = pd.DataFrame(
        [
            {
                "customer_segment": customer_segment,
                "application_type": application_type,
                "loan_amount": loan_amount,
                "risk_score": risk_score,
                "complexity_score": complexity_score,
                "channel": channel,
                "region": region,
                "priority": priority,
                "day_of_week": now.weekday(),
                "hour_of_day": now.hour,
                "current_queue_size": current_queue_size,
            }
        ]
    )

    result = explain_single_prediction(model, explainer, application, top_n=5)
    probability = result["probability"]
    category = risk_category(probability)

    st.markdown("---")
    result_col, factor_col = st.columns([1, 1.4])

    with result_col:
        st.markdown("### Result")
        st.markdown(
            f"""
            <div class="kpi-card" style="text-align:center; padding: 28px 18px;">
                <div class="kpi-label">SLA Breach Probability</div>
                <div style="font-size:3rem; font-weight:800; color:#0b0b0b; line-height:1.1;">{probability:.0%}</div>
                <div style="margin-top:10px;">{risk_badge_html(category)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with factor_col:
        st.markdown("### Main Factors")
        for f in result["top_factors"]:
            sign = "➕" if f["direction"] == "increases" else "➖"
            verb = "increases" if f["direction"] == "increases" else "decreases"
            st.markdown(f"{sign} **{f['label']}** — {verb} breach risk")
        st.markdown(
            '<p class="section-note" style="margin-top:12px;">Factors are ranked by SHAP contribution — '
            "see the Delay Prediction notebook (04) for the full methodology.</p>",
            unsafe_allow_html=True,
        )
else:
    st.info("Fill in the application details above and click **Predict SLA Breach Risk**.")
