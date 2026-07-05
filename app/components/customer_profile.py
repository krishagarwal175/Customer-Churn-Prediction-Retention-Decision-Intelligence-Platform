"""Single-customer profile rendering."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.components.kpis import risk_badge
from churn_platform.explainability.business_translator import BusinessTranslator


def render_profile(
    customer: pd.Series,
    explanation: pd.DataFrame,
) -> None:
    """Render a customer's key attributes, risk, and churn drivers.

    Args:
        customer: A row from the scored customer frame.
        explanation: Per-customer SHAP explanation for the same customer.
    """
    probability = float(customer.get("churn_probability", 0.0))
    st.markdown(risk_badge(probability), unsafe_allow_html=True)

    attributes = {
        "Persona": customer.get("persona", "—"),
        "Tenure (months)": customer.get("Tenure Months", "—"),
        "Monthly charge": customer.get("Monthly Charges", "—"),
        "Contract": customer.get("Contract", "—"),
        "CLTV": customer.get("CLTV", "—"),
    }
    columns = st.columns(len(attributes))
    for column, (label, value) in zip(columns, attributes.items()):
        display = f"{value:,.0f}" if isinstance(value, (int, float)) else str(value)
        column.metric(label, display)

    st.markdown(
        '<div class="section-title">Why this customer may churn</div>',
        unsafe_allow_html=True,
    )
    drivers = BusinessTranslator().translate(explanation, top_n=5)
    for driver in drivers:
        arrow = "▲" if "increases" in str(driver["direction"]) else "▼"
        st.markdown(f"- {arrow} **{driver['readable']}** — {driver['direction']}")
