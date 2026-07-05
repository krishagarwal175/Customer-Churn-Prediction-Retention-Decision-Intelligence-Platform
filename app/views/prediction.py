"""Single-customer prediction page."""

from __future__ import annotations

import streamlit as st

from app import services
from app.components.customer_profile import render_profile
from app.components.filters import header


def render() -> None:
    """Render a per-customer churn prediction and explanation."""
    header(
        "Prediction",
        "Look up a customer to see their churn risk and the reasons behind it.",
        eyebrow="Predict",
    )

    scored = services.get_scored_customers()
    bundle = services.get_model_bundle()

    ranked = scored.sort_values("churn_probability", ascending=False)
    options = ranked["CustomerID"].tolist()
    customer_id = st.selectbox(
        "Customer", options, help="Sorted by predicted churn risk."
    )

    row = scored[scored["CustomerID"] == customer_id].iloc[0]
    feature_columns = list(bundle["X"].columns)
    explanation = bundle["explainer"].explain_customer(
        bundle["X"].loc[[row.name], feature_columns]
    )
    render_profile(row, explanation)
