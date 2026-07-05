"""About page."""

from __future__ import annotations

import streamlit as st

from app.components.filters import header


def render() -> None:
    """Render project background and methodology."""
    header(
        "About",
        "Customer Churn Prediction & Retention Decision Intelligence Platform.",
        eyebrow="About",
    )

    st.markdown("""
        This platform turns raw telecom customer data into retention decisions.
        The analytics engine runs as a modular Python pipeline; this dashboard is
        the decision-intelligence layer on top of it.

        **Pipeline**

        - Ingestion, validation, and deterministic cleaning
        - Schema-driven business EDA and feature engineering
        - Calibrated churn model (XGBoost) with train / validation / test splits
        - SHAP explainability translated into plain-language drivers
        - K-Means segmentation with retention personas
        - Expected-value revenue simulation and grid sensitivity
        - Hybrid persona + driver recommendations, prioritized by net benefit

        **How to read the app**

        - Executive dashboard — headline KPIs and top opportunities
        - Customer explorer — filter and inspect scored customers
        - Segmentation — behavioural segments and personas
        - Prediction — per-customer risk and reasons
        - Explainability — global drivers and model performance
        - Revenue simulator — campaign ROI and sensitivity
        - Business insights — churn by contract, tenure, and service
        """)
