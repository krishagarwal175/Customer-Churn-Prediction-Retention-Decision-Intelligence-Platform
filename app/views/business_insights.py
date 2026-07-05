"""Business insights (EDA) page."""

from __future__ import annotations

import streamlit as st

from app import services
from app.components import charts, tables
from app.components.filters import header


def render() -> None:
    """Render contract, tenure, and service churn insights."""
    header(
        "Business insights",
        "How churn varies across contracts, tenure, and services.",
        eyebrow="Insights",
    )

    eda = services.get_business_eda()

    st.markdown(
        '<div class="section-title">Churn by contract</div>', unsafe_allow_html=True
    )
    contract = eda.contract_analysis().reset_index()
    charts.bar(
        contract["Contract"].astype(str).tolist(),
        (contract["churn_rate"] * 100).round(1).tolist(),
        "Churn rate by contract (%)",
        ylabel="Churn rate %",
    )
    tables.dataframe(contract.round(2))

    st.markdown(
        '<div class="section-title">Churn by tenure</div>', unsafe_allow_html=True
    )
    tenure = eda.tenure_analysis()
    bucket = tenure["churn_rate_by_bucket"].reset_index()
    charts.bar(
        bucket["tenure_bucket"].astype(str).tolist(),
        (bucket["churn_rate"] * 100).round(1).tolist(),
        "Churn rate by tenure bucket (%)",
        ylabel="Churn rate %",
        color_key="warning",
    )

    st.markdown(
        '<div class="section-title">Churn by service</div>', unsafe_allow_html=True
    )
    services_analysis = eda.service_analysis()
    options = list(services_analysis)
    if options:
        choice = st.selectbox("Service", options)
        frame = services_analysis[choice].reset_index()
        charts.bar(
            frame[choice].astype(str).tolist(),
            (frame["churn_rate"] * 100).round(1).tolist(),
            f"Churn rate by {choice} (%)",
            ylabel="Churn rate %",
        )
