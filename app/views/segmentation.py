"""Customer segmentation page."""

from __future__ import annotations

import streamlit as st

from app import services
from app.components import charts, tables
from app.components.filters import header


def render() -> None:
    """Render segment profiles and personas."""
    header(
        "Segmentation",
        "Behavioural customer segments and their retention personas.",
        eyebrow="Segments",
    )

    segmentation = services.get_segmentation()
    profile = segmentation["profile"].reset_index()

    left, right = st.columns(2)
    with left:
        charts.bar(
            profile["persona"].tolist(),
            profile["customer_count"].tolist(),
            "Customers per segment",
            ylabel="Customers",
        )
    with right:
        charts.bar(
            profile["persona"].tolist(),
            (profile["churn_rate"] * 100).round(1).tolist(),
            "Churn rate by segment (%)",
            ylabel="Churn rate %",
            color_key="warning",
        )

    st.markdown(
        '<div class="section-title">Segment profiles</div>', unsafe_allow_html=True
    )
    display = profile.copy()
    numeric = display.select_dtypes(include="number").columns
    display[numeric] = display[numeric].round(2)
    tables.dataframe(display)
