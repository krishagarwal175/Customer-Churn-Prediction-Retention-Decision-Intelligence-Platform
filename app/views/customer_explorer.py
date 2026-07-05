"""Customer explorer page."""

from __future__ import annotations

import streamlit as st

from app import services
from app.components import charts, tables
from app.components.filters import customer_filters, header


def render() -> None:
    """Render a filterable customer table."""
    header(
        "Customer explorer",
        "Filter and inspect scored customers by persona and churn risk.",
        eyebrow="Customers",
    )

    scored = services.get_scored_customers()
    filtered = customer_filters(scored)

    st.markdown(
        f'<div class="app-sub">{len(filtered):,} of {len(scored):,} customers '
        "match the current filters.</div>",
        unsafe_allow_html=True,
    )
    charts.probability_histogram(filtered["churn_probability"])

    display = tables.format_customer_table(
        filtered.sort_values("churn_probability", ascending=False)
    )
    tables.dataframe(display, height=460)

    st.download_button(
        "Download filtered customers (CSV)",
        display.to_csv(index=False).encode("utf-8"),
        file_name="filtered_customers.csv",
        mime="text/csv",
    )
