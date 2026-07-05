"""Retention revenue simulator page."""

from __future__ import annotations

import streamlit as st

from app import services
from app.components import tables
from app.components.filters import header
from app.components.kpis import kpi_row
from churn_platform.simulation.revenue import RevenueSimulator


def render() -> None:
    """Render interactive retention economics and sensitivity."""
    header(
        "Revenue simulator",
        "Model the ROI of a retention campaign under different assumptions.",
        eyebrow="Simulate",
    )

    scored = services.get_scored_customers()

    left, mid, right = st.columns(3)
    with left:
        uplift = st.slider(
            "Campaign uplift",
            0.05,
            0.60,
            0.30,
            0.05,
            help="Fraction of churn averted by the campaign.",
        )
    with mid:
        cost = st.slider("Cost per customer ($)", 10, 200, 60, 5)
    with right:
        top_n = st.slider("Outreach capacity", 50, 2000, 500, 50)

    simulated = services.simulate(
        scored, default_uplift=uplift, default_cost=float(cost)
    )
    summary = RevenueSimulator().campaign_summary(simulated)

    kpi_row(
        [
            ("Customers targeted", f"{summary['customers_targeted']:,}", None),
            ("Campaign cost", tables.money(summary["total_intervention_cost"]), None),
            (
                "Expected net benefit",
                tables.money(summary["total_expected_net_benefit"]),
                None,
            ),
            ("ROI", f"{summary['roi']:.1f}×", "net benefit / cost"),
        ]
    )

    st.markdown('<hr class="divider" />', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">Sensitivity: net benefit by assumptions</div>',
        unsafe_allow_html=True,
    )
    grid = services.sensitivity_grid(
        scored,
        uplifts=[round(uplift - 0.1, 2), uplift, round(uplift + 0.1, 2)],
        costs=[float(cost) - 20, float(cost), float(cost) + 20],
    )
    grid_display = grid[
        [
            "default_uplift",
            "default_cost",
            "customers_targeted",
            "total_expected_net_benefit",
            "roi",
        ]
    ].round(2)
    tables.dataframe(grid_display)

    st.markdown(
        '<div class="section-title">Prioritized shortlist</div>', unsafe_allow_html=True
    )
    shortlist = services.recommendation_shortlist(simulated, top_n=min(top_n, 25))
    tables.dataframe(shortlist)
