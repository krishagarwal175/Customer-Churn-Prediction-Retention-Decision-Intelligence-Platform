"""Executive dashboard page."""

from __future__ import annotations

import streamlit as st

from app import services
from app.components import charts, tables
from app.components.filters import header
from app.components.kpis import kpi_row


def render() -> None:
    """Render the executive overview."""
    header(
        "Executive dashboard",
        "Churn, revenue at risk, and the retention opportunity at a glance.",
        eyebrow="Overview",
    )

    eda = services.get_business_eda()
    churn = eda.churn_summary()
    revenue = eda.revenue_summary()
    scored = services.get_scored_customers()

    kpi_row(
        [
            ("Customers", f"{churn['total_customers']:,}", None),
            ("Churn rate", f"{churn['churn_rate']:.1%}", "historical"),
            (
                "Revenue at risk",
                tables.money(revenue["churned_customer_revenue"]),
                "from churned customers",
            ),
            (
                "High-risk now",
                f"{int((scored['churn_probability'] >= 0.5).sum()):,}",
                "predicted p ≥ 0.5",
            ),
        ]
    )

    st.markdown('<hr class="divider" />', unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        charts.churn_overview(churn)
    with right:
        charts.revenue_at_risk(revenue)

    st.markdown(
        '<div class="section-title">Top retention opportunities</div>',
        unsafe_allow_html=True,
    )
    simulated = services.simulate(scored, default_uplift=0.30, default_cost=60.0)
    shortlist = services.recommendation_shortlist(simulated, top_n=10)
    tables.dataframe(shortlist)
