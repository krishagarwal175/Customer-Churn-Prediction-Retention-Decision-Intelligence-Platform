"""Streamlit application entry point.

Wires the churn platform's analytics layers into a themed, multi-page
decision-intelligence dashboard.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Ensure the project root is importable when run via `streamlit run`.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.views import (  # noqa: E402
    about,
    business_insights,
    customer_explorer,
    executive_dashboard,
    explainability,
    prediction,
    revenue_simulator,
    segmentation,
)
from app.theme import inject_css, mode_toggle  # noqa: E402

st.set_page_config(
    page_title="Churn Decision Intelligence",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.session_state.setdefault("theme_mode", "light")
inject_css()

with st.sidebar:
    st.markdown("### Churn Intelligence")
    st.caption("Retention decision platform")
    st.markdown("---")
    mode_toggle()

PAGES = [
    st.Page(
        executive_dashboard.render,
        title="Executive dashboard",
        icon=":material/dashboard:",
        url_path="executive",
        default=True,
    ),
    st.Page(
        customer_explorer.render,
        title="Customer explorer",
        icon=":material/table:",
        url_path="customers",
    ),
    st.Page(
        segmentation.render,
        title="Segmentation",
        icon=":material/group:",
        url_path="segmentation",
    ),
    st.Page(
        prediction.render,
        title="Prediction",
        icon=":material/person_search:",
        url_path="prediction",
    ),
    st.Page(
        explainability.render,
        title="Explainability",
        icon=":material/insights:",
        url_path="explainability",
    ),
    st.Page(
        revenue_simulator.render,
        title="Revenue simulator",
        icon=":material/payments:",
        url_path="simulator",
    ),
    st.Page(
        business_insights.render,
        title="Business insights",
        icon=":material/query_stats:",
        url_path="insights",
    ),
    st.Page(about.render, title="About", icon=":material/info:", url_path="about"),
]

st.navigation(PAGES).run()
