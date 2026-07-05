"""Reusable filter controls for the dashboard."""

from __future__ import annotations

import pandas as pd
import streamlit as st


def customer_filters(scored: pd.DataFrame) -> pd.DataFrame:
    """Render persona and risk filters and return the filtered frame.

    Args:
        scored: Scored customer frame with ``persona`` and
            ``churn_probability`` columns.

    Returns:
        The filtered customer frame.
    """
    left, right = st.columns([2, 1])
    with left:
        personas = sorted(scored["persona"].dropna().unique().tolist())
        chosen = st.multiselect("Persona", personas, default=personas)
    with right:
        min_risk = st.slider("Minimum churn risk", 0.0, 1.0, 0.0, 0.05)

    filtered = scored[
        scored["persona"].isin(chosen) & (scored["churn_probability"] >= min_risk)
    ]
    return filtered


def header(title: str, subtitle: str, eyebrow: str = "") -> None:
    """Render a consistent page header."""
    if eyebrow:
        st.markdown(f'<div class="app-eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.markdown(f"# {title}")
    st.markdown(f'<div class="app-sub">{subtitle}</div>', unsafe_allow_html=True)
    st.markdown('<hr class="divider" />', unsafe_allow_html=True)
