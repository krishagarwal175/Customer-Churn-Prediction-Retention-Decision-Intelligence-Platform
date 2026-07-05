"""KPI card rendering for the dashboard."""

from __future__ import annotations

from collections.abc import Sequence

import streamlit as st

from app.theme import palette


def kpi_card(label: str, value: str, note: str | None = None) -> str:
    """Return HTML for a single KPI card."""
    note_html = f'<div class="kpi-note">{note}</div>' if note else ""
    return (
        f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>{note_html}</div>'
    )


def kpi_row(cards: Sequence[tuple[str, str, str | None]]) -> None:
    """Render a responsive row of KPI cards.

    Args:
        cards: Sequence of ``(label, value, note)`` tuples.
    """
    columns = st.columns(len(cards))
    for column, (label, value, note) in zip(columns, cards):
        with column:
            st.markdown(kpi_card(label, value, note), unsafe_allow_html=True)


def risk_badge(probability: float) -> str:
    """Return an HTML pill describing a churn-risk band."""
    p = palette()
    if probability >= 0.66:
        color, text = p["danger"], "High risk"
    elif probability >= 0.33:
        color, text = p["warning"], "Medium risk"
    else:
        color, text = p["success"], "Low risk"
    return (
        f'<span class="pill" style="border-color:{color};color:{color}">'
        f"{text} · {probability:.0%}</span>"
    )
