"""Table rendering helpers for the dashboard."""

from __future__ import annotations

import pandas as pd
import streamlit as st


def dataframe(df: pd.DataFrame, height: int | None = None) -> None:
    """Render a DataFrame with sensible defaults."""
    kwargs: dict[str, object] = {"width": "stretch", "hide_index": True}
    if height is not None:
        kwargs["height"] = height
    st.dataframe(df, **kwargs)


def money(value: float) -> str:
    """Format a value as compact currency."""
    return f"${value:,.0f}"


def format_customer_table(scored: pd.DataFrame) -> pd.DataFrame:
    """Select and rename customer columns for display."""
    columns = {
        "CustomerID": "Customer",
        "persona": "Persona",
        "segment": "Segment",
        "churn_probability": "Churn probability",
        "Tenure Months": "Tenure (mo)",
        "Monthly Charges": "Monthly charge",
        "CLTV": "CLTV",
        "Contract": "Contract",
        "Churn Label": "Churned",
    }
    present = {k: v for k, v in columns.items() if k in scored.columns}
    return scored[list(present)].rename(columns=present)
