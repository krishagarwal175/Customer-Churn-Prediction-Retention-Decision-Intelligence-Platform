"""Chart rendering helpers that theme and display reporting figures."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.theme import palette, style_fig
from churn_platform.reporting import charts as reporting_charts


def show(fig: Any) -> None:
    """Theme and render a Plotly figure full width."""
    st.plotly_chart(style_fig(fig), width="stretch")


def churn_overview(churn_summary: dict[str, Any]) -> None:
    show(reporting_charts.churn_overview_chart(churn_summary, backend="plotly"))


def revenue_at_risk(revenue_summary: dict[str, Any]) -> None:
    show(reporting_charts.revenue_at_risk_chart(revenue_summary, backend="plotly"))


def roc(y_true: Any, y_score: Any) -> None:
    show(reporting_charts.roc_curve_chart(y_true, y_score, backend="plotly"))


def precision_recall(y_true: Any, y_score: Any) -> None:
    show(reporting_charts.precision_recall_chart(y_true, y_score, backend="plotly"))


def calibration(y_true: Any, y_score: Any) -> None:
    show(reporting_charts.calibration_chart(y_true, y_score, backend="plotly"))


def feature_importance(importance: pd.DataFrame, top_n: int = 12) -> None:
    fig = reporting_charts.feature_importance_chart(
        importance, top_n=top_n, backend="plotly"
    )
    # Horizontal reads better for feature names.
    top = importance.head(top_n).iloc[::-1]
    fig = go.Figure(go.Bar(x=top["mean_abs_shap"], y=top["feature"], orientation="h"))
    fig.update_layout(title="Global churn drivers (mean |SHAP|)")
    show(fig)


def bar(
    labels: list[str],
    values: list[float],
    title: str,
    ylabel: str = "",
    color_key: str = "accent",
) -> None:
    """Render a simple themed bar chart."""
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=palette()[color_key]))
    fig.update_layout(title=title, yaxis_title=ylabel)
    show(fig)


def probability_histogram(probabilities: pd.Series) -> None:
    """Render a churn-probability distribution."""
    fig = go.Figure(
        go.Histogram(x=probabilities, nbinsx=30, marker_color=palette()["accent"])
    )
    fig.update_layout(
        title="Churn probability distribution",
        xaxis_title="Predicted churn probability",
        yaxis_title="Customers",
    )
    show(fig)
