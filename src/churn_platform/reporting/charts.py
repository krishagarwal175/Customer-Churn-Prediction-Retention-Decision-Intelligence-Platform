"""Reusable reporting charts (Plotly and Matplotlib backends).

Every function returns a figure object and performs no file IO, so the caller
(the Streamlit app or a batch report script) decides how to render or persist
it. Each chart supports a ``backend`` of ``"plotly"`` (interactive, default) or
``"matplotlib"`` (static).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

LOGGER = logging.getLogger(__name__)

__all__ = [
    "churn_overview_chart",
    "revenue_at_risk_chart",
    "roc_curve_chart",
    "precision_recall_chart",
    "calibration_chart",
    "feature_importance_chart",
]

_BACKENDS = ("plotly", "matplotlib")


def _validate_backend(backend: str) -> None:
    if backend not in _BACKENDS:
        raise ValueError(f"Unsupported backend '{backend}'. Choose from {_BACKENDS}.")


def _bar(
    labels: list[str], values: list[float], title: str, ylabel: str, backend: str
) -> Any:
    """Return a bar chart in the requested backend."""
    _validate_backend(backend)
    if backend == "plotly":
        import plotly.graph_objects as go

        fig = go.Figure(go.Bar(x=labels, y=values))
        fig.update_layout(title=title, yaxis_title=ylabel)
        return fig

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    return fig


def _line(
    series: list[dict[str, Any]],
    title: str,
    xlabel: str,
    ylabel: str,
    backend: str,
    diagonal: bool = False,
) -> Any:
    """Return a multi-line chart; ``series`` is a list of {x, y, name} dicts."""
    _validate_backend(backend)
    if backend == "plotly":
        import plotly.graph_objects as go

        fig = go.Figure()
        for line in series:
            fig.add_trace(
                go.Scatter(x=line["x"], y=line["y"], mode="lines", name=line["name"])
            )
        if diagonal:
            fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    name="Chance",
                    line={"dash": "dash"},
                )
            )
        fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel)
        return fig

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    for line in series:
        ax.plot(line["x"], line["y"], label=line["name"])
    if diagonal:
        ax.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    return fig


def churn_overview_chart(churn_summary: dict[str, Any], backend: str = "plotly") -> Any:
    """Bar chart of churned vs active customers from a churn summary."""
    return _bar(
        ["Active", "Churned"],
        [churn_summary["active_customers"], churn_summary["churned_customers"]],
        "Customer Churn Overview",
        "Customers",
        backend,
    )


def revenue_at_risk_chart(
    revenue_summary: dict[str, Any], backend: str = "plotly"
) -> Any:
    """Bar chart of retained vs churned customer revenue."""
    return _bar(
        ["Retained", "Churned"],
        [
            revenue_summary["retained_customer_revenue"],
            revenue_summary["churned_customer_revenue"],
        ],
        "Revenue by Churn Status",
        "Total revenue",
        backend,
    )


def roc_curve_chart(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    backend: str = "plotly",
) -> Any:
    """ROC curve with AUC in the title."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    return _line(
        [{"x": fpr, "y": tpr, "name": "ROC"}],
        f"ROC Curve (AUC = {auc:.3f})",
        "False positive rate",
        "True positive rate",
        backend,
        diagonal=True,
    )


def precision_recall_chart(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    backend: str = "plotly",
) -> Any:
    """Precision-recall curve with average precision in the title."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    return _line(
        [{"x": recall, "y": precision, "name": "PR"}],
        f"Precision-Recall Curve (AP = {ap:.3f})",
        "Recall",
        "Precision",
        backend,
    )


def calibration_chart(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    n_bins: int = 10,
    backend: str = "plotly",
) -> Any:
    """Reliability (calibration) curve against the ideal diagonal."""
    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=n_bins)
    return _line(
        [{"x": prob_pred, "y": prob_true, "name": "Model"}],
        "Calibration Curve",
        "Mean predicted probability",
        "Observed frequency",
        backend,
        diagonal=True,
    )


def feature_importance_chart(
    importance: pd.DataFrame,
    top_n: int = 15,
    backend: str = "plotly",
) -> Any:
    """Bar chart of the top mean-absolute-SHAP features.

    Args:
        importance: DataFrame with ``feature`` and ``mean_abs_shap`` columns.
        top_n: Number of top features to display.
        backend: Chart backend.

    Raises:
        ValueError: If required columns are missing.
    """
    required = {"feature", "mean_abs_shap"}
    if not required <= set(importance.columns):
        raise ValueError(f"`importance` must have columns {required}.")

    top = importance.head(top_n)
    return _bar(
        top["feature"].tolist(),
        top["mean_abs_shap"].tolist(),
        "Global Feature Importance (mean |SHAP|)",
        "mean |SHAP|",
        backend,
    )
