"""Tests for reporting charts and tables."""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

matplotlib.use("Agg")
from matplotlib.figure import Figure as MplFigure  # noqa: E402

from churn_platform.reporting.charts import (  # noqa: E402
    calibration_chart,
    churn_overview_chart,
    feature_importance_chart,
    precision_recall_chart,
    roc_curve_chart,
)
from churn_platform.reporting.tables import (  # noqa: E402
    kpi_overview_table,
    model_metrics_table,
    retention_shortlist_table,
    segment_profile_table,
)


@pytest.fixture
def scores() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=100)
    y_score = np.clip(y_true * 0.5 + rng.uniform(size=100) * 0.5, 0, 1)
    return y_true, y_score


@pytest.mark.parametrize(
    "backend,expected", [("plotly", go.Figure), ("matplotlib", MplFigure)]
)
def test_curve_charts_return_backend_figure(scores, backend, expected) -> None:
    y_true, y_score = scores
    assert isinstance(roc_curve_chart(y_true, y_score, backend=backend), expected)
    assert isinstance(
        precision_recall_chart(y_true, y_score, backend=backend), expected
    )
    assert isinstance(calibration_chart(y_true, y_score, backend=backend), expected)


def test_churn_overview_chart_both_backends() -> None:
    summary = {"active_customers": 80, "churned_customers": 20}
    assert isinstance(churn_overview_chart(summary, backend="plotly"), go.Figure)
    assert isinstance(churn_overview_chart(summary, backend="matplotlib"), MplFigure)


def test_invalid_backend_raises() -> None:
    with pytest.raises(ValueError):
        churn_overview_chart({"active_customers": 1, "churned_customers": 1}, "bogus")


def test_feature_importance_chart_validates_columns() -> None:
    with pytest.raises(ValueError):
        feature_importance_chart(pd.DataFrame({"wrong": [1]}))
    good = pd.DataFrame({"feature": ["a", "b"], "mean_abs_shap": [0.3, 0.1]})
    assert isinstance(feature_importance_chart(good), go.Figure)


def test_kpi_overview_table() -> None:
    churn = {"total_customers": 100, "churned_customers": 27, "churn_rate": 0.27}
    revenue = {
        "total_revenue": 5000.0,
        "churned_customer_revenue": 1200.0,
        "average_monthly_charges": 64.0,
    }
    table = kpi_overview_table(churn, revenue)
    assert list(table.columns) == ["metric", "value"]
    assert len(table) == 6


def test_model_metrics_table_preserves_order() -> None:
    metrics = {"roc_auc": 0.85, "pr_auc": 0.65}
    table = model_metrics_table(metrics)
    assert table["metric"].tolist() == ["roc_auc", "pr_auc"]


def test_segment_profile_table_rounds_and_resets_index() -> None:
    profile = pd.DataFrame(
        {"customer_count": [10, 20], "churn_rate": [0.123456, 0.98765]},
        index=pd.Index([0, 1], name="segment"),
    )
    table = segment_profile_table(profile, round_to=2)
    assert "segment" in table.columns
    assert table.loc[0, "churn_rate"] == 0.12


def test_retention_shortlist_table() -> None:
    prioritized = pd.DataFrame(
        {
            "priority_rank": pd.array([1, 2, pd.NA], dtype="Int64"),
            "selected": [True, True, False],
            "expected_net_benefit": [500.0, 300.0, -10.0],
            "persona": ["High-Value At-Risk", "Low-Value At-Risk", "x"],
        }
    )
    actions = {0: ["Call", "Discount"], 1: ["Email"]}
    table = retention_shortlist_table(prioritized, actions, persona_column="persona")

    assert len(table) == 2
    assert table.loc[0, "recommended_actions"] == "Call; Discount"
    assert "persona" in table.columns
    assert table["priority_rank"].tolist() == [1, 2]


def test_retention_shortlist_missing_columns_raises() -> None:
    with pytest.raises(ValueError):
        retention_shortlist_table(pd.DataFrame({"selected": [True]}), {})
