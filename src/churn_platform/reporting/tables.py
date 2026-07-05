"""Reusable reporting tables.

Each function returns a plain, aggregated pandas DataFrame with no formatting or
side effects, so presentation (styling, number formats) stays a caller concern
and the outputs remain easy to test and reuse across the app and reports.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

LOGGER = logging.getLogger(__name__)

__all__ = [
    "kpi_overview_table",
    "model_metrics_table",
    "segment_profile_table",
    "retention_shortlist_table",
]


def kpi_overview_table(
    churn_summary: Mapping[str, Any],
    revenue_summary: Mapping[str, Any],
) -> pd.DataFrame:
    """Build an executive KPI overview from churn and revenue summaries.

    Args:
        churn_summary: Output of ``BusinessEDA.churn_summary``.
        revenue_summary: Output of ``BusinessEDA.revenue_summary``.

    Returns:
        Two-column ``metric`` / ``value`` DataFrame of headline KPIs.
    """
    rows = [
        ("Total customers", churn_summary["total_customers"]),
        ("Churned customers", churn_summary["churned_customers"]),
        ("Churn rate", churn_summary["churn_rate"]),
        ("Total revenue", revenue_summary["total_revenue"]),
        ("Revenue at risk (churned)", revenue_summary["churned_customer_revenue"]),
        ("Avg monthly charge", revenue_summary["average_monthly_charges"]),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def model_metrics_table(metrics: Mapping[str, float]) -> pd.DataFrame:
    """Tabulate a model evaluation metrics dictionary.

    Args:
        metrics: Output of ``evaluate_classifier``.

    Returns:
        Two-column ``metric`` / ``value`` DataFrame, ordered as given.
    """
    return pd.DataFrame(list(metrics.items()), columns=["metric", "value"])


def segment_profile_table(profile: pd.DataFrame, round_to: int = 2) -> pd.DataFrame:
    """Format a segment profile for display.

    Args:
        profile: Output of ``CustomerSegmenter.profile`` (segment-indexed).
        round_to: Decimal places for numeric columns.

    Returns:
        A copy with the segment index promoted to a ``segment`` column and
        numeric values rounded.
    """
    table = profile.copy(deep=True).reset_index()
    numeric_columns = table.select_dtypes(include="number").columns
    table[numeric_columns] = table[numeric_columns].round(round_to)
    return table


def retention_shortlist_table(
    prioritized: pd.DataFrame,
    actions: Mapping[Any, Sequence[str]],
    persona_column: str | None = None,
    extra_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Build the prioritized retention outreach shortlist.

    Args:
        prioritized: Output of ``RetentionPrioritizer.prioritize`` (must contain
            ``priority_rank``, ``selected`` and ``expected_net_benefit``).
        actions: Mapping of customer index to recommended actions.
        persona_column: Optional column holding each customer's persona.
        extra_columns: Additional columns to carry through (e.g. churn prob).

    Returns:
        Selected customers ordered by rank with ``priority_rank``,
        ``expected_net_benefit``, optional persona/extra columns, and a
        semicolon-joined ``recommended_actions`` string.

    Raises:
        ValueError: If required columns are missing.
    """
    required = {"priority_rank", "selected", "expected_net_benefit"}
    missing = required - set(prioritized.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    selected = prioritized[prioritized["selected"]].sort_values("priority_rank")

    columns = ["priority_rank"]
    if persona_column and persona_column in selected.columns:
        columns.append(persona_column)
    columns.extend(c for c in extra_columns if c in selected.columns)
    columns.append("expected_net_benefit")

    table = selected[columns].copy()
    table["recommended_actions"] = [
        "; ".join(actions.get(idx, [])) for idx in selected.index
    ]
    return table.reset_index(drop=True)
