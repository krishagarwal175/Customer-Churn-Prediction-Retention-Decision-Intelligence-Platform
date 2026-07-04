"""Text-only dataset profiling for ingestion validation."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from pandas.api import types as pd_types

from churn_platform.data.schema import DatasetSchema, get_schema

LOGGER = logging.getLogger(__name__)


def build_profile_report(
    dataframe: pd.DataFrame,
    schema: DatasetSchema | None = None,
) -> str:
    """Create a lightweight textual profile report.

    The report intentionally avoids plots and advanced EDA. It provides only
    the structural summaries required by the ingestion milestone.

    Args:
        dataframe: Loaded and validated dataset.
        schema: Optional schema override.

    Returns:
        Markdown-formatted textual profile.
    """

    active_schema = schema or get_schema()
    dtype_map = active_schema.dtype_map
    numerical_columns = [
        column
        for column in dataframe.columns
        if pd_types.is_numeric_dtype(dataframe[column])
        or dtype_map.get(column) in {"integer", "float"}
    ]
    categorical_columns = [
        column for column in dataframe.columns if column not in numerical_columns
    ]

    lines: list[str] = [
        "# Data Profile Report",
        "",
        "## Dataset Shape",
        "",
        f"- Rows: {dataframe.shape[0]}",
        f"- Columns: {dataframe.shape[1]}",
        f"- Memory usage: {dataframe.memory_usage(deep=True).sum():,} bytes",
        "",
        "## Missing Values Summary",
        "",
        "| Column | Missing Values | Missing Percentage |",
        "|---|---:|---:|",
    ]

    for column, missing_count in dataframe.isna().sum().items():
        missing_pct = (missing_count / len(dataframe) * 100) if len(dataframe) else 0
        lines.append(f"| {column} | {int(missing_count)} | {missing_pct:.2f}% |")

    lines.extend(
        [
            "",
            "## Numerical Features",
            "",
            ", ".join(numerical_columns) if numerical_columns else "None",
            "",
            "## Categorical Features",
            "",
            ", ".join(categorical_columns) if categorical_columns else "None",
            "",
            "## Class Distribution",
            "",
        ]
    )

    target = active_schema.target_column
    if target in dataframe.columns:
        lines.extend(["| Class | Count | Percentage |", "|---|---:|---:|"])
        class_counts = dataframe[target].value_counts(dropna=False)
        for value, count in class_counts.items():
            pct = count / len(dataframe) * 100 if len(dataframe) else 0
            lines.append(f"| {value} | {int(count)} | {pct:.2f}% |")
    else:
        lines.append("Target column is not available.")

    lines.extend(
        [
            "",
            "## Unique Values",
            "",
            "| Column | Unique Values | Sample Values |",
            "|---|---:|---|",
        ]
    )

    for column in dataframe.columns:
        unique_count = int(dataframe[column].nunique(dropna=True))
        sample_values = dataframe[column].dropna().unique()[:10]
        sample = ", ".join(str(value) for value in sample_values)
        lines.append(f"| {column} | {unique_count} | {sample} |")

    return "\n".join(lines) + "\n"


def save_profile_report(report: str, output_path: str | Path) -> None:
    """Save a textual profile report to disk."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    LOGGER.info("Data profile report saved to %s", path)
