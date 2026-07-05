"""Validation checks for the IBM Telco Customer Churn dataset."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.api import types as pd_types

from churn_platform.data.schema import DatasetSchema, get_schema

LOGGER = logging.getLogger(__name__)


@dataclass
class ValidationIssue:
    """Single validation warning or error."""

    check: str
    severity: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Structured validation result returned by the validator."""

    passed: bool
    row_count: int
    column_count: int
    issues: list[ValidationIssue] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the validation result to a plain dictionary."""

        return {
            "passed": self.passed,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "issues": [asdict(issue) for issue in self.issues],
            "metrics": self.metrics,
        }


def _add_issue(
    issues: list[ValidationIssue],
    check: str,
    severity: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> None:
    issues.append(
        ValidationIssue(
            check=check,
            severity=severity,
            message=message,
            details=details or {},
        )
    )


def _matches_dtype(series: pd.Series, expected_dtype: str) -> bool:
    if expected_dtype == "string":
        return pd_types.is_object_dtype(series) or pd_types.is_string_dtype(series)
    if expected_dtype == "category":
        return (
            pd_types.is_object_dtype(series)
            or pd_types.is_string_dtype(series)
            or pd_types.is_categorical_dtype(series)
        )
    if expected_dtype == "integer":
        return pd_types.is_integer_dtype(series)
    if expected_dtype == "float":
        return pd_types.is_float_dtype(series) or pd_types.is_integer_dtype(series)
    return True


def validate_dataset(
    dataframe: pd.DataFrame,
    schema: DatasetSchema | None = None,
) -> ValidationResult:
    """Validate the loaded dataset against structural and semantic checks.

    Args:
        dataframe: Loaded raw dataset.
        schema: Optional schema override. Defaults to the IBM Telco schema.

    Returns:
        Structured validation result with pass/fail status, metrics, and
        issue details.
    """

    active_schema = schema or get_schema()
    LOGGER.info("Starting dataset validation")
    issues: list[ValidationIssue] = []

    row_count, column_count = dataframe.shape
    metrics: dict[str, Any] = {
        "shape": {"rows": row_count, "columns": column_count},
        "duplicate_rows": int(dataframe.duplicated().sum()),
        "missing_values": dataframe.isna().sum().astype(int).to_dict(),
        "empty_strings": {},
        "duplicate_customer_ids": None,
    }

    if row_count == 0:
        _add_issue(issues, "row_count", "error", "Dataset contains zero rows.")

    expected_columns = set(active_schema.expected_column_names)
    actual_columns = set(dataframe.columns)
    missing_columns = sorted(expected_columns - actual_columns)
    unexpected_columns = sorted(actual_columns - expected_columns)

    if missing_columns:
        _add_issue(
            issues,
            "required_columns",
            "error",
            "Dataset is missing expected columns.",
            {"missing_columns": missing_columns},
        )

    if unexpected_columns:
        _add_issue(
            issues,
            "unexpected_columns",
            "warning",
            "Dataset contains columns not present in the source schema.",
            {"unexpected_columns": unexpected_columns},
        )

    if metrics["duplicate_rows"]:
        _add_issue(
            issues,
            "duplicate_rows",
            "warning",
            "Dataset contains duplicate rows.",
            {"duplicate_rows": metrics["duplicate_rows"]},
        )

    identifier = active_schema.identifier_column
    if identifier in dataframe.columns:
        duplicate_ids = int(dataframe[identifier].duplicated().sum())
        metrics["duplicate_customer_ids"] = duplicate_ids
        if duplicate_ids:
            _add_issue(
                issues,
                "duplicate_customer_ids",
                "error",
                "Dataset contains duplicate CustomerID values.",
                {"duplicate_customer_ids": duplicate_ids},
            )

    for column in active_schema.expected_columns:
        if column.name not in dataframe.columns:
            continue

        series = dataframe[column.name]

        if not column.nullable:
            missing_count = int(series.isna().sum())
            if missing_count:
                _add_issue(
                    issues,
                    "missing_values",
                    "error",
                    f"Column '{column.name}' contains missing values.",
                    {"column": column.name, "missing_count": missing_count},
                )

        if pd_types.is_object_dtype(series) or pd_types.is_string_dtype(series):
            empty_count = int(series.fillna("").astype(str).str.strip().eq("").sum())
            metrics["empty_strings"][column.name] = empty_count
            if empty_count:
                severity = (
                    "warning" if column.nullable or column.coerce_numeric else "error"
                )
                _add_issue(
                    issues,
                    "empty_strings",
                    severity,
                    f"Column '{column.name}' contains empty strings.",
                    {"column": column.name, "empty_string_count": empty_count},
                )

        if column.coerce_numeric:
            dtype_matches = pd.to_numeric(series, errors="coerce").notna().sum() > 0
        else:
            dtype_matches = _matches_dtype(series, column.dtype)

        if not dtype_matches:
            _add_issue(
                issues,
                "datatype_consistency",
                "error",
                f"Column '{column.name}' does not match expected dtype group.",
                {
                    "column": column.name,
                    "expected_dtype": column.dtype,
                    "actual_dtype": str(series.dtype),
                },
            )

        if column.allowed_values is not None:
            observed = set(series.dropna().unique().tolist())
            allowed = set(column.allowed_values)
            unexpected = sorted(str(value) for value in observed - allowed)
            if unexpected:
                _add_issue(
                    issues,
                    "unexpected_categorical_values",
                    "error",
                    f"Column '{column.name}' contains unexpected values.",
                    {"column": column.name, "unexpected_values": unexpected},
                )

        if column.minimum is not None:
            invalid_min = int(
                (pd.to_numeric(series, errors="coerce") < column.minimum).sum()
            )
            if invalid_min:
                _add_issue(
                    issues,
                    "invalid_numerical_values",
                    "error",
                    f"Column '{column.name}' contains values below minimum.",
                    {
                        "column": column.name,
                        "minimum": column.minimum,
                        "count": invalid_min,
                    },
                )

        if column.maximum is not None:
            invalid_max = int(
                (pd.to_numeric(series, errors="coerce") > column.maximum).sum()
            )
            if invalid_max:
                _add_issue(
                    issues,
                    "invalid_numerical_values",
                    "error",
                    f"Column '{column.name}' contains values above maximum.",
                    {
                        "column": column.name,
                        "maximum": column.maximum,
                        "count": invalid_max,
                    },
                )

    passed = not any(issue.severity == "error" for issue in issues)
    LOGGER.info(
        "Validation completed: passed=%s errors=%s warnings=%s",
        passed,
        sum(issue.severity == "error" for issue in issues),
        sum(issue.severity == "warning" for issue in issues),
    )
    return ValidationResult(
        passed=passed,
        row_count=row_count,
        column_count=column_count,
        issues=issues,
        metrics=metrics,
    )


def save_validation_result(result: ValidationResult, output_path: str | Path) -> None:
    """Save validation results as JSON."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    LOGGER.info("Validation report saved to %s", path)
