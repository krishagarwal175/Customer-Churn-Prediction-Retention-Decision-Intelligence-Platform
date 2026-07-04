"""Tests for dataset schema definitions."""

from __future__ import annotations

from churn_platform.data.schema import get_schema


def test_schema_contains_target_and_identifier() -> None:
    schema = get_schema()

    assert schema.target_column == "Churn Label"
    assert schema.identifier_column == "CustomerID"
    assert schema.target_column in schema.expected_column_names
    assert schema.identifier_column in schema.expected_column_names


def test_schema_marks_known_leakage_columns() -> None:
    schema = get_schema()

    assert "Churn Reason" in schema.leakage_columns
    assert "Churn Score" in schema.leakage_columns
    assert "Churn Value" in schema.leakage_columns
