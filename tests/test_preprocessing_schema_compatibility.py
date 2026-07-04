"""Tests for preprocessing configuration compatibility with source schema."""

from __future__ import annotations

from churn_platform.data.schema import get_schema
from churn_platform.preprocessing.preprocessing_config import CleaningConfig


def test_preprocessing_target_exists_in_source_schema() -> None:
    schema = get_schema()
    config = CleaningConfig()

    assert config.target_column in schema.expected_column_names


def test_configured_leakage_columns_include_schema_leakage_columns() -> None:
    schema = get_schema()
    config = CleaningConfig()

    configured = set(config.leakage_columns)
    required = {"Churn Value", "Churn Score", "Churn Reason"}

    assert required.issubset(configured)
    assert required.issubset(set(schema.leakage_columns))

