"""Tests for dataset validation."""

from __future__ import annotations

import pandas as pd

from churn_platform.data.schema import ColumnSchema, DatasetSchema
from churn_platform.data.validator import validate_dataset


def test_validate_dataset_passes_for_valid_minimal_schema() -> None:
    schema = DatasetSchema(
        expected_columns=(
            ColumnSchema("CustomerID", "string"),
            ColumnSchema("Churn Label", "category", allowed_values=("Yes", "No")),
            ColumnSchema("Monthly Charges", "float", minimum=0),
        ),
        target_column="Churn Label",
        identifier_column="CustomerID",
    )
    dataframe = pd.DataFrame(
        {
            "CustomerID": ["A", "B"],
            "Churn Label": ["Yes", "No"],
            "Monthly Charges": [10.0, 20.0],
        }
    )

    result = validate_dataset(dataframe, schema)

    assert result.passed is True
    assert result.row_count == 2
    assert result.column_count == 3


def test_validate_dataset_fails_for_duplicate_customer_ids() -> None:
    schema = DatasetSchema(
        expected_columns=(
            ColumnSchema("CustomerID", "string"),
            ColumnSchema("Churn Label", "category", allowed_values=("Yes", "No")),
        ),
        target_column="Churn Label",
        identifier_column="CustomerID",
    )
    dataframe = pd.DataFrame(
        {
            "CustomerID": ["A", "A"],
            "Churn Label": ["Yes", "No"],
        }
    )

    result = validate_dataset(dataframe, schema)

    assert result.passed is False
    assert any(issue.check == "duplicate_customer_ids" for issue in result.issues)


def test_validate_dataset_fails_for_unexpected_category() -> None:
    schema = DatasetSchema(
        expected_columns=(
            ColumnSchema("CustomerID", "string"),
            ColumnSchema("Churn Label", "category", allowed_values=("Yes", "No")),
        ),
        target_column="Churn Label",
        identifier_column="CustomerID",
    )
    dataframe = pd.DataFrame(
        {
            "CustomerID": ["A", "B"],
            "Churn Label": ["Maybe", "No"],
        }
    )

    result = validate_dataset(dataframe, schema)

    assert result.passed is False
    assert any(issue.check == "unexpected_categorical_values" for issue in result.issues)
