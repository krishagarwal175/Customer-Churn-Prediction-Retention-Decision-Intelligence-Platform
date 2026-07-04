# tests/test_preprocessing_pipeline.py
"""Tests for the preprocessing pipeline."""

from __future__ import annotations

import pandas as pd
import pytest

from churn_platform.preprocessing.pipeline import PreprocessingPipeline
from churn_platform.utils.config import load_config


@pytest.fixture
def config() -> dict:
    """Load reusable project configuration for tests."""
    return load_config("config/config.yaml")


@pytest.fixture
def customer_df() -> pd.DataFrame:
    """Provide a schema-compliant IBM Telco customer churn DataFrame."""
    return pd.DataFrame(
        {
            "CustomerID": ["A", "B", "C", "D"],
            "Count": [1, 1, 1, 1],
            "Country": ["United States"] * 4,
            "State": ["California", "California", "Texas", "New York"],
            "City": ["Los Angeles", "San Diego", "Dallas", "New York"],
            "Zip Code": [90001, 92101, 75001, 10001],
            "Lat Long": [
                "34.05, -118.24",
                "32.72, -117.16",
                "32.78, -96.80",
                "40.71, -74.00",
            ],
            "Latitude": [34.05, 32.72, 32.78, 40.71],
            "Longitude": [-118.24, -117.16, -96.80, -74.00],
            "Gender": ["Female", "Male", "Female", "Male"],
            "Senior Citizen": ["No", "No", "Yes", "No"],
            "Partner": ["Yes", "No", "Yes", "No"],
            "Dependents": ["No", "No", "Yes", "Yes"],
            "Tenure Months": [1, 2, 3, 4],
            "Phone Service": ["No", "Yes", "Yes", "Yes"],
            "Multiple Lines": ["No phone service", "No", "Yes", "No"],
            "Internet Service": ["DSL", "Fiber optic", "DSL", "Fiber optic"],
            "Online Security": ["No", "Yes", "No", "Yes"],
            "Online Backup": ["Yes", "No", "Yes", "No"],
            "Device Protection": ["No", "Yes", "No", "Yes"],
            "Tech Support": ["No", "No", "Yes", "Yes"],
            "Streaming TV": ["No", "Yes", "Yes", "No"],
            "Streaming Movies": ["No", "Yes", "Yes", "No"],
            "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month"],
            "Paperless Billing": ["Yes", "No", "Yes", "No"],
            "Payment Method": [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
            "Monthly Charges": [10.0, 20.0, 30.0, 40.0],
            "Total Charges": ["10.0", "40.0", "90.0", "160.0"],
            "Churn Label": ["No", "Yes", "No", "Yes"],
            "Churn Value": [0, 1, 0, 1],
            "Churn Score": [10, 90, 20, 80],
            "CLTV": [2000, 3000, 2500, 4000],
            "Churn Reason": [None, "Price too high", None, "Moved"],
        }
    )


def test_preprocessing_pipeline_transform_returns_cleaned_dataframe(
    config: dict,
    customer_df: pd.DataFrame,
) -> None:
    """Pipeline should transform a valid DataFrame and preserve row count."""
    pipeline = PreprocessingPipeline(
        config=config["runtime"],
        schema=config["schema"],
    )

    transformed = pipeline.transform(customer_df)

    assert isinstance(transformed, pd.DataFrame)
    assert transformed.shape[0] == customer_df.shape[0]
    assert "Total Charges" in transformed.columns
    assert pd.api.types.is_numeric_dtype(transformed["Total Charges"])


def test_preprocessing_pipeline_supports_inference_without_target_column(
    config: dict,
    customer_df: pd.DataFrame,
) -> None:
    """Pipeline should support transform when Churn Label is absent."""
    pipeline = PreprocessingPipeline(
        config=config["runtime"],
        schema=config["schema"],
    )

    training_like = pipeline.transform(customer_df)
    inference_df = customer_df.drop(columns=["Churn Label"]).copy()
    inference_df.loc[0, "Contract"] = "Unexpected Contract"
    inference_like = pipeline.transform(inference_df)

    assert training_like.shape[0] == inference_like.shape[0] == 4
    assert "Total Charges" in inference_like.columns


def test_preprocessing_pipeline_rejects_invalid_input(config: dict) -> None:
    """Pipeline should reject invalid non-DataFrame input."""
    pipeline = PreprocessingPipeline(
        config=config["runtime"],
        schema=config["schema"],
    )

    with pytest.raises(TypeError):
        pipeline.transform(None)  # type: ignore[arg-type]
