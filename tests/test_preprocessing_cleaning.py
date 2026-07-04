"""Tests for preprocessing cleaning behavior."""

from __future__ import annotations

import pandas as pd
import pytest

from churn_platform.preprocessing.cleaning import DataCleaner
from churn_platform.utils.config import load_config


@pytest.fixture
def config() -> dict:
    """Load reusable project configuration for tests."""
    return load_config("config/config.yaml")


@pytest.fixture
def raw_customer_df() -> pd.DataFrame:
    """Provide a reusable raw customer churn DataFrame for tests."""
    return pd.DataFrame(
        {
            "CustomerID": ["0001-A", "0002-B", "0003-C"],
            "Count": [1, 1, 1],
            "Country": ["United States", "United States", "United States"],
            "State": ["California", "Texas", "New York"],
            "City": ["Los Angeles", "Dallas", "New York"],
            "Zip Code": [90001, 75001, 10001],
            "Lat Long": ["34.05, -118.24", "32.78, -96.80", "40.71, -74.00"],
            "Latitude": [34.05, 32.78, 40.71],
            "Longitude": [-118.24, -96.80, -74.00],
            "Gender": ["Female", "Male", "Female"],
            "Senior Citizen": ["No", "Yes", "No"],
            "Partner": ["Yes", "No", "Yes"],
            "Dependents": ["No", "Yes", "No"],
            "Tenure Months": [1, 24, 60],
            "Phone Service": ["No", "Yes", "Yes"],
            "Multiple Lines": ["No phone service", "No", "Yes"],
            "Internet Service": ["DSL", "Fiber optic", "DSL"],
            "Online Security": ["No", "Yes", "Yes"],
            "Online Backup": ["Yes", "No", "Yes"],
            "Device Protection": ["No", "Yes", "Yes"],
            "Tech Support": ["No", "No", "Yes"],
            "Streaming TV": ["No", "Yes", "Yes"],
            "Streaming Movies": ["No", "Yes", "Yes"],
            "Contract": ["Month-to-month", "One year", "Two year"],
            "Paperless Billing": ["Yes", "No", "Yes"],
            "Payment Method": [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
            ],
            "Monthly Charges": [29.85, 56.95, 1889.50],
            "Total Charges": ["29.85", " ", "1889.5"],
            "Churn Label": ["No", "Yes", "No"],
            "Churn Value": [0, 1, 0],
            "Churn Score": [10, 90, 20],
            "CLTV": [2000, 3500, 5000],
            "Churn Reason": [None, "Price too high", None],
        }
    )


def test_data_cleaner_returns_dataframe(
    raw_customer_df: pd.DataFrame,
    config: dict,
) -> None:
    """DataCleaner should return a pandas DataFrame."""
    cleaner = DataCleaner(
        config=config["runtime"],
        schema=config["schema"],
    )
    result = cleaner.clean(raw_customer_df)

    assert isinstance(result, pd.DataFrame)
    assert not result.empty


def test_data_cleaner_parses_total_charges_to_numeric(
    raw_customer_df: pd.DataFrame,
    config: dict,
) -> None:
    """DataCleaner should coerce Total Charges to numeric values."""
    cleaner = DataCleaner(
        config=config["runtime"],
        schema=config["schema"],
    )
    result = cleaner.clean(raw_customer_df)

    assert "Total Charges" in result.columns
    assert pd.api.types.is_numeric_dtype(result["Total Charges"])
    assert result["Total Charges"].notna().sum() >= 2


def test_data_cleaner_preserves_churn_label_without_define_binary_target(
    raw_customer_df: pd.DataFrame,
    config: dict,
) -> None:
    """DataCleaner should preserve the churn label column while cleaning."""
    cleaner = DataCleaner(
        config=config["runtime"],
        schema=config["schema"],
    )
    result = cleaner.clean(raw_customer_df)

    assert "Churn Label" in result.columns
    assert result["Churn Label"].tolist() == ["No", "Yes", "No"]


def test_data_cleaner_rejects_invalid_input(config: dict) -> None:
    """DataCleaner should raise an exception for invalid input types."""
    cleaner = DataCleaner(
        config=config["runtime"],
        schema=config["schema"],
    )

    with pytest.raises(TypeError):
        cleaner.clean(None)  # type: ignore[arg-type]


def test_data_cleaner_exposes_report_after_cleaning(
    raw_customer_df: pd.DataFrame,
    config: dict,
) -> None:
    """DataCleaner should expose a cleaning report after running."""
    cleaner = DataCleaner(
        config=config["runtime"],
        schema=config["schema"],
    )
    cleaner.clean(raw_customer_df)
    report = cleaner.get_report()

    assert report is not None
