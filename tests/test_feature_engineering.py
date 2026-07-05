"""Tests for schema-driven business feature engineering."""

from __future__ import annotations

import pandas as pd
import pytest

from churn_platform.features.engineering import FeatureEngineer


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Provide a small cleaned churn-like DataFrame."""
    return pd.DataFrame(
        {
            "Tenure Months": [0, 6, 30, 72],
            "Contract": ["Month-to-month", "Month-to-month", "One year", "Two year"],
            "Monthly Charges": [50.0, 20.0, 30.0, 40.0],
            "Total Charges": [0.0, 120.0, 900.0, 2880.0],
            "CLTV": [2000, 3000, 5000, 8000],
            "Phone Service": ["Yes", "No", "Yes", "Yes"],
            "Internet Service": ["DSL", "No", "Fiber optic", "DSL"],
            "Online Security": ["No", "No internet service", "Yes", "Yes"],
        }
    )


def test_transform_adds_expected_columns(sample_df: pd.DataFrame) -> None:
    result = FeatureEngineer().transform(sample_df)

    for column in (
        "tenure_bucket",
        "service_count",
        "average_monthly_spend",
        "contract_commitment_score",
        "risk_value_quadrant",
    ):
        assert column in result.columns


def test_transform_does_not_mutate_input(sample_df: pd.DataFrame) -> None:
    original = sample_df.copy(deep=True)
    FeatureEngineer().transform(sample_df)
    pd.testing.assert_frame_equal(sample_df, original)


def test_service_count_ignores_negative_service_values(sample_df: pd.DataFrame) -> None:
    result = FeatureEngineer().transform(sample_df)
    assert result.loc[0, "service_count"] == 2  # Phone=Yes, Internet=DSL
    assert result.loc[1, "service_count"] == 0  # all No / No internet service


def test_average_monthly_spend_handles_zero_tenure(sample_df: pd.DataFrame) -> None:
    result = FeatureEngineer().transform(sample_df)
    assert result.loc[0, "average_monthly_spend"] == 50.0  # zero tenure -> monthly
    assert result.loc[2, "average_monthly_spend"] == pytest.approx(30.0)


def test_contract_commitment_score_is_ordinal(sample_df: pd.DataFrame) -> None:
    result = FeatureEngineer().transform(sample_df)
    assert result.loc[0, "contract_commitment_score"] == 0
    assert result.loc[2, "contract_commitment_score"] == 1
    assert result.loc[3, "contract_commitment_score"] == 2


def test_risk_value_quadrant_labels(sample_df: pd.DataFrame) -> None:
    result = FeatureEngineer().transform(sample_df)
    assert result.loc[3, "risk_value_quadrant"] == "High Value / Low Risk"
    assert result.loc[0, "risk_value_quadrant"] == "Low Value / High Risk"


def test_missing_columns_are_skipped_gracefully() -> None:
    df = pd.DataFrame({"Tenure Months": [1, 2, 3]})
    result = FeatureEngineer().transform(df)
    assert "tenure_bucket" in result.columns
    assert "contract_commitment_score" not in result.columns


def test_rejects_non_dataframe() -> None:
    with pytest.raises(TypeError):
        FeatureEngineer().transform(None)  # type: ignore[arg-type]


def test_rejects_empty_dataframe() -> None:
    with pytest.raises(ValueError):
        FeatureEngineer().transform(pd.DataFrame({"Tenure Months": []}))
