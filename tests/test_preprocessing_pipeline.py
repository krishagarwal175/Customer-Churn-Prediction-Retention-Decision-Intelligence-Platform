"""Tests for the sklearn preprocessing pipeline."""

from __future__ import annotations

import pandas as pd

from churn_platform.preprocessing.pipeline import ChurnPreprocessor
from churn_platform.preprocessing.preprocessing_config import PreprocessingConfig


def test_preprocessor_fit_transform_supports_unseen_categories() -> None:
    dataframe = pd.DataFrame(
        {
            "CustomerID": ["A", "B", "C", "D"],
            "Count": [1, 1, 1, 1],
            "Country": ["United States"] * 4,
            "State": ["California"] * 4,
            "Lat Long": ["0, 0"] * 4,
            "Gender": ["Female", "Male", "Female", "Male"],
            "Tenure Months": [1, 2, 3, 4],
            "Monthly Charges": [10.0, 20.0, 30.0, 40.0],
            "Total Charges": [10.0, 40.0, 90.0, 160.0],
            "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month"],
            "Churn Label": ["No", "Yes", "No", "Yes"],
            "Churn Value": [0, 1, 0, 1],
            "Churn Score": [10, 90, 20, 80],
            "Churn Reason": [None, "Price too high", None, "Moved"],
        }
    )
    preprocessor = ChurnPreprocessor(PreprocessingConfig())

    transformed_train = preprocessor.fit_transform(dataframe)
    inference_data = dataframe.drop(columns=["Churn Label"]).copy()
    inference_data.loc[0, "Contract"] = "Unexpected Contract"
    transformed_inference = preprocessor.transform(inference_data)

    assert transformed_train.shape[0] == 4
    assert transformed_inference.shape[0] == 4
    assert transformed_train.shape[1] == transformed_inference.shape[1]

