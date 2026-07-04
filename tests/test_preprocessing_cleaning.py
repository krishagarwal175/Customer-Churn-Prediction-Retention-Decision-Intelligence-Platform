"""Tests for preprocessing cleaning behavior."""

from __future__ import annotations

import pandas as pd

from churn_platform.preprocessing.cleaning import DataCleaner, define_binary_target
from churn_platform.preprocessing.preprocessing_config import CleaningConfig


def test_cleaner_trims_strings_and_converts_total_charges() -> None:
    config = CleaningConfig()
    dataframe = pd.DataFrame(
        {
            "CustomerID": [" A "],
            "Tenure Months": [0],
            "Total Charges": [" "],
            "Monthly Charges": [20.0],
        }
    )

    cleaner = DataCleaner(config).fit(dataframe)
    cleaned = cleaner.transform(dataframe)

    assert cleaned.loc[0, "CustomerID"] == "A"
    assert cleaned.loc[0, "Total Charges"] == 0.0


def test_define_binary_target_maps_churn_label() -> None:
    dataframe = pd.DataFrame({"Churn Label": ["Yes", "No"]})

    target = define_binary_target(dataframe, CleaningConfig())

    assert target.tolist() == [1, 0]

