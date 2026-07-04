"""Cleaning transformers for raw validated churn data."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from churn_platform.preprocessing.preprocessing_config import CleaningConfig

LOGGER = logging.getLogger(__name__)


class DataCleaner(BaseEstimator, TransformerMixin):
    """Clean source columns without engineering new features.

    The transformer is intentionally conservative: it standardizes whitespace,
    normalizes empty strings, converts known numeric columns, fills missing
    values according to configuration, and keeps original feature semantics.
    """

    def __init__(self, config: CleaningConfig | None = None) -> None:
        self.config = config or CleaningConfig()
        self.numeric_fill_values_: dict[str, float] = {}
        self.categorical_columns_: list[str] = []
        self.numeric_columns_: list[str] = []

    def fit(self, X: pd.DataFrame, y: Any = None) -> "DataCleaner":
        """Learn training-set imputation values."""

        dataframe = self._basic_clean(X)
        self.numeric_columns_ = dataframe.select_dtypes(
            include=["number"]
        ).columns.tolist()
        self.categorical_columns_ = [
            column
            for column in dataframe.columns
            if column not in self.numeric_columns_
        ]

        for column in self.numeric_columns_:
            if self.config.numeric_imputation_strategy == "median":
                fill_value = dataframe[column].median()
            elif self.config.numeric_imputation_strategy == "mean":
                fill_value = dataframe[column].mean()
            else:
                fill_value = 0
            self.numeric_fill_values_[column] = float(
                0 if pd.isna(fill_value) else fill_value
            )

        LOGGER.info(
            "DataCleaner fitted: numeric_columns=%s categorical_columns=%s",
            len(self.numeric_columns_),
            len(self.categorical_columns_),
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply cleaning and configured missing-value handling."""

        dataframe = self._basic_clean(X)

        for column, fill_value in self.numeric_fill_values_.items():
            if column in dataframe.columns:
                missing_before = int(dataframe[column].isna().sum())
                dataframe[column] = dataframe[column].fillna(fill_value)
                if missing_before:
                    LOGGER.info(
                        "Filled %s missing numeric values in '%s' with %s",
                        missing_before,
                        column,
                        fill_value,
                    )

        for column in dataframe.columns:
            if column not in self.numeric_fill_values_:
                missing_before = int(dataframe[column].isna().sum())
                dataframe[column] = dataframe[column].fillna(
                    self.config.categorical_fill_value
                )
                if missing_before:
                    LOGGER.info(
                        "Filled %s missing categorical values in '%s' with '%s'",
                        missing_before,
                        column,
                        self.config.categorical_fill_value,
                    )

        return dataframe

    def _basic_clean(self, X: pd.DataFrame) -> pd.DataFrame:
        dataframe = X.copy()

        for column in dataframe.select_dtypes(include=["object", "string"]).columns:
            dataframe[column] = dataframe[column].astype("string").str.strip()
            dataframe[column] = dataframe[column].replace("", pd.NA)

        total_charges = self.config.total_charges_column
        tenure = self.config.tenure_column
        if total_charges in dataframe.columns:
            LOGGER.info("Converting '%s' to numeric", total_charges)
            dataframe[total_charges] = pd.to_numeric(
                dataframe[total_charges], errors="coerce"
            )

            if tenure in dataframe.columns:
                zero_tenure_mask = dataframe[tenure].fillna(-1).eq(0)
                missing_total_mask = dataframe[total_charges].isna()
                fill_mask = zero_tenure_mask & missing_total_mask
                if fill_mask.any():
                    dataframe.loc[fill_mask, total_charges] = (
                        self.config.zero_tenure_total_charges_value
                    )
                    LOGGER.info(
                        "Filled %s zero-tenure '%s' values with %s",
                        int(fill_mask.sum()),
                        total_charges,
                        self.config.zero_tenure_total_charges_value,
                    )

        for column in dataframe.columns:
            if column != total_charges:
                numeric_candidate = pd.to_numeric(dataframe[column], errors="coerce")
                non_missing_count = int(dataframe[column].notna().sum())
                numeric_count = int(numeric_candidate.notna().sum())
                if non_missing_count > 0 and numeric_count == non_missing_count:
                    dataframe[column] = numeric_candidate

        return dataframe


def define_binary_target(dataframe: pd.DataFrame, config: CleaningConfig) -> pd.Series:
    """Convert the configured churn target into binary labels.

    Args:
        dataframe: Cleaned dataset containing the configured target column.
        config: Cleaning configuration with target label definitions.

    Returns:
        Binary target series where the positive churn label is 1.
    """

    if config.target_column not in dataframe.columns:
        raise KeyError(f"Target column is missing: {config.target_column}")

    target = dataframe[config.target_column].astype("string").str.strip()
    mapping = {
        config.positive_target_label: 1,
        config.negative_target_label: 0,
    }
    encoded = target.map(mapping)
    if encoded.isna().any():
        unexpected = sorted(target[encoded.isna()].dropna().unique().tolist())
        raise ValueError(f"Unexpected target labels found: {unexpected}")

    LOGGER.info("Defined binary target from '%s'", config.target_column)
    return encoded.astype(int)
