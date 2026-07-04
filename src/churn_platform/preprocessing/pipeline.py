# src/churn_platform/preprocessing/pipeline.py
"""Preprocessing pipeline orchestration for churn data."""

from __future__ import annotations

import logging

import pandas as pd

from churn_platform.preprocessing.cleaning import DataCleaner

logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """Coordinate preprocessing steps using the DataCleaner API."""

    def __init__(self, config: dict, schema: dict) -> None:
        """Initialize the preprocessing pipeline."""
        self.cleaner = DataCleaner(config=config, schema=schema)
        logger.info("PreprocessingPipeline initialized.")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the preprocessing pipeline on the provided DataFrame.

        Args:
            df: Input raw customer churn DataFrame.

        Returns:
            Cleaned DataFrame produced by the DataCleaner.

        Raises:
            TypeError: If ``df`` is not a pandas DataFrame.
        """
        if not isinstance(df, pd.DataFrame):
            logger.error("Input validation failed: df is not a pandas DataFrame.")
            raise TypeError("df must be a pandas DataFrame")

        logger.info(
            "Starting preprocessing transform on DataFrame with shape %s.", df.shape
        )
        cleaned_df = self.cleaner.clean(df)
        logger.info(
            "Completed preprocessing transform. Output DataFrame shape: %s.",
            cleaned_df.shape,
        )
        return cleaned_df
