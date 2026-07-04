"""Reproducible train, validation, and test splitting."""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from churn_platform.preprocessing.preprocessing_config import SplitConfig

LOGGER = logging.getLogger(__name__)


def split_train_validation_test(
    X: pd.DataFrame,
    y: pd.Series,
    config: SplitConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Create stratified train, validation, and test splits.

    The default 70/15/15 split gives the model enough training data while
    preserving separate validation and final holdout sets.
    """

    split_total = round(
        config.train_size + config.validation_size + config.test_size, 6
    )
    if split_total != 1.0:
        raise ValueError("Train, validation, and test ratios must sum to 1.0")

    stratify = y if config.stratify else None
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=(config.validation_size + config.test_size),
        random_state=config.random_state,
        stratify=stratify,
    )

    relative_test_size = config.test_size / (config.validation_size + config.test_size)
    temp_stratify = y_temp if config.stratify else None
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=relative_test_size,
        random_state=config.random_state,
        stratify=temp_stratify,
    )

    LOGGER.info(
        "Created train/validation/test split: train=%s validation=%s test=%s",
        len(X_train),
        len(X_validation),
        len(X_test),
    )
    return X_train, X_validation, X_test, y_train, y_validation, y_test
