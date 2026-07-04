"""Categorical encoding helpers."""

from __future__ import annotations

from sklearn.preprocessing import OneHotEncoder

from churn_platform.preprocessing.preprocessing_config import CleaningConfig


def build_categorical_encoder(config: CleaningConfig) -> OneHotEncoder:
    """Build a categorical encoder that supports unseen inference categories."""

    if config.encoder != "onehot":
        raise ValueError(f"Unsupported encoder: {config.encoder}")
    return OneHotEncoder(handle_unknown=config.handle_unknown, sparse_output=False)
