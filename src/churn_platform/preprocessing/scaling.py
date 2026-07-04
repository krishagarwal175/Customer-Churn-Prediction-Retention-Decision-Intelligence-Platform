"""Numerical scaling helpers."""

from __future__ import annotations

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from churn_platform.preprocessing.preprocessing_config import CleaningConfig


def build_scaler(config: CleaningConfig) -> StandardScaler | MinMaxScaler | str:
    """Build the configured numeric scaler.

    Returns ``passthrough`` when numeric scaling is disabled.
    """

    if not config.scale_numeric:
        return "passthrough"
    if config.scaler == "standard":
        return StandardScaler()
    if config.scaler == "minmax":
        return MinMaxScaler()
    raise ValueError(f"Unsupported scaler: {config.scaler}")
