"""Backward-compatible import location for dataset loading utilities."""

from churn_platform.data.loader import DatasetLoadError, load_dataset

__all__ = ["DatasetLoadError", "load_dataset"]
