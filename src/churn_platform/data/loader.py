"""Dataset loading utilities for the ingestion pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)


class DatasetLoadError(RuntimeError):
    """Raised when the raw dataset cannot be loaded."""


def load_dataset(file_path: str | Path) -> pd.DataFrame:
    """Load the raw churn dataset from a configured local path.

    Supported file formats are CSV, XLS, and XLSX. The function performs
    loading only; validation is handled by the validator module.

    Args:
        file_path: Relative or absolute path to the raw dataset file.

    Returns:
        Loaded dataset as a pandas DataFrame.

    Raises:
        FileNotFoundError: If the configured file path does not exist.
        DatasetLoadError: If the extension is unsupported or pandas cannot
            parse the file.
    """

    path = Path(file_path)
    LOGGER.info("Loading dataset from %s", path)

    if not path.exists():
        LOGGER.error("Dataset file does not exist: %s", path)
        raise FileNotFoundError(f"Dataset file does not exist: {path}")

    try:
        if path.suffix.lower() == ".csv":
            dataframe = pd.read_csv(path)
        elif path.suffix.lower() in {".xls", ".xlsx"}:
            dataframe = pd.read_excel(path)
        else:
            raise DatasetLoadError(
                f"Unsupported dataset format '{path.suffix}'. "
                "Expected .csv, .xls, or .xlsx."
            )
    except DatasetLoadError:
        raise
    except Exception as exc:  # pragma: no cover - defensive wrapping
        LOGGER.exception("Failed to load dataset from %s", path)
        raise DatasetLoadError(f"Failed to load dataset from {path}: {exc}") from exc

    LOGGER.info(
        "Dataset loaded successfully: rows=%s columns=%s",
        dataframe.shape[0],
        dataframe.shape[1],
    )
    return dataframe
