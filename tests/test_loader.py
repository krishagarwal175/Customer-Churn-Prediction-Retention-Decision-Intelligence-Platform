"""Tests for dataset loading."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from churn_platform.data.loader import DatasetLoadError, load_dataset


def test_load_dataset_reads_csv(tmp_path: Path) -> None:
    dataset_path = tmp_path / "sample.csv"
    pd.DataFrame({"CustomerID": ["A"], "Churn Label": ["No"]}).to_csv(
        dataset_path,
        index=False,
    )

    dataframe = load_dataset(dataset_path)

    assert dataframe.shape == (1, 2)
    assert list(dataframe.columns) == ["CustomerID", "Churn Label"]


def test_load_dataset_raises_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "missing.csv")


def test_load_dataset_raises_for_unsupported_extension(tmp_path: Path) -> None:
    dataset_path = tmp_path / "sample.txt"
    dataset_path.write_text("not a supported dataset", encoding="utf-8")

    with pytest.raises(DatasetLoadError):
        load_dataset(dataset_path)

