"""Tests for preprocessing split logic."""

from __future__ import annotations

import pandas as pd

from churn_platform.preprocessing.preprocessing_config import SplitConfig
from churn_platform.preprocessing.splitting import split_train_validation_test


def test_split_train_validation_test_sizes() -> None:
    X = pd.DataFrame({"feature": range(100)})
    y = pd.Series([0, 1] * 50)

    X_train, X_validation, X_test, y_train, y_validation, y_test = split_train_validation_test(
        X,
        y,
        SplitConfig(train_size=0.70, validation_size=0.15, test_size=0.15, random_state=42),
    )

    assert len(X_train) == 70
    assert len(X_validation) == 15
    assert len(X_test) == 15
    assert len(y_train) == 70
    assert len(y_validation) == 15
    assert len(y_test) == 15

