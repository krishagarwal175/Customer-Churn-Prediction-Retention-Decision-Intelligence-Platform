"""Model training: dataset-agnostic churn classifier pipelines.

Builds a full scikit-learn pipeline (imputation + encoding + scaling +
estimator) from any cleaned feature frame. Column roles are inferred by dtype
so the trainer adapts across similar datasets without code changes; estimator
choice and hyperparameters are configuration-driven.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

LOGGER = logging.getLogger(__name__)

__all__ = [
    "ModelConfig",
    "encode_target",
    "build_preprocessor",
    "build_estimator",
    "build_model_pipeline",
    "train_model",
]


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for building a churn classifier."""

    name: str = "logistic_regression"
    random_state: int = 42
    params: dict[str, Any] = field(default_factory=dict)


def encode_target(y: pd.Series, positive_label: str = "Yes") -> pd.Series:
    """Encode a categorical churn target into 1 (positive) / 0 (negative).

    Args:
        y: Target series (e.g. ``Churn Label`` with values Yes/No).
        positive_label: Value representing the positive (churn) class.

    Returns:
        Integer series aligned to ``y`` with 1 for the positive class.
    """
    positive = str(positive_label).strip().lower()
    encoded = (y.astype(str).str.strip().str.lower() == positive).astype(int)
    encoded.name = y.name
    return encoded


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build a dtype-inferred preprocessing transformer.

    Numeric columns are median-imputed and standardized; categorical columns
    are most-frequent-imputed and one-hot encoded with unknown categories
    ignored so unseen inference values do not break scoring.

    Args:
        X: Feature frame used to infer column roles.

    Returns:
        A ``ColumnTransformer`` ready to be fit within a pipeline.
    """
    numeric_columns = X.select_dtypes(include="number").columns.tolist()
    categorical_columns = [c for c in X.columns if c not in numeric_columns]

    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )


def build_estimator(config: ModelConfig) -> ClassifierMixin:
    """Instantiate the configured classifier.

    Args:
        config: Model configuration with estimator name and hyperparameters.

    Returns:
        An unfitted scikit-learn compatible classifier.

    Raises:
        ValueError: If the estimator name is not supported.
    """
    name = config.name.strip().lower()

    if name in {"logistic_regression", "logistic", "logreg"}:
        params = {"max_iter": 1000, "class_weight": "balanced", **config.params}
        return LogisticRegression(random_state=config.random_state, **params)

    if name in {"xgboost_classifier", "xgboost", "xgb"}:
        from xgboost import XGBClassifier

        params = {
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "eval_metric": "logloss",
            **config.params,
        }
        return XGBClassifier(random_state=config.random_state, **params)

    raise ValueError(f"Unsupported estimator: {config.name}")


def build_model_pipeline(X: pd.DataFrame, config: ModelConfig) -> Pipeline:
    """Assemble the preprocessing + estimator pipeline.

    Args:
        X: Feature frame used to infer preprocessing.
        config: Model configuration.

    Returns:
        An unfitted end-to-end pipeline.
    """
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(X)),
            ("model", build_estimator(config)),
        ]
    )


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: ModelConfig | None = None,
) -> Pipeline:
    """Train a churn classifier pipeline on the provided data.

    Args:
        X_train: Training feature frame.
        y_train: Encoded (0/1) training target.
        config: Optional model configuration; defaults to a balanced
            logistic regression baseline.

    Returns:
        The fitted pipeline.

    Raises:
        TypeError: If ``X_train`` is not a DataFrame.
        ValueError: If ``X_train`` is empty or lengths mismatch.
    """
    active_config = config or ModelConfig()

    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("`X_train` must be a pandas DataFrame.")
    if X_train.empty:
        raise ValueError("`X_train` must contain at least one row.")
    if len(X_train) != len(y_train):
        raise ValueError("`X_train` and `y_train` must have equal length.")

    pipeline = build_model_pipeline(X_train, active_config)
    LOGGER.info(
        "Training %s on %d rows, %d features.",
        active_config.name,
        X_train.shape[0],
        X_train.shape[1],
    )
    pipeline.fit(X_train, y_train)
    LOGGER.info("Model training complete.")
    return pipeline
