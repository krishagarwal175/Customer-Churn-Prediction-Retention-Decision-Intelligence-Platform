"""Model evaluation metrics for churn classifiers.

Computes standard discrimination and calibration metrics plus the project's
primary business metric: the maximum recall achievable while holding precision
at or above a configured floor. All functions are computation-only.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

LOGGER = logging.getLogger(__name__)

__all__ = ["recall_at_precision", "evaluate_classifier"]


def recall_at_precision(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    precision_floor: float = 0.45,
) -> float:
    """Return the max recall achievable while precision >= ``precision_floor``.

    This reflects the retention use case: catch as many churners as possible
    without letting campaign precision drop below an actionable floor.

    Args:
        y_true: Ground-truth binary labels (1 = churn).
        y_score: Predicted positive-class probabilities or scores.
        precision_floor: Minimum acceptable precision in ``[0, 1]``.

    Returns:
        Best feasible recall, or ``0.0`` if the floor is never met.

    Raises:
        ValueError: If ``precision_floor`` is outside ``[0, 1]``.
    """
    if not 0.0 <= precision_floor <= 1.0:
        raise ValueError("`precision_floor` must be within [0, 1].")

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    feasible = recall[precision >= precision_floor]
    if feasible.size == 0:
        return 0.0
    return float(feasible.max())


def evaluate_classifier(
    model: Any,
    X: pd.DataFrame,
    y_true: pd.Series | np.ndarray,
    threshold: float = 0.5,
    precision_floor: float = 0.45,
) -> dict[str, float]:
    """Evaluate a fitted classifier on a holdout set.

    Args:
        model: Fitted estimator exposing ``predict_proba``.
        X: Feature frame for evaluation.
        y_true: Ground-truth binary labels.
        threshold: Decision threshold for hard-label metrics.
        precision_floor: Floor for the primary recall-at-precision metric.

    Returns:
        Dictionary of metric name to value.

    Raises:
        AttributeError: If the model does not support ``predict_proba``.
    """
    if not hasattr(model, "predict_proba"):
        raise AttributeError("Model must implement `predict_proba` for evaluation.")

    y_score = model.predict_proba(X)[:, 1]
    y_pred = (y_score >= threshold).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, y_score)),
        "recall_at_precision_floor": recall_at_precision(
            y_true, y_score, precision_floor
        ),
    }

    LOGGER.info(
        "Evaluation: roc_auc=%.3f pr_auc=%.3f recall@p>=%.2f=%.3f",
        metrics["roc_auc"],
        metrics["pr_auc"],
        precision_floor,
        metrics["recall_at_precision_floor"],
    )
    return metrics
