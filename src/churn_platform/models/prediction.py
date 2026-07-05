"""Churn scoring and model persistence.

Wraps a fitted classifier (or calibrated wrapper) to produce churn
probabilities and thresholded decisions, and to persist/load the artifact.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

LOGGER = logging.getLogger(__name__)

__all__ = ["ChurnPredictor"]


class ChurnPredictor:
    """Score customers with a fitted churn model."""

    def __init__(self, model: object, threshold: float = 0.5) -> None:
        """Initialize the predictor.

        Args:
            model: Fitted estimator exposing ``predict_proba``.
            threshold: Default decision threshold for hard labels.

        Raises:
            AttributeError: If the model lacks ``predict_proba``.
            ValueError: If the threshold is outside ``[0, 1]``.
        """
        if not hasattr(model, "predict_proba"):
            raise AttributeError("`model` must implement `predict_proba`.")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("`threshold` must be within [0, 1].")

        self.model = model
        self.threshold = threshold
        self.logger = logging.getLogger(__name__)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return positive-class (churn) probabilities.

        Args:
            X: Feature frame to score.

        Returns:
            1-D array of churn probabilities.
        """
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float | None = None) -> np.ndarray:
        """Return hard churn labels at the given (or default) threshold.

        Args:
            X: Feature frame to score.
            threshold: Optional override for the decision threshold.

        Returns:
            1-D integer array of predicted labels (1 = churn).
        """
        cutoff = self.threshold if threshold is None else threshold
        if not 0.0 <= cutoff <= 1.0:
            raise ValueError("`threshold` must be within [0, 1].")
        return (self.predict_proba(X) >= cutoff).astype(int)

    def save(self, path: str | Path) -> Path:
        """Persist the predictor to disk.

        Args:
            path: Destination file path (``.joblib``).

        Returns:
            The resolved path written.
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "threshold": self.threshold}, destination)
        self.logger.info("Saved predictor to %s", destination)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> ChurnPredictor:
        """Load a persisted predictor from disk.

        Args:
            path: Source ``.joblib`` file path.

        Returns:
            A restored ``ChurnPredictor``.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"Predictor artifact not found: {source}")
        payload = joblib.load(source)
        return cls(model=payload["model"], threshold=payload["threshold"])
