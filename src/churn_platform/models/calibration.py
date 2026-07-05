"""Probability calibration for churn classifiers.

Well-calibrated probabilities matter for retention economics: expected-value
targeting and revenue simulation rely on scores that behave like true churn
probabilities. This wraps a fitted classifier with post-hoc calibration.
"""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

LOGGER = logging.getLogger(__name__)

__all__ = ["calibrate_classifier"]

_VALID_METHODS = {"isotonic", "sigmoid"}


def calibrate_classifier(
    fitted_model: object,
    X_calibration: pd.DataFrame,
    y_calibration: pd.Series,
    method: str = "isotonic",
) -> CalibratedClassifierCV:
    """Calibrate a pre-fitted classifier on a held-out calibration set.

    Args:
        fitted_model: An already-fitted classifier exposing ``predict_proba``.
        X_calibration: Calibration feature frame (disjoint from training).
        y_calibration: Encoded (0/1) calibration target.
        method: Calibration method, ``"isotonic"`` or ``"sigmoid"``.

    Returns:
        A fitted ``CalibratedClassifierCV`` wrapping ``fitted_model``.

    Raises:
        ValueError: If ``method`` is unsupported or inputs are empty/mismatched.
    """
    if method not in _VALID_METHODS:
        raise ValueError(
            f"Unsupported calibration method '{method}'. "
            f"Choose one of {sorted(_VALID_METHODS)}."
        )
    if X_calibration.empty:
        raise ValueError("`X_calibration` must contain at least one row.")
    if len(X_calibration) != len(y_calibration):
        raise ValueError("Calibration features and target must have equal length.")

    # A pre-fitted estimator is frozen so calibration does not refit it.
    calibrated = CalibratedClassifierCV(FrozenEstimator(fitted_model), method=method)
    calibrated.fit(X_calibration, y_calibration)
    LOGGER.info("Calibrated model using '%s' on %d rows.", method, len(X_calibration))
    return calibrated
