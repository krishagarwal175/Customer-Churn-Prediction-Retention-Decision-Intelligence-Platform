"""SHAP-based explainability for churn model pipelines.

Explains predictions from a fitted ``preprocess -> model`` pipeline. SHAP values
are computed on the transformed feature space and mapped back to readable
transformed feature names, so both global driver importance and per-customer
explanations are available regardless of the underlying estimator.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

LOGGER = logging.getLogger(__name__)

__all__ = ["ChurnExplainer"]

_MAX_BACKGROUND = 100


class ChurnExplainer:
    """Compute SHAP explanations for a fitted churn pipeline."""

    def __init__(self, pipeline: Pipeline, background: pd.DataFrame) -> None:
        """Initialize the explainer.

        Args:
            pipeline: Fitted pipeline whose last step is the estimator and whose
                preceding steps transform the raw feature frame.
            background: Representative feature frame used as the SHAP reference
                distribution (typically a sample of the training data).

        Raises:
            TypeError: If ``pipeline`` is not a scikit-learn Pipeline.
            ValueError: If ``background`` is empty.
        """
        if not isinstance(pipeline, Pipeline):
            raise TypeError("`pipeline` must be a scikit-learn Pipeline.")
        if background.empty:
            raise ValueError("`background` must contain at least one row.")

        self.logger = logging.getLogger(__name__)
        self._preprocessor = pipeline[:-1]
        self._model = pipeline[-1]

        self.feature_names: list[str] = list(self._preprocessor.get_feature_names_out())

        background_t = np.asarray(self._preprocessor.transform(background))
        reference = shap.utils.sample(
            background_t, min(_MAX_BACKGROUND, background_t.shape[0]), random_state=42
        )
        self._explainer = self._build_explainer(reference)

    def _build_explainer(self, reference: np.ndarray) -> shap.Explainer:
        """Select a SHAP explainer suited to the estimator.

        Tree ensembles use the exact path-dependent TreeExplainer (no masker);
        other estimators fall back to the masker-based unified explainer.
        """
        try:
            return shap.TreeExplainer(self._model)
        except Exception:  # noqa: BLE001 - shap raises assorted errors by model
            return shap.Explainer(self._model, reference)

    def _shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Return a 2-D array of SHAP values (n_samples, n_features)."""
        transformed = np.asarray(self._preprocessor.transform(X))
        values = self._explainer(transformed).values
        if values.ndim == 3:
            # (n_samples, n_features, n_classes) -> positive class.
            values = values[:, :, -1]
        return np.asarray(values)

    def global_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        """Rank features by mean absolute SHAP value across ``X``.

        Args:
            X: Feature frame to explain.

        Returns:
            DataFrame with columns ``feature`` and ``mean_abs_shap``, sorted
            descending by importance.
        """
        values = self._shap_values(X)
        importance = np.abs(values).mean(axis=0)
        return pd.DataFrame(
            {"feature": self.feature_names, "mean_abs_shap": importance}
        ).sort_values("mean_abs_shap", ascending=False, ignore_index=True)

    def explain_customer(self, customer: pd.DataFrame) -> pd.DataFrame:
        """Explain a single customer's churn prediction.

        Args:
            customer: Single-row feature frame.

        Returns:
            DataFrame with ``feature``, ``shap_value`` (signed; positive pushes
            toward churn), and ``abs_shap``, sorted by absolute contribution.

        Raises:
            ValueError: If ``customer`` does not contain exactly one row.
        """
        if len(customer) != 1:
            raise ValueError("`customer` must contain exactly one row.")

        values = self._shap_values(customer)[0]
        return pd.DataFrame(
            {
                "feature": self.feature_names,
                "shap_value": values,
                "abs_shap": np.abs(values),
            }
        ).sort_values("abs_shap", ascending=False, ignore_index=True)
