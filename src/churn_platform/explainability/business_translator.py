"""Translate SHAP feature contributions into plain-language retention drivers.

Turns the transformed feature names produced by the modeling pipeline (e.g.
``categorical__Contract_Month-to-month``) into readable statements a retention
analyst can act on, labelled by whether each driver raises or lowers churn risk.
"""

from __future__ import annotations

import logging

import pandas as pd

LOGGER = logging.getLogger(__name__)

__all__ = ["BusinessTranslator", "humanize_feature"]

# Prefixes injected by the ColumnTransformer branches.
_TRANSFORMER_PREFIXES = ("numeric__", "categorical__", "remainder__")


def humanize_feature(feature: str) -> str:
    """Convert a transformed feature name into a readable label.

    Args:
        feature: Transformed feature name from the pipeline.

    Returns:
        A human-readable label, e.g. ``"Contract = Month-to-month"`` or
        ``"Tenure Months"``.
    """
    name = feature
    is_categorical = False
    for prefix in _TRANSFORMER_PREFIXES:
        if name.startswith(prefix):
            is_categorical = prefix == "categorical__"
            name = name[len(prefix) :]
            break

    # Only one-hot (categorical) columns follow the "<Column>_<Value>" pattern;
    # numeric feature names may legitimately contain underscores and are kept
    # verbatim. Split on the last underscore so multi-word columns stay intact.
    if is_categorical and "_" in name:
        column, _, value = name.rpartition("_")
        if column:
            return f"{column} = {value}"
    return name


class BusinessTranslator:
    """Render SHAP explanations as business-readable churn drivers."""

    def __init__(self, positive_direction: str = "increases churn risk") -> None:
        """Initialize the translator.

        Args:
            positive_direction: Phrase describing a positive SHAP contribution.
        """
        self.positive_direction = positive_direction
        self.negative_direction = positive_direction.replace("increases", "reduces")
        self.logger = logging.getLogger(__name__)

    def translate(
        self, explanation: pd.DataFrame, top_n: int = 5
    ) -> list[dict[str, object]]:
        """Translate a per-customer SHAP explanation into readable drivers.

        Args:
            explanation: DataFrame with ``feature`` and ``shap_value`` columns
                (as returned by ``ChurnExplainer.explain_customer``).
            top_n: Maximum number of drivers to return.

        Returns:
            List of driver dictionaries ordered by absolute impact, each with
            ``feature``, ``readable``, ``direction``, and ``impact`` keys.

        Raises:
            ValueError: If required columns are missing or ``top_n`` < 1.
        """
        required = {"feature", "shap_value"}
        missing = required - set(explanation.columns)
        if missing:
            raise ValueError(f"Explanation missing required columns: {sorted(missing)}")
        if top_n < 1:
            raise ValueError("`top_n` must be >= 1.")

        ranked = explanation.reindex(
            explanation["shap_value"].abs().sort_values(ascending=False).index
        ).head(top_n)

        drivers: list[dict[str, object]] = []
        for _, row in ranked.iterrows():
            shap_value = float(row["shap_value"])
            direction = (
                self.positive_direction if shap_value > 0 else self.negative_direction
            )
            drivers.append(
                {
                    "feature": row["feature"],
                    "readable": humanize_feature(str(row["feature"])),
                    "direction": direction,
                    "impact": abs(shap_value),
                }
            )
        return drivers

    def summarize(self, explanation: pd.DataFrame, top_n: int = 3) -> list[str]:
        """Return a short list of sentence-form churn driver explanations.

        Args:
            explanation: Per-customer SHAP explanation.
            top_n: Number of drivers to include.

        Returns:
            List of readable sentences, e.g.
            ``"Contract = Month-to-month increases churn risk."``.
        """
        return [
            f"{driver['readable']} {driver['direction']}."
            for driver in self.translate(explanation, top_n=top_n)
        ]
