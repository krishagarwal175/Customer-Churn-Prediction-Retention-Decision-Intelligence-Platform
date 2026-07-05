"""Assign business persona labels to customer segments.

Maps a numeric segment profile (as produced by ``CustomerSegmenter.profile``)
onto interpretable persona names using a value axis and a risk axis, each split
at its median across segments. Purely rule-based and computation-only.
"""

from __future__ import annotations

import logging

import pandas as pd

LOGGER = logging.getLogger(__name__)

__all__ = ["PersonaLabeler", "DEFAULT_PERSONA_NAMES"]

# Persona names keyed by (high_value, high_risk).
DEFAULT_PERSONA_NAMES: dict[tuple[bool, bool], str] = {
    (True, True): "High-Value At-Risk",
    (True, False): "High-Value Loyal",
    (False, True): "Low-Value At-Risk",
    (False, False): "Low-Value Stable",
}

_PERSONA_COLUMN = "persona"


class PersonaLabeler:
    """Label segment profiles with business personas."""

    def __init__(
        self,
        value_metric: str,
        risk_metric: str,
        persona_names: dict[tuple[bool, bool], str] | None = None,
    ) -> None:
        """Initialize the labeler.

        Args:
            value_metric: Profile column measuring customer value (higher is
                more valuable), e.g. ``"Monthly Charges"`` or ``"CLTV"``.
            risk_metric: Profile column measuring churn risk (higher is
                riskier), e.g. ``"churn_rate"``.
            persona_names: Optional override for the quadrant-to-name mapping.
        """
        self.value_metric = value_metric
        self.risk_metric = risk_metric
        self.persona_names = persona_names or DEFAULT_PERSONA_NAMES
        self.logger = logging.getLogger(__name__)

    def label(self, profile: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of ``profile`` with a ``persona`` column.

        Value and risk are each split at their median across segments. Segments
        at or above the median are treated as high value / high risk.

        Args:
            profile: Segment profile indexed by segment id.

        Returns:
            New DataFrame with an added ``persona`` column; input not mutated.

        Raises:
            ValueError: If the value or risk metric columns are missing.
        """
        missing = [
            column
            for column in (self.value_metric, self.risk_metric)
            if column not in profile.columns
        ]
        if missing:
            raise ValueError(f"Profile missing required metric columns: {missing}")

        result = profile.copy(deep=True)
        value_threshold = result[self.value_metric].median()
        risk_threshold = result[self.risk_metric].median()

        high_value = result[self.value_metric] >= value_threshold
        high_risk = result[self.risk_metric] >= risk_threshold

        result[_PERSONA_COLUMN] = [
            self.persona_names[(bool(v), bool(r))]
            for v, r in zip(high_value, high_risk)
        ]
        self.logger.info("Labelled %d segments with personas.", len(result))
        return result
