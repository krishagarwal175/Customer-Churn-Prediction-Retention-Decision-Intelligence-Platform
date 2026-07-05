"""Retention revenue simulation via expected-value targeting.

Estimates the financial impact of a retention campaign at the customer level:

    expected saved revenue = P(churn) x customer value x campaign uplift
    expected net benefit   = expected saved revenue - intervention cost

Campaign uplift and cost are resolved per customer segment (with global
defaults), so retention economics can differ across personas. Computation only:
inputs are never mutated.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

LOGGER = logging.getLogger(__name__)

__all__ = ["RetentionEconomicsConfig", "RevenueSimulator"]


@dataclass(frozen=True)
class RetentionEconomicsConfig:
    """Economic assumptions for a retention campaign.

    Attributes:
        value_column: Column holding the revenue at stake per customer.
        churn_probability_column: Column with predicted churn probabilities.
        segment_column: Optional column used to resolve per-segment economics.
        default_uplift: Fraction of churn probability a campaign averts.
        default_cost: Per-customer intervention cost.
        segment_uplift: Optional per-segment uplift overrides.
        segment_cost: Optional per-segment cost overrides.
        uplift_scale: Global multiplier applied to every resolved uplift
            (default and per-segment); the effective uplift is clipped to
            ``[0, 1]``. Enables sensitivity sweeps over segment-specific economics.
        cost_scale: Global multiplier applied to every resolved cost.
    """

    value_column: str = "CLTV"
    churn_probability_column: str = "churn_probability"
    segment_column: str | None = "segment"
    default_uplift: float = 0.30
    default_cost: float = 50.0
    segment_uplift: Mapping[Any, float] = field(default_factory=dict)
    segment_cost: Mapping[Any, float] = field(default_factory=dict)
    uplift_scale: float = 1.0
    cost_scale: float = 1.0


class RevenueSimulator:
    """Simulate expected retention revenue and campaign economics."""

    def __init__(self, config: RetentionEconomicsConfig | None = None) -> None:
        """Initialize the simulator.

        Args:
            config: Retention economics configuration.

        Raises:
            ValueError: If default uplift or cost are out of range.
        """
        self.config = config or RetentionEconomicsConfig()
        if not 0.0 <= self.config.default_uplift <= 1.0:
            raise ValueError("`default_uplift` must be within [0, 1].")
        if self.config.default_cost < 0:
            raise ValueError("`default_cost` must be non-negative.")
        self.logger = logging.getLogger(__name__)

    def _resolve_per_segment(
        self, df: pd.DataFrame, overrides: Mapping[Any, float], default: float
    ) -> pd.Series:
        """Return a per-row Series of a parameter using segment overrides."""
        segment_column = self.config.segment_column
        if not segment_column or segment_column not in df.columns or not overrides:
            return pd.Series(default, index=df.index)
        return df[segment_column].map(overrides).fillna(default).astype(float)

    def simulate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute expected retention economics per customer.

        Args:
            df: Customer frame with churn probability and value columns (and an
                optional segment column).

        Returns:
            Copy of ``df`` with ``campaign_uplift``, ``intervention_cost``,
            ``expected_saved_revenue``, ``expected_net_benefit`` and ``target``
            columns added. ``target`` is True when net benefit is positive.

        Raises:
            ValueError: If required columns are missing.
        """
        required = [
            self.config.churn_probability_column,
            self.config.value_column,
        ]
        missing = [column for column in required if column not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        result = df.copy(deep=True)
        churn_prob = result[self.config.churn_probability_column].astype(float)
        value = result[self.config.value_column].astype(float)

        uplift = (
            self._resolve_per_segment(
                result, self.config.segment_uplift, self.config.default_uplift
            )
            * self.config.uplift_scale
        ).clip(0.0, 1.0)
        cost = (
            self._resolve_per_segment(
                result, self.config.segment_cost, self.config.default_cost
            )
            * self.config.cost_scale
        )

        saved = churn_prob * uplift * value
        net = saved - cost

        result["campaign_uplift"] = uplift
        result["intervention_cost"] = cost
        result["expected_saved_revenue"] = saved
        result["expected_net_benefit"] = net
        result["target"] = net > 0
        return result

    def campaign_summary(self, df: pd.DataFrame) -> dict[str, float]:
        """Summarize campaign economics over targeted customers.

        Only customers flagged ``target`` incur cost and contribute benefit,
        reflecting a campaign that targets positive-net-benefit customers.

        Args:
            df: Customer frame (raw or already simulated).

        Returns:
            Dictionary with campaign totals and ROI.
        """
        simulated = df if "target" in df.columns else self.simulate(df)
        targeted = simulated[simulated["target"]]

        total_cost = float(targeted["intervention_cost"].sum())
        total_saved = float(targeted["expected_saved_revenue"].sum())
        total_net = float(targeted["expected_net_benefit"].sum())
        roi = float(total_net / total_cost) if total_cost > 0 else 0.0

        return {
            "customers_total": int(len(simulated)),
            "customers_targeted": int(len(targeted)),
            "total_intervention_cost": total_cost,
            "total_expected_saved_revenue": total_saved,
            "total_expected_net_benefit": total_net,
            "roi": roi,
        }
