"""Prioritize customers for retention outreach.

Ranks customers by expected net benefit (from the revenue simulation) and
selects an actionable shortlist under an optional team-capacity (top-N) and/or
budget cap. Only customers where acting is expected to pay off are eligible.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

LOGGER = logging.getLogger(__name__)

__all__ = ["PrioritizationConfig", "RetentionPrioritizer"]


@dataclass(frozen=True)
class PrioritizationConfig:
    """Configuration for outreach prioritization.

    Attributes:
        net_benefit_column: Column holding expected net benefit per customer.
        cost_column: Column holding per-customer intervention cost.
        top_n: Optional maximum number of customers to select.
        budget: Optional total spend cap across selected customers.
        min_net_benefit: Minimum net benefit required to be eligible.
    """

    net_benefit_column: str = "expected_net_benefit"
    cost_column: str = "intervention_cost"
    top_n: int | None = None
    budget: float | None = None
    min_net_benefit: float = 0.0


class RetentionPrioritizer:
    """Rank and select customers for a retention campaign."""

    def __init__(self, config: PrioritizationConfig | None = None) -> None:
        """Initialize the prioritizer.

        Args:
            config: Prioritization configuration.

        Raises:
            ValueError: If ``top_n`` or ``budget`` are non-positive.
        """
        self.config = config or PrioritizationConfig()
        if self.config.top_n is not None and self.config.top_n <= 0:
            raise ValueError("`top_n` must be positive when set.")
        if self.config.budget is not None and self.config.budget <= 0:
            raise ValueError("`budget` must be positive when set.")
        self.logger = logging.getLogger(__name__)

    def prioritize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rank customers and flag the selected outreach shortlist.

        Args:
            df: Simulated customer frame containing the net-benefit and cost
                columns.

        Returns:
            Copy of ``df`` sorted by descending net benefit with added
            ``priority_rank`` (1-based over eligible customers) and ``selected``
            columns. Ineligible customers have ``priority_rank`` of ``<NA>`` and
            ``selected`` of ``False``. Input is not mutated.

        Raises:
            ValueError: If required columns are missing.
        """
        config = self.config
        required = [config.net_benefit_column, config.cost_column]
        missing = [column for column in required if column not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        result = df.copy(deep=True).sort_values(
            config.net_benefit_column, ascending=False, kind="stable"
        )

        eligible = result[config.net_benefit_column] > config.min_net_benefit
        result["priority_rank"] = (
            pd.Series(range(1, int(eligible.sum()) + 1), index=result.index[eligible])
            .reindex(result.index)
            .astype("Int64")
        )

        selected = eligible.copy()
        if config.top_n is not None:
            selected &= result["priority_rank"].le(config.top_n).fillna(False)
        if config.budget is not None:
            cumulative_cost = result[config.cost_column].where(selected, 0).cumsum()
            selected &= cumulative_cost.le(config.budget)

        result["selected"] = selected
        self.logger.info(
            "Prioritized %d eligible customers; %d selected.",
            int(eligible.sum()),
            int(selected.sum()),
        )
        return result
