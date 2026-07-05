"""Grid sensitivity analysis for retention campaign economics.

Sweeps combinations of economic assumptions (e.g. uplift and cost) across a
full parameter grid, re-running the revenue simulation for each combination so
the joint effect of the assumptions on campaign ROI can be inspected.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import fields, replace
from itertools import product

import pandas as pd

from churn_platform.simulation.revenue import (
    RetentionEconomicsConfig,
    RevenueSimulator,
)

LOGGER = logging.getLogger(__name__)

__all__ = ["SensitivityAnalyzer"]


class SensitivityAnalyzer:
    """Run full-grid sensitivity sweeps over campaign economics."""

    def __init__(
        self, df: pd.DataFrame, base_config: RetentionEconomicsConfig | None = None
    ) -> None:
        """Initialize the analyzer.

        Args:
            df: Customer frame with churn probability and value columns.
            base_config: Baseline economics held fixed except for swept params.
        """
        self.df = df
        self.base_config = base_config or RetentionEconomicsConfig()
        self.logger = logging.getLogger(__name__)
        self._config_fields = {f.name for f in fields(RetentionEconomicsConfig)}

    def run_grid(self, param_grid: dict[str, Sequence[float]]) -> pd.DataFrame:
        """Evaluate campaign economics across the full parameter grid.

        Args:
            param_grid: Mapping of ``RetentionEconomicsConfig`` field name to the
                sequence of values to sweep (e.g.
                ``{"default_uplift": [0.2, 0.3], "default_cost": [30, 50]}``).

        Returns:
            One row per parameter combination: the swept values alongside the
            resulting campaign-summary metrics.

        Raises:
            ValueError: If ``param_grid`` is empty or names an unknown field.
        """
        if not param_grid:
            raise ValueError("`param_grid` must contain at least one parameter.")

        unknown = set(param_grid) - self._config_fields
        if unknown:
            raise ValueError(f"Unknown config fields in grid: {sorted(unknown)}")

        keys = list(param_grid)
        rows: list[dict[str, float]] = []
        for combination in product(*(param_grid[key] for key in keys)):
            overrides = dict(zip(keys, combination))
            config = replace(self.base_config, **overrides)
            summary = RevenueSimulator(config).campaign_summary(self.df)
            rows.append({**overrides, **summary})

        self.logger.info(
            "Sensitivity grid evaluated %d combinations over %s.",
            len(rows),
            keys,
        )
        return pd.DataFrame(rows)
