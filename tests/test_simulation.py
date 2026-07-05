"""Tests for retention revenue simulation and sensitivity analysis."""

from __future__ import annotations

import pandas as pd
import pytest

from churn_platform.simulation.revenue import (
    RetentionEconomicsConfig,
    RevenueSimulator,
)
from churn_platform.simulation.sensitivity import SensitivityAnalyzer


@pytest.fixture
def customers() -> pd.DataFrame:
    """Customers with churn probabilities, value, and segments."""
    return pd.DataFrame(
        {
            "churn_probability": [0.8, 0.5, 0.1, 0.6],
            "CLTV": [1000.0, 2000.0, 5000.0, 800.0],
            "segment": ["A", "A", "B", "B"],
        }
    )


def test_simulate_computes_expected_value(customers) -> None:
    config = RetentionEconomicsConfig(default_uplift=0.5, default_cost=100.0)
    result = RevenueSimulator(config).simulate(customers)

    # Row 0: 0.8 * 0.5 * 1000 = 400 saved, minus 100 cost = 300 net.
    assert result.loc[0, "expected_saved_revenue"] == pytest.approx(400.0)
    assert result.loc[0, "expected_net_benefit"] == pytest.approx(300.0)
    assert bool(result.loc[0, "target"]) is True
    # Row 2: 0.1 * 0.5 * 5000 = 250 saved, minus 100 = 150 net (still positive).
    assert bool(result.loc[2, "target"]) is True


def test_simulate_is_non_mutating(customers) -> None:
    original = customers.copy(deep=True)
    RevenueSimulator().simulate(customers)
    pd.testing.assert_frame_equal(customers, original)


def test_segment_specific_uplift_and_cost(customers) -> None:
    config = RetentionEconomicsConfig(
        default_uplift=0.2,
        default_cost=50.0,
        segment_uplift={"A": 0.5},
        segment_cost={"B": 500.0},
    )
    result = RevenueSimulator(config).simulate(customers)

    assert result.loc[0, "campaign_uplift"] == pytest.approx(0.5)  # segment A
    assert result.loc[2, "campaign_uplift"] == pytest.approx(0.2)  # default
    assert result.loc[2, "intervention_cost"] == pytest.approx(500.0)  # segment B


def test_campaign_summary_targets_positive_net(customers) -> None:
    config = RetentionEconomicsConfig(default_uplift=0.5, default_cost=100.0)
    summary = RevenueSimulator(config).campaign_summary(customers)

    assert summary["customers_total"] == 4
    assert 0 <= summary["customers_targeted"] <= 4
    assert summary["total_expected_net_benefit"] >= 0
    if summary["total_intervention_cost"] > 0:
        assert summary["roi"] >= 0


def test_invalid_config_rejected() -> None:
    with pytest.raises(ValueError):
        RevenueSimulator(RetentionEconomicsConfig(default_uplift=1.5))
    with pytest.raises(ValueError):
        RevenueSimulator(RetentionEconomicsConfig(default_cost=-1))


def test_simulate_missing_columns_raises() -> None:
    with pytest.raises(ValueError):
        RevenueSimulator().simulate(pd.DataFrame({"churn_probability": [0.5]}))


def test_sensitivity_grid_covers_all_combinations(customers) -> None:
    analyzer = SensitivityAnalyzer(customers)
    grid = analyzer.run_grid(
        {"default_uplift": [0.2, 0.4], "default_cost": [50.0, 150.0]}
    )

    assert len(grid) == 4  # 2 x 2 full grid
    assert {"default_uplift", "default_cost", "roi"} <= set(grid.columns)
    # Higher cost never increases net benefit at fixed uplift.
    low_cost = grid[grid["default_cost"] == 50.0]["total_expected_net_benefit"].sum()
    high_cost = grid[grid["default_cost"] == 150.0]["total_expected_net_benefit"].sum()
    assert low_cost >= high_cost


def test_uplift_scale_moves_segment_economics(customers) -> None:
    config = RetentionEconomicsConfig(
        default_cost=50.0, segment_uplift={"A": 0.4, "B": 0.4}
    )
    base = RevenueSimulator(config).simulate(customers)
    scaled = RevenueSimulator(
        RetentionEconomicsConfig(
            default_cost=50.0, segment_uplift={"A": 0.4, "B": 0.4}, uplift_scale=0.5
        )
    ).simulate(customers)
    # Halving the scale halves the saved revenue even with segment overrides.
    assert scaled.loc[0, "expected_saved_revenue"] == pytest.approx(
        base.loc[0, "expected_saved_revenue"] * 0.5
    )


def test_uplift_scale_clips_to_one(customers) -> None:
    config = RetentionEconomicsConfig(default_uplift=0.8, uplift_scale=10.0)
    result = RevenueSimulator(config).simulate(customers)
    assert (result["campaign_uplift"] <= 1.0).all()


def test_sensitivity_grid_over_scales_is_meaningful(customers) -> None:
    # Segment economics cover all rows; scale sweeps still move the result.
    config = RetentionEconomicsConfig(segment_uplift={"A": 0.3, "B": 0.3})
    grid = SensitivityAnalyzer(customers, config).run_grid(
        {"uplift_scale": [0.5, 1.0], "cost_scale": [1.0, 2.0]}
    )
    assert len(grid) == 4
    assert grid["total_expected_saved_revenue"].nunique() > 1


def test_sensitivity_rejects_unknown_field(customers) -> None:
    with pytest.raises(ValueError):
        SensitivityAnalyzer(customers).run_grid({"not_a_field": [1, 2]})
