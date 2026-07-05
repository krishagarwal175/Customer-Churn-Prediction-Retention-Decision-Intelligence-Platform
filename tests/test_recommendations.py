"""Tests for retention recommendation rules and prioritization."""

from __future__ import annotations

import pandas as pd
import pytest

from churn_platform.recommendations.prioritization import (
    PrioritizationConfig,
    RetentionPrioritizer,
)
from churn_platform.recommendations.rules import RetentionRuleEngine


@pytest.fixture
def engine() -> RetentionRuleEngine:
    return RetentionRuleEngine(
        personas={
            "High-Value At-Risk": ["Priority retention call", "Loyalty discount"],
            "Low-Value Stable": ["Standard nurture"],
        },
        driver_rules=[
            {"match": "Contract = Month-to-month", "action": "Offer annual contract"},
            {"match": "Tech Support", "action": "Bundle tech support"},
        ],
    )


def test_recommend_merges_playbook_and_drivers(engine) -> None:
    actions = engine.recommend(
        "High-Value At-Risk",
        ["Contract = Month-to-month increases churn", "Tech Support = No"],
    )
    assert actions == [
        "Priority retention call",
        "Loyalty discount",
        "Offer annual contract",
        "Bundle tech support",
    ]


def test_recommend_dedupes_and_handles_unknown_persona(engine) -> None:
    assert engine.recommend("Unknown", []) == []
    # Driver appearing twice yields the action once.
    actions = engine.driver_actions(["Tech Support = No", "Tech Support = No"])
    assert actions == ["Bundle tech support"]


def test_from_config_builds_engine() -> None:
    config = {
        "recommendations": {
            "personas": {"P": ["Action A"]},
            "driver_rules": [{"match": "X", "action": "Do X"}],
        }
    }
    engine = RetentionRuleEngine.from_config(config)
    assert engine.recommend("P", ["contains X here"]) == ["Action A", "Do X"]


def test_invalid_driver_rule_rejected() -> None:
    with pytest.raises(ValueError):
        RetentionRuleEngine(personas={}, driver_rules=[{"match": "X"}])


@pytest.fixture
def simulated() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "expected_net_benefit": [500.0, 300.0, 100.0, -20.0, 50.0],
            "intervention_cost": [60.0, 60.0, 60.0, 60.0, 60.0],
        }
    )


def test_prioritize_ranks_by_net_benefit(simulated) -> None:
    result = RetentionPrioritizer().prioritize(simulated)
    ranked = result.dropna(subset=["priority_rank"]).sort_values("priority_rank")
    assert ranked["expected_net_benefit"].tolist() == [500.0, 300.0, 100.0, 50.0]
    # Negative net benefit is ineligible.
    assert result.loc[3, "selected"] == False  # noqa: E712


def test_prioritize_top_n_cap(simulated) -> None:
    result = RetentionPrioritizer(PrioritizationConfig(top_n=2)).prioritize(simulated)
    assert int(result["selected"].sum()) == 2
    assert result.loc[0, "selected"] and result.loc[1, "selected"]


def test_prioritize_budget_cap(simulated) -> None:
    # Budget for exactly 3 interventions at cost 60 each.
    result = RetentionPrioritizer(PrioritizationConfig(budget=180.0)).prioritize(
        simulated
    )
    assert int(result["selected"].sum()) == 3


def test_prioritize_is_non_mutating(simulated) -> None:
    original = simulated.copy(deep=True)
    RetentionPrioritizer().prioritize(simulated)
    pd.testing.assert_frame_equal(simulated, original)


def test_prioritize_missing_columns_raises() -> None:
    with pytest.raises(ValueError):
        RetentionPrioritizer().prioritize(pd.DataFrame({"x": [1]}))


def test_invalid_prioritization_config() -> None:
    with pytest.raises(ValueError):
        RetentionPrioritizer(PrioritizationConfig(top_n=0))
    with pytest.raises(ValueError):
        RetentionPrioritizer(PrioritizationConfig(budget=-5))
