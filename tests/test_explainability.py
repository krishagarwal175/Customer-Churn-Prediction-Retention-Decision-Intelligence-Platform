"""Tests for SHAP explainability and business translation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churn_platform.explainability.business_translator import (
    BusinessTranslator,
    humanize_feature,
)
from churn_platform.explainability.shap_explainer import ChurnExplainer
from churn_platform.models.training import ModelConfig, encode_target, train_model


@pytest.fixture
def fitted_pipeline() -> tuple[object, pd.DataFrame]:
    """Train a small pipeline and return it with its feature frame."""
    rng = np.random.default_rng(0)
    n = 150
    tenure = rng.integers(0, 72, size=n)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], size=n)
    monthly = rng.uniform(20, 120, size=n)
    logit = -0.06 * tenure + 0.03 * monthly + (contract == "Month-to-month") * 1.5
    churn = np.where(
        rng.uniform(size=n) < 1 / (1 + np.exp(-(logit - logit.mean()))), "Yes", "No"
    )
    X = pd.DataFrame(
        {"Tenure Months": tenure, "Monthly Charges": monthly, "Contract": contract}
    )
    y = encode_target(pd.Series(churn, name="Churn Label"))
    model = train_model(X, y, ModelConfig(name="xgboost_classifier"))
    return model, X


def test_humanize_feature_variants() -> None:
    assert humanize_feature("numeric__Tenure Months") == "Tenure Months"
    assert (
        humanize_feature("categorical__Contract_Month-to-month")
        == "Contract = Month-to-month"
    )


def test_global_importance_ranks_features(fitted_pipeline) -> None:
    model, X = fitted_pipeline
    explainer = ChurnExplainer(model, background=X)
    importance = explainer.global_importance(X)

    assert list(importance.columns) == ["feature", "mean_abs_shap"]
    assert len(importance) == len(explainer.feature_names)
    # Monotonic non-increasing ranking.
    assert importance["mean_abs_shap"].is_monotonic_decreasing
    assert (importance["mean_abs_shap"] >= 0).all()


def test_explain_customer_shapes(fitted_pipeline) -> None:
    model, X = fitted_pipeline
    explainer = ChurnExplainer(model, background=X)
    explanation = explainer.explain_customer(X.iloc[[0]])

    assert set(explanation.columns) == {"feature", "shap_value", "abs_shap"}
    assert len(explanation) == len(explainer.feature_names)
    assert explanation["abs_shap"].is_monotonic_decreasing


def test_explain_customer_requires_single_row(fitted_pipeline) -> None:
    model, X = fitted_pipeline
    explainer = ChurnExplainer(model, background=X)
    with pytest.raises(ValueError):
        explainer.explain_customer(X.iloc[:2])


def test_explainer_rejects_non_pipeline() -> None:
    with pytest.raises(TypeError):
        ChurnExplainer(object(), background=pd.DataFrame({"a": [1]}))  # type: ignore


def test_translator_produces_directional_drivers() -> None:
    explanation = pd.DataFrame(
        {
            "feature": [
                "categorical__Contract_Month-to-month",
                "numeric__Tenure Months",
                "numeric__Monthly Charges",
            ],
            "shap_value": [0.8, -0.5, 0.2],
        }
    )
    drivers = BusinessTranslator().translate(explanation, top_n=2)

    assert len(drivers) == 2
    assert drivers[0]["readable"] == "Contract = Month-to-month"
    assert drivers[0]["direction"] == "increases churn risk"
    assert drivers[1]["direction"] == "reduces churn risk"


def test_translator_summarize_sentences() -> None:
    explanation = pd.DataFrame(
        {"feature": ["numeric__Tenure Months"], "shap_value": [-0.4]}
    )
    sentences = BusinessTranslator().summarize(explanation, top_n=1)
    assert sentences == ["Tenure Months reduces churn risk."]


def test_translator_validates_input() -> None:
    with pytest.raises(ValueError):
        BusinessTranslator().translate(pd.DataFrame({"feature": ["x"]}))
