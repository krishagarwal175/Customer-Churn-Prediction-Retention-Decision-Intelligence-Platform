"""Tests for the churn modeling pipeline: train, evaluate, calibrate, predict."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churn_platform.models.calibration import calibrate_classifier
from churn_platform.models.evaluation import evaluate_classifier, recall_at_precision
from churn_platform.models.prediction import ChurnPredictor
from churn_platform.models.training import (
    ModelConfig,
    encode_target,
    train_model,
)


@pytest.fixture
def dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Build a small separable churn-like dataset with mixed dtypes."""
    rng = np.random.default_rng(42)
    n = 200
    tenure = rng.integers(0, 72, size=n)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], size=n)
    monthly = rng.uniform(20, 120, size=n)

    # Churn more likely for short tenure + month-to-month + high charges.
    logit = -0.05 * tenure + 0.02 * monthly + (contract == "Month-to-month") * 1.5
    churn_prob = 1 / (1 + np.exp(-(logit - logit.mean())))
    churn = np.where(rng.uniform(size=n) < churn_prob, "Yes", "No")

    X = pd.DataFrame(
        {
            "Tenure Months": tenure,
            "Monthly Charges": monthly,
            "Contract": contract,
        }
    )
    y = pd.Series(churn, name="Churn Label")
    return X, y


def test_encode_target_maps_positive_label() -> None:
    y = pd.Series(["Yes", "No", " yes ", "No"])
    encoded = encode_target(y)
    assert encoded.tolist() == [1, 0, 1, 0]


def test_train_model_returns_fitted_pipeline(dataset) -> None:
    X, y = dataset
    model = train_model(X, encode_target(y))
    assert hasattr(model, "predict_proba")
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)


def test_train_model_rejects_bad_input() -> None:
    with pytest.raises(TypeError):
        train_model(None, pd.Series([0, 1]))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        train_model(pd.DataFrame({"a": []}), pd.Series([], dtype=int))


def test_evaluate_classifier_returns_bounded_metrics(dataset) -> None:
    X, y = dataset
    y_enc = encode_target(y)
    model = train_model(X, y_enc)
    metrics = evaluate_classifier(model, X, y_enc)

    expected_keys = {
        "roc_auc",
        "pr_auc",
        "precision",
        "recall",
        "f1",
        "brier_score",
        "recall_at_precision_floor",
    }
    assert expected_keys <= set(metrics)
    for key in expected_keys - {"brier_score"}:
        assert 0.0 <= metrics[key] <= 1.0
    # Model should learn the injected signal.
    assert metrics["roc_auc"] > 0.7


def test_recall_at_precision_floor_edge_cases() -> None:
    y_true = np.array([0, 0, 1, 1])
    perfect = np.array([0.1, 0.2, 0.8, 0.9])
    assert recall_at_precision(y_true, perfect, 0.9) == pytest.approx(1.0)
    # Unreachable precision floor -> zero recall.
    inverted = np.array([0.9, 0.8, 0.2, 0.1])
    assert recall_at_precision(y_true, inverted, 1.0) == 0.0
    with pytest.raises(ValueError):
        recall_at_precision(y_true, perfect, 1.5)


def test_xgboost_estimator_trains(dataset) -> None:
    X, y = dataset
    model = train_model(X, encode_target(y), ModelConfig(name="xgboost_classifier"))
    assert model.predict_proba(X).shape == (len(X), 2)


def test_calibration_produces_probabilities(dataset) -> None:
    X, y = dataset
    y_enc = encode_target(y)
    model = train_model(X.iloc[:150], y_enc.iloc[:150])
    calibrated = calibrate_classifier(model, X.iloc[150:], y_enc.iloc[150:])
    proba = calibrated.predict_proba(X.iloc[150:])[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()


def test_predictor_predict_and_roundtrip(dataset, tmp_path) -> None:
    X, y = dataset
    model = train_model(X, encode_target(y))
    predictor = ChurnPredictor(model, threshold=0.5)

    labels = predictor.predict(X)
    assert set(np.unique(labels)) <= {0, 1}

    path = predictor.save(tmp_path / "predictor.joblib")
    restored = ChurnPredictor.load(path)
    np.testing.assert_array_equal(restored.predict(X), labels)


def test_predictor_rejects_uncalibratable_model() -> None:
    class NoProba:
        pass

    with pytest.raises(AttributeError):
        ChurnPredictor(NoProba())
