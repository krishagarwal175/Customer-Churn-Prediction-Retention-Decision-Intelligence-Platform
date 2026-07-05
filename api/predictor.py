"""Lightweight, dependency-free live churn scoring.

Reproduces the offline pipeline for a single customer using only the exported
model spec and NumPy: feature engineering, preprocessing (impute/scale/one-hot),
a logistic score, signed churn drivers, nearest-centroid persona assignment, and
hybrid retention recommendations. No scikit-learn/xgboost/shap at runtime.
"""

from __future__ import annotations

import math
from typing import Any

from api import data

_TRANSFORMER_PREFIXES = ("numeric__", "categorical__", "remainder__")


def humanize(feature: str) -> str:
    """Readable label for a transformed feature name (mirrors the UI translator)."""
    name = feature
    is_categorical = False
    for prefix in _TRANSFORMER_PREFIXES:
        if name.startswith(prefix):
            is_categorical = prefix == "categorical__"
            name = name[len(prefix) :]
            break
    if is_categorical and "_" in name:
        column, _, value = name.rpartition("_")
        if column:
            return f"{column} = {value}"
    return name


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def engineer_features(customer: dict[str, Any]) -> dict[str, Any]:
    """Add the derived features used by the model to a customer record."""
    fe = data.model_spec()["feature_engineering"]
    merged = dict(customer)

    tenure = _num(merged.get(fe["tenure_column"]))
    total = _num(merged.get(fe["total_charges_column"]))
    monthly = _num(merged.get(fe["monthly_charges_column"]))
    contract = str(merged.get(fe["contract_column"], ""))

    # tenure_bucket
    edges, labels = fe["tenure_bucket_edges"], fe["tenure_bucket_labels"]
    bucket = labels[-1]
    for i, upper in enumerate(edges[1:]):
        if tenure <= upper:
            bucket = labels[i]
            break
    merged["tenure_bucket"] = bucket

    # service_count
    non_service = {str(v).strip().lower() for v in fe["non_service_values"]}
    merged["service_count"] = sum(
        1
        for col in fe["service_columns"]
        if str(merged.get(col, "")).strip().lower() not in non_service
        and merged.get(col) is not None
    )

    # average_monthly_spend
    merged["average_monthly_spend"] = total / tenure if tenure > 0 else monthly

    # contract_commitment_score
    scores = fe["contract_commitment_scores"]
    merged["contract_commitment_score"] = scores.get(contract)

    # risk_value_quadrant
    value = _num(merged.get(fe["cltv_column"]), total)
    high_value = value >= fe["cltv_median"]
    commitment = scores.get(contract, 0)
    high_risk = commitment < fe["max_commitment"]
    merged["risk_value_quadrant"] = (
        f"{'High Value' if high_value else 'Low Value'} / "
        f"{'High Risk' if high_risk else 'Low Risk'}"
    )
    return merged


def _score(merged: dict[str, Any]) -> tuple[float, list[tuple[str, float]]]:
    """Return (logit, contributions) for a feature-engineered record."""
    spec = data.model_spec()
    coef = spec["coef"]
    logit = spec["intercept"]
    contributions: list[tuple[str, float]] = []

    for col in spec["numeric_features"]:
        raw = merged.get(col)
        val = _num(raw, spec["numeric_impute"][col])
        scaled = (val - spec["numeric_mean"][col]) / spec["numeric_scale"][col]
        name = f"numeric__{col}"
        contribution = coef.get(name, 0.0) * scaled
        logit += contribution
        contributions.append((name, contribution))

    for col in spec["categorical_features"]:
        raw = merged.get(col)
        value = (
            str(raw)
            if raw is not None and raw != ""
            else spec["categorical_impute"][col]
        )
        for category in spec["categorical_categories"][col]:
            if value == category:
                name = f"categorical__{col}_{category}"
                contribution = coef.get(name, 0.0)
                logit += contribution
                contributions.append((name, contribution))
    return logit, contributions


def assign_persona(customer: dict[str, Any]) -> tuple[int, str]:
    """Assign a segment and persona via nearest K-Means centroid."""
    spec = data.segmentation_spec()
    scaled = []
    for col in spec["features"]:
        val = _num(customer.get(col), spec["impute"][col])
        scaled.append((val - spec["mean"][col]) / spec["scale"][col])

    best_segment, best_distance = 0, float("inf")
    for index, centroid in enumerate(spec["centroids"]):
        distance = sum((a - b) ** 2 for a, b in zip(scaled, centroid))
        if distance < best_distance:
            best_segment, best_distance = index, distance
    persona = spec["seg_to_persona"].get(str(best_segment), "")
    return best_segment, persona


def recommend(persona: str, drivers: list[str]) -> list[str]:
    """Hybrid persona-playbook + driver-triggered actions from the catalog."""
    cat = data.catalog()
    actions: list[str] = list(cat.get("personas", {}).get(persona, []))
    for rule in cat.get("driver_rules", []):
        match, action = rule.get("match", ""), rule.get("action", "")
        if any(match in driver for driver in drivers) and action not in actions:
            actions.append(action)
    return actions


def predict(customer: dict[str, Any]) -> dict[str, Any]:
    """Score a customer and return probability, drivers, persona, and actions.

    Args:
        customer: Raw customer attributes (subset of the model input columns).

    Returns:
        Prediction payload with churn probability, risk band, top drivers,
        persona, and recommended retention actions.
    """
    merged = engineer_features(customer)
    logit, contributions = _score(merged)
    logit = max(-30.0, min(30.0, logit))
    probability = 1.0 / (1.0 + math.exp(-logit))

    contributions.sort(key=lambda c: abs(c[1]), reverse=True)
    top = [c for c in contributions if abs(c[1]) > 1e-9][:5]
    driver_objects = [
        {
            "feature": humanize(name),
            "direction": "increases churn risk" if value > 0 else "reduces churn risk",
            "impact": round(abs(value), 4),
        }
        for name, value in top
    ]
    readable_drivers = [d["feature"] for d in driver_objects]

    segment, persona = assign_persona(customer)
    band = "high" if probability >= 0.66 else "medium" if probability >= 0.33 else "low"
    return {
        "churn_probability": round(probability, 4),
        "risk_band": band,
        "segment": segment,
        "persona": persona,
        "drivers": driver_objects,
        "recommended_actions": recommend(persona, readable_drivers),
    }
