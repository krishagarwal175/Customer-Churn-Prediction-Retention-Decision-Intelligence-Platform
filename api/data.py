"""Artifact loading and lightweight runtime analytics.

Loads the precomputed JSON artifacts once and provides read accessors plus pure
Python simulation math — no scikit-learn, xgboost, or shap at runtime.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

_ARTIFACTS = Path(__file__).parent / "artifacts"


@lru_cache(maxsize=None)
def _load(name: str) -> Any:
    """Load and cache a single artifact file."""
    path = _ARTIFACTS / name
    return json.loads(path.read_text(encoding="utf-8"))


def kpis() -> dict[str, Any]:
    return _load("kpis.json")


def metrics() -> dict[str, float]:
    return _load("metrics.json")


def drivers() -> list[dict[str, Any]]:
    return _load("drivers.json")


def segments() -> list[dict[str, Any]]:
    return _load("segments.json")


def catalog() -> dict[str, Any]:
    return _load("catalog.json")


def model_spec() -> dict[str, Any]:
    return _load("model.json")


def segmentation_spec() -> dict[str, Any]:
    return _load("segmentation.json")


def recommendations(top_n: int = 20) -> list[dict[str, Any]]:
    """Return the precomputed retention shortlist, truncated to ``top_n``."""
    return _load("recommendations.json")[:top_n]


def customers() -> list[dict[str, Any]]:
    return _load("customers.json")


def filter_customers(
    persona: str | None = None,
    min_probability: float = 0.0,
    limit: int = 100,
    offset: int = 0,
) -> dict[str, Any]:
    """Filter and paginate scored customers.

    Args:
        persona: Optional persona to filter by.
        min_probability: Minimum churn probability.
        limit: Page size.
        offset: Rows to skip.

    Returns:
        Dict with ``total`` matched and the ``items`` page.
    """
    rows = [
        row
        for row in customers()
        if row.get("churn_probability", 0.0) >= min_probability
        and (persona is None or row.get("persona") == persona)
    ]
    rows.sort(key=lambda r: r.get("churn_probability", 0.0), reverse=True)
    return {"total": len(rows), "items": rows[offset : offset + limit]}


def get_customer(customer_id: str) -> dict[str, Any] | None:
    """Return a single scored customer by id."""
    for row in customers():
        if str(row.get("CustomerID")) == str(customer_id):
            return row
    return None


def simulate_campaign(
    uplift: float, cost: float, value_field: str = "CLTV"
) -> dict[str, Any]:
    """Compute retention campaign economics over the scored customer base.

    Expected saved revenue = churn_probability x value x uplift; a customer is
    targeted when the expected net benefit exceeds the intervention cost.

    Args:
        uplift: Fraction of churn averted (0-1).
        cost: Per-customer intervention cost.
        value_field: Customer value column.

    Returns:
        Campaign summary with targeted count, totals, and ROI.
    """
    targeted = 0
    total_saved = 0.0
    total_cost = 0.0
    for row in customers():
        probability = float(row.get("churn_probability", 0.0))
        value = float(row.get(value_field, 0.0) or 0.0)
        saved = probability * uplift * value
        if saved - cost > 0:
            targeted += 1
            total_saved += saved
            total_cost += cost
    net = total_saved - total_cost
    roi = net / total_cost if total_cost > 0 else 0.0
    return {
        "uplift": uplift,
        "cost": cost,
        "customers_total": len(customers()),
        "customers_targeted": targeted,
        "total_expected_saved_revenue": round(total_saved, 2),
        "total_intervention_cost": round(total_cost, 2),
        "total_expected_net_benefit": round(net, 2),
        "roi": round(roi, 3),
    }
