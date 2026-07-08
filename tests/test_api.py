"""Tests for the FastAPI churn service."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.index import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def test_health(client: TestClient) -> None:
    assert client.get("/api/health").json() == {"status": "ok"}


def test_home_page_renders(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "Churn Decision Intelligence" in response.text
    assert 'id="pform"' in response.text


def test_kpis_and_metrics(client: TestClient) -> None:
    kpis = client.get("/api/kpis").json()
    assert kpis["churn"]["total_customers"] == 7043
    metrics = client.get("/api/metrics").json()
    assert 0.5 <= metrics["roc_auc"] <= 1.0


def test_drivers_have_no_geographic_leakage(client: TestClient) -> None:
    features = " ".join(d["feature"] for d in client.get("/api/drivers").json())
    for leaked in ("Zip", "Latitude", "Longitude", "City"):
        assert leaked not in features


def test_customers_filter_and_pagination(client: TestClient) -> None:
    payload = client.get("/api/customers?min_probability=0.8&limit=5").json()
    assert payload["total"] >= len(payload["items"])
    assert len(payload["items"]) <= 5
    assert all(c["churn_probability"] >= 0.8 for c in payload["items"])


def test_customer_lookup_and_404(client: TestClient) -> None:
    some_id = client.get("/api/customers?limit=1").json()["items"][0]["CustomerID"]
    assert client.get(f"/api/customers/{some_id}").status_code == 200
    assert client.get("/api/customers/NON-EXISTENT").status_code == 404


def test_recommendations(client: TestClient) -> None:
    recs = client.get("/api/recommendations?top_n=5").json()
    assert len(recs) == 5
    assert recs[0]["recommended_actions"]


def test_simulate_math(client: TestClient) -> None:
    summary = client.get("/api/simulate?uplift=0.3&cost=60").json()
    assert summary["customers_total"] == 7043
    assert summary["roi"] >= 0


def test_predict_high_risk_customer(client: TestClient) -> None:
    response = client.post(
        "/api/predict",
        json={
            "Tenure Months": 1,
            "Monthly Charges": 100,
            "Total Charges": 100,
            "CLTV": 3000,
            "Contract": "Month-to-month",
            "Internet Service": "Fiber optic",
            "Online Security": "No",
            "Tech Support": "No",
            "Dependents": "No",
        },
    )
    body = response.json()
    assert response.status_code == 200
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["risk_band"] in {"low", "medium", "high"}
    assert body["persona"]
    assert body["drivers"]
    assert body["recommended_actions"]


def test_predict_low_risk_customer_scores_lower(client: TestClient) -> None:
    high = client.post(
        "/api/predict",
        json={"Tenure Months": 1, "Contract": "Month-to-month", "Monthly Charges": 100},
    ).json()["churn_probability"]
    low = client.post(
        "/api/predict",
        json={"Tenure Months": 70, "Contract": "Two year", "Monthly Charges": 25},
    ).json()["churn_probability"]
    assert low < high


def test_predict_accepts_partial_input(client: TestClient) -> None:
    response = client.post("/api/predict", json={"Contract": "Two year"})
    assert response.status_code == 200
