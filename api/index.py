"""FastAPI application entry point (Vercel serverless function).

Serves precomputed churn analytics artifacts and a lightweight, dependency-free
live prediction endpoint. Heavy training/explainability runs offline; only slim
artifacts and a NumPy model are loaded here.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Query

from api import data, predictor
from api.schemas import CampaignSummary, CustomerInput, PredictionResponse

app = FastAPI(
    title="Customer Churn Decision Intelligence API",
    description=(
        "Retention decision-intelligence service: precomputed churn analytics "
        "plus live scoring for new customers."
    ),
    version="0.1.0",
)


@app.get("/api/health", tags=["system"])
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


@app.get("/api/kpis", tags=["analytics"])
def get_kpis() -> dict[str, Any]:
    """Headline churn and revenue KPIs."""
    return data.kpis()


@app.get("/api/metrics", tags=["analytics"])
def get_metrics() -> dict[str, float]:
    """Model evaluation metrics."""
    return data.metrics()


@app.get("/api/drivers", tags=["analytics"])
def get_drivers() -> list[dict[str, Any]]:
    """Global churn drivers (mean absolute SHAP)."""
    return data.drivers()


@app.get("/api/segments", tags=["analytics"])
def get_segments() -> list[dict[str, Any]]:
    """Customer segment profiles and personas."""
    return data.segments()


@app.get("/api/customers", tags=["customers"])
def list_customers(
    persona: str | None = None,
    min_probability: float = Query(0.0, ge=0.0, le=1.0),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> dict[str, Any]:
    """Filter and paginate scored customers."""
    return data.filter_customers(persona, min_probability, limit, offset)


@app.get("/api/customers/{customer_id}", tags=["customers"])
def get_customer(customer_id: str) -> dict[str, Any]:
    """Return a single scored customer by id."""
    customer = data.get_customer(customer_id)
    if customer is None:
        raise HTTPException(status_code=404, detail="Customer not found")
    return customer


@app.get("/api/recommendations", tags=["retention"])
def get_recommendations(top_n: int = Query(20, ge=1, le=50)) -> list[dict[str, Any]]:
    """Prioritized retention shortlist with recommended actions."""
    return data.recommendations(top_n)


@app.get("/api/simulate", response_model=CampaignSummary, tags=["retention"])
def simulate(
    uplift: float = Query(0.30, ge=0.0, le=1.0),
    cost: float = Query(60.0, ge=0.0),
) -> CampaignSummary:
    """Simulate retention campaign economics over the customer base."""
    return CampaignSummary(**data.simulate_campaign(uplift, cost))


@app.post("/api/predict", response_model=PredictionResponse, tags=["retention"])
def predict(customer: CustomerInput) -> PredictionResponse:
    """Live churn scoring and recommendations for a new customer."""
    return PredictionResponse(**predictor.predict(customer.to_record()))
