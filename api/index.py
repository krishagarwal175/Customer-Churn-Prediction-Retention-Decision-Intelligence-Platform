"""FastAPI application entry point (Vercel serverless function).

Serves precomputed churn analytics artifacts and a lightweight, dependency-free
live prediction endpoint. Heavy training/explainability runs offline; only slim
artifacts and a NumPy model are loaded here.
"""

from __future__ import annotations

from fastapi import FastAPI

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
