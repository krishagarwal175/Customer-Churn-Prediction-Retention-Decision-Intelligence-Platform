"""Pydantic request/response models for the churn API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class CustomerInput(BaseModel):
    """Attributes for live churn scoring of a new customer.

    Unknown fields are allowed and ignored; missing fields fall back to the
    model's imputation values, so partial inputs still score.
    """

    model_config = {"extra": "allow"}

    tenure_months: int | None = Field(default=None, alias="Tenure Months", ge=0)
    monthly_charges: float | None = Field(default=None, alias="Monthly Charges", ge=0)
    total_charges: float | None = Field(default=None, alias="Total Charges", ge=0)
    cltv: float | None = Field(default=None, alias="CLTV", ge=0)
    contract: str | None = Field(default=None, alias="Contract")
    internet_service: str | None = Field(default=None, alias="Internet Service")

    def to_record(self) -> dict[str, Any]:
        """Return the raw attribute dict keyed by dataset column names."""
        return self.model_dump(by_alias=True, exclude_none=True)


class Driver(BaseModel):
    feature: str
    direction: str
    impact: float


class PredictionResponse(BaseModel):
    churn_probability: float
    risk_band: str
    segment: int
    persona: str
    drivers: list[Driver]
    recommended_actions: list[str]


class CampaignSummary(BaseModel):
    uplift: float
    cost: float
    customers_total: int
    customers_targeted: int
    total_expected_saved_revenue: float
    total_intervention_cost: float
    total_expected_net_benefit: float
    roi: float
