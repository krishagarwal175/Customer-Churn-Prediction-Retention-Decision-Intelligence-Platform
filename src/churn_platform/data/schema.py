"""Schema definitions for the IBM Telco Customer Churn dataset.

This module is intentionally declarative. It records the expected source
dataset contract so that loading, validation, profiling, and future
preprocessing modules can share the same understanding of the data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ColumnSchema:
    """Expected constraints for a single source dataset column."""

    name: str
    dtype: str
    required: bool = True
    allowed_values: tuple[Any, ...] | None = None
    nullable: bool = False
    minimum: float | None = None
    maximum: float | None = None


@dataclass(frozen=True)
class DatasetSchema:
    """Reusable schema contract for the raw churn dataset."""

    expected_columns: tuple[ColumnSchema, ...]
    target_column: str
    identifier_column: str
    duplicate_target_columns: tuple[str, ...] = field(default_factory=tuple)
    leakage_columns: tuple[str, ...] = field(default_factory=tuple)
    optional_columns: tuple[str, ...] = field(default_factory=tuple)
    future_engineered_columns: tuple[str, ...] = field(default_factory=tuple)

    @property
    def required_columns(self) -> tuple[str, ...]:
        """Return all columns required in the raw dataset."""

        return tuple(column.name for column in self.expected_columns if column.required)

    @property
    def expected_column_names(self) -> tuple[str, ...]:
        """Return expected raw column names in source order."""

        return tuple(column.name for column in self.expected_columns)

    @property
    def dtype_map(self) -> dict[str, str]:
        """Return the expected pandas-compatible dtype group for each column."""

        return {column.name: column.dtype for column in self.expected_columns}


YES_NO = ("Yes", "No")
YES_NO_WITH_NO_PHONE = ("Yes", "No", "No phone service")
YES_NO_WITH_NO_INTERNET = ("Yes", "No", "No internet service")


IBM_TELCO_SCHEMA = DatasetSchema(
    expected_columns=(
        ColumnSchema("CustomerID", "string"),
        ColumnSchema("Count", "integer", minimum=1),
        ColumnSchema("Country", "string"),
        ColumnSchema("State", "string"),
        ColumnSchema("City", "string"),
        ColumnSchema("Zip Code", "integer"),
        ColumnSchema("Lat Long", "string"),
        ColumnSchema("Latitude", "float", minimum=-90, maximum=90),
        ColumnSchema("Longitude", "float", minimum=-180, maximum=180),
        ColumnSchema("Gender", "category", allowed_values=("Female", "Male")),
        ColumnSchema("Senior Citizen", "category", allowed_values=YES_NO),
        ColumnSchema("Partner", "category", allowed_values=YES_NO),
        ColumnSchema("Dependents", "category", allowed_values=YES_NO),
        ColumnSchema("Tenure Months", "integer", minimum=0),
        ColumnSchema("Phone Service", "category", allowed_values=YES_NO),
        ColumnSchema(
            "Multiple Lines",
            "category",
            allowed_values=YES_NO_WITH_NO_PHONE,
        ),
        ColumnSchema(
            "Internet Service",
            "category",
            allowed_values=("DSL", "Fiber optic", "No"),
        ),
        ColumnSchema(
            "Online Security",
            "category",
            allowed_values=YES_NO_WITH_NO_INTERNET,
        ),
        ColumnSchema(
            "Online Backup",
            "category",
            allowed_values=YES_NO_WITH_NO_INTERNET,
        ),
        ColumnSchema(
            "Device Protection",
            "category",
            allowed_values=YES_NO_WITH_NO_INTERNET,
        ),
        ColumnSchema(
            "Tech Support",
            "category",
            allowed_values=YES_NO_WITH_NO_INTERNET,
        ),
        ColumnSchema(
            "Streaming TV",
            "category",
            allowed_values=YES_NO_WITH_NO_INTERNET,
        ),
        ColumnSchema(
            "Streaming Movies",
            "category",
            allowed_values=YES_NO_WITH_NO_INTERNET,
        ),
        ColumnSchema(
            "Contract",
            "category",
            allowed_values=("Month-to-month", "One year", "Two year"),
        ),
        ColumnSchema("Paperless Billing", "category", allowed_values=YES_NO),
        ColumnSchema(
            "Payment Method",
            "category",
            allowed_values=(
                "Bank transfer (automatic)",
                "Credit card (automatic)",
                "Electronic check",
                "Mailed check",
            ),
        ),
        ColumnSchema("Monthly Charges", "float", minimum=0),
        ColumnSchema("Total Charges", "float", minimum=0),
        ColumnSchema("Churn Label", "category", allowed_values=YES_NO),
        ColumnSchema("Churn Value", "integer", allowed_values=(0, 1)),
        ColumnSchema("Churn Score", "integer", minimum=0, maximum=100),
        ColumnSchema("CLTV", "integer", minimum=0),
        ColumnSchema("Churn Reason", "category", nullable=True),
    ),
    target_column="Churn Label",
    identifier_column="CustomerID",
    duplicate_target_columns=("Churn Value",),
    leakage_columns=(
        "CustomerID",
        "Churn Reason",
        "Churn Category",
        "Churn Score",
        "Customer Status",
        "Churn Value",
    ),
    optional_columns=("Churn Category", "Customer Status"),
    future_engineered_columns=(
        "tenure_bucket",
        "service_count",
        "average_monthly_spend",
        "contract_commitment_score",
        "risk_value_quadrant",
    ),
)


def get_schema() -> DatasetSchema:
    """Return the active source dataset schema."""

    return IBM_TELCO_SCHEMA

