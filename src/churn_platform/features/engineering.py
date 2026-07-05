"""Deterministic, schema-driven business feature engineering.

FeatureEngineer derives interpretable retention features from a cleaned churn
dataset. It is computation only: it never mutates the input DataFrame, performs
no IO/plotting/model fitting, and resolves every column from a semantic mapping
rather than hardcoding dataset schema. Features whose source columns are absent
are skipped with a warning so the transform degrades gracefully.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["FeatureEngineer", "DEFAULT_FEATURE_SCHEMA"]

# Domain constants for tenure bucketing (months). Mirrors BusinessEDA buckets.
_TENURE_BUCKET_EDGES: tuple[float, ...] = (-1, 12, 24, 48, 60, 72)
_TENURE_BUCKET_LABELS: tuple[str, ...] = ("0-12", "13-24", "25-48", "49-60", "61-72")

# Default semantic mapping for the active IBM Telco Customer Churn Status
# dataset. Overridable via the ``schema`` argument (e.g. config["schema"]
# ["features"]) so the feature layer never hardcodes dataset schema.
DEFAULT_FEATURE_SCHEMA: dict[str, Any] = {
    "tenure_column": "Tenure Months",
    "contract_column": "Contract",
    "monthly_charges_column": "Monthly Charges",
    "total_charges_column": "Total Charges",
    "cltv_column": "CLTV",
    "service_columns": (
        "Phone Service",
        "Multiple Lines",
        "Internet Service",
        "Online Security",
        "Online Backup",
        "Device Protection",
        "Tech Support",
        "Streaming TV",
        "Streaming Movies",
    ),
    "non_service_values": ("No", "No phone service", "No internet service"),
    "contract_commitment_scores": {
        "Month-to-month": 0,
        "One year": 1,
        "Two year": 2,
    },
}


class FeatureEngineer:
    """Derive interpretable churn/retention features from a cleaned dataset."""

    def __init__(self, schema: Mapping[str, Any] | None = None) -> None:
        """Initialize the feature engineer.

        Args:
            schema: Optional semantic column mapping overriding
                :data:`DEFAULT_FEATURE_SCHEMA`. Typically
                ``config["schema"]["features"]``.
        """
        self.logger = logging.getLogger(__name__)

        resolved: dict[str, Any] = {**DEFAULT_FEATURE_SCHEMA, **dict(schema or {})}

        self._tenure_column: str = resolved["tenure_column"]
        self._contract_column: str = resolved["contract_column"]
        self._monthly_charges_column: str = resolved["monthly_charges_column"]
        self._total_charges_column: str = resolved["total_charges_column"]
        self._cltv_column: str = resolved["cltv_column"]
        self._service_columns: list[str] = list(resolved["service_columns"])
        self._non_service_values: set[str] = {
            str(value).strip().lower() for value in resolved["non_service_values"]
        }
        self._contract_commitment_scores: dict[str, int] = dict(
            resolved["contract_commitment_scores"]
        )

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a new DataFrame with engineered features appended.

        The input DataFrame is never mutated. Each feature is generated only
        when its required source columns are present; otherwise it is skipped
        and a warning is logged.

        Args:
            df: Cleaned customer churn dataset.

        Returns:
            A new DataFrame containing the original columns plus any engineered
            features that could be produced.

        Raises:
            TypeError: If ``df`` is not a pandas DataFrame.
            ValueError: If ``df`` is empty.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"`df` must be a pandas DataFrame, got {type(df).__name__}."
            )
        if df.empty:
            raise ValueError("Input DataFrame must contain at least one row.")

        result = df.copy(deep=True)

        result = self._add_tenure_bucket(result)
        result = self._add_service_count(result)
        result = self._add_average_monthly_spend(result)
        result = self._add_contract_commitment_score(result)
        result = self._add_risk_value_quadrant(result)

        self.logger.info(
            "Feature engineering complete: %d columns (%d added).",
            result.shape[1],
            result.shape[1] - df.shape[1],
        )
        return result

    def _add_tenure_bucket(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ``tenure_bucket`` categorical from the tenure column."""
        if not self._require(df, [self._tenure_column], "tenure_bucket"):
            return df

        df["tenure_bucket"] = pd.cut(
            df[self._tenure_column],
            bins=list(_TENURE_BUCKET_EDGES),
            labels=list(_TENURE_BUCKET_LABELS),
            include_lowest=True,
        )
        return df

    def _add_service_count(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ``service_count``: number of subscribed services per customer."""
        available = [c for c in self._service_columns if c in df.columns]
        if not available:
            self.logger.warning(
                "Skipping 'service_count': no configured service columns present."
            )
            return df

        def _is_subscribed(series: pd.Series) -> pd.Series:
            normalized = series.astype(str).str.strip().str.lower()
            return (~normalized.isin(self._non_service_values)).astype(int)

        df["service_count"] = (
            pd.concat([_is_subscribed(df[c]) for c in available], axis=1)
            .sum(axis=1)
            .astype(int)
        )
        return df

    def _add_average_monthly_spend(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ``average_monthly_spend``: total charges amortized over tenure."""
        if not self._require(
            df,
            [self._total_charges_column, self._tenure_column],
            "average_monthly_spend",
        ):
            return df

        tenure = df[self._tenure_column].astype(float)
        total = df[self._total_charges_column].astype(float)
        spend = np.where(tenure > 0, total / tenure.replace(0, np.nan), np.nan)

        # For zero-tenure customers fall back to monthly charges when available.
        if self._monthly_charges_column in df.columns:
            fallback = df[self._monthly_charges_column].astype(float)
            spend = np.where(tenure > 0, spend, fallback)

        df["average_monthly_spend"] = pd.Series(spend, index=df.index).astype(float)
        return df

    def _add_contract_commitment_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ordinal ``contract_commitment_score`` from the contract column."""
        if not self._require(df, [self._contract_column], "contract_commitment_score"):
            return df

        df["contract_commitment_score"] = (
            df[self._contract_column]
            .map(self._contract_commitment_scores)
            .astype("Int64")
        )
        return df

    def _add_risk_value_quadrant(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ``risk_value_quadrant`` combining customer value and churn risk.

        Value is derived from CLTV (falling back to total charges); risk is
        proxied by contract commitment (lower commitment => higher risk). Each
        axis is split at its median to form four interpretable quadrants.
        """
        value_column = self._select_first_present(
            df, [self._cltv_column, self._total_charges_column]
        )
        if value_column is None or self._contract_column not in df.columns:
            self.logger.warning(
                "Skipping 'risk_value_quadrant': value or contract column missing."
            )
            return df

        value = pd.to_numeric(df[value_column], errors="coerce")
        high_value = value >= value.median()

        commitment = df[self._contract_column].map(self._contract_commitment_scores)
        # Highest commitment tier is treated as low risk; everything else high.
        max_commitment = max(self._contract_commitment_scores.values(), default=0)
        high_risk = commitment < max_commitment

        value_label = np.where(high_value, "High Value", "Low Value")
        risk_label = np.where(high_risk, "High Risk", "Low Risk")
        df["risk_value_quadrant"] = pd.Series(
            [f"{v} / {r}" for v, r in zip(value_label, risk_label)],
            index=df.index,
            dtype="object",
        )
        return df

    def _require(self, df: pd.DataFrame, columns: list[str], feature_name: str) -> bool:
        """Return True if all columns exist in ``df``, else warn and skip.

        Args:
            df: DataFrame being transformed.
            columns: Source columns required for the feature.
            feature_name: Name of the feature being generated.

        Returns:
            True when every required column is present.
        """
        missing = [column for column in columns if column not in df.columns]
        if missing:
            self.logger.warning(
                "Skipping '%s': missing required columns: %s",
                feature_name,
                ", ".join(missing),
            )
            return False
        return True

    @staticmethod
    def _select_first_present(df: pd.DataFrame, candidates: list[str]) -> str | None:
        """Return the first candidate column that is present in ``df``."""
        for candidate in candidates:
            if candidate and candidate in df.columns:
                return candidate
        return None
