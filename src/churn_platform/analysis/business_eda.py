"""Business analytics metrics for cleaned customer churn data."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class BusinessEDA:
    """Compute business analytics metrics from a cleaned churn dataset.

    This class provides business-focused summary statistics for a cleaned
    customer churn dataset. It does not perform preprocessing, plotting,
    feature engineering, file I/O, or machine learning tasks.

    Attributes:
        df: Input customer churn DataFrame.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """Initialize the BusinessEDA instance.

        Args:
            df: Cleaned customer churn dataset.

        Raises:
            TypeError: If ``df`` is not a pandas DataFrame.
            ValueError: If ``df`` is empty.
        """
        if not isinstance(df, pd.DataFrame):
            logger.error("Input validation failed: df is not a pandas DataFrame.")
            raise TypeError("df must be a pandas DataFrame")

        if df.empty:
            logger.error("Input validation failed: df is empty.")
            raise ValueError("df must not be empty")

        self.df = df
        logger.info(
            "BusinessEDA initialized with %d rows and %d columns.", *self.df.shape
        )

    def churn_rate(self) -> dict[str, Any]:
        """Calculate overall churn metrics.

        Returns:
            Dictionary containing total customers, churned customers,
            active customers, and churn rate.
        """
        churn_series = self._get_churn_indicator()
        total_customers = int(len(self.df))
        churned_customers = int(churn_series.sum())
        active_customers = int(total_customers - churned_customers)
        churn_rate = (
            float(churned_customers / total_customers) if total_customers else 0.0
        )

        result = {
            "total_customers": total_customers,
            "churned_customers": churned_customers,
            "active_customers": active_customers,
            "churn_rate": churn_rate,
        }
        logger.info("Computed churn rate metrics: %s", result)
        return result

    def revenue_summary(self) -> dict[str, float]:
        """Calculate revenue-related summary statistics.

        Returns:
            Dictionary containing total monthly revenue, average monthly charge,
            total revenue, and average customer lifetime value.
        """
        monthly_charge_col = self._find_column(["MonthlyCharges", "monthly_charges"])
        total_revenue_col = self._find_column(["TotalCharges", "total_charges"])
        cltv_col = self._find_column(["CLTV", "cltv"], required=False)

        monthly_charges = pd.to_numeric(self.df[monthly_charge_col], errors="coerce")
        total_charges = pd.to_numeric(self.df[total_revenue_col], errors="coerce")

        monthly_nan_ratio = float(monthly_charges.isna().mean())
        total_nan_ratio = float(total_charges.isna().mean())

        if monthly_nan_ratio > 0.1:
            logger.warning(
                "High percentage of MonthlyCharges values could not be parsed: %.2f%%",
                monthly_nan_ratio * 100,
            )

        if total_nan_ratio > 0.1:
            logger.warning(
                "High percentage of TotalCharges values could not be parsed: %.2f%%",
                total_nan_ratio * 100,
            )

        average_cltv = 0.0
        if cltv_col is not None:
            cltv_series = pd.to_numeric(self.df[cltv_col], errors="coerce")
            cltv_nan_ratio = float(cltv_series.isna().mean())
            if cltv_nan_ratio > 0.1:
                logger.warning(
                    "High percentage of CLTV values could not be parsed: %.2f%%",
                    cltv_nan_ratio * 100,
                )
            average_cltv = float(cltv_series.mean())
        else:
            logger.warning("CLTV column not found. Returning 0.0 for average_cltv.")

        result = {
            "total_monthly_revenue": float(monthly_charges.sum()),
            "average_monthly_charge": float(monthly_charges.mean()),
            "total_revenue": float(total_charges.sum()),
            "average_cltv": average_cltv,
        }
        logger.info("Computed revenue summary: %s", result)
        return result

    def contract_summary(self) -> dict[str, dict[str, Any]]:
        """Calculate contract counts and churn rates by contract type.

        Returns:
            Dictionary keyed by contract type with count and churn rate metrics.
        """
        contract_col = self._find_column(["Contract", "contract"])
        churn_series = self._get_churn_indicator()

        working_df = pd.DataFrame(
            {
                "contract": self.df[contract_col].astype(str),
                "churn": churn_series,
            }
        )

        grouped = (
            working_df.groupby("contract", dropna=False)
            .agg(customer_count=("churn", "size"), churned_customers=("churn", "sum"))
            .assign(
                churn_rate=lambda frame: frame["churned_customers"]
                / frame["customer_count"]
            )
        )

        summary = {
            str(contract_type): {
                "customer_count": int(row["customer_count"]),
                "churned_customers": int(row["churned_customers"]),
                "churn_rate": float(row["churn_rate"]),
            }
            for contract_type, row in grouped.iterrows()
        }

        logger.info("Computed contract summary for %d contract types.", len(summary))
        return summary

    def tenure_summary(self) -> dict[str, float]:
        """Calculate tenure summary statistics.

        Returns:
            Dictionary containing average, median, maximum, and minimum tenure.
        """
        tenure_col = self._find_column(["tenure", "Tenure"])
        tenure_series = pd.to_numeric(self.df[tenure_col], errors="coerce")

        result = {
            "average_tenure": float(tenure_series.mean()),
            "median_tenure": float(tenure_series.median()),
            "maximum_tenure": float(tenure_series.max()),
            "minimum_tenure": float(tenure_series.min()),
        }
        logger.info("Computed tenure summary: %s", result)
        return result

    def service_summary(self) -> dict[str, dict[str, int]]:
        """Calculate counts for selected service-related columns.

        Returns:
            Dictionary containing value counts for Phone Service, Internet
            Service, Online Security, and Tech Support.
        """
        service_columns = {
            "Phone Service": self._find_column(
                ["PhoneService", "Phone Service", "phone_service"]
            ),
            "Internet Service": self._find_column(
                ["InternetService", "Internet Service", "internet_service"]
            ),
            "Online Security": self._find_column(
                ["OnlineSecurity", "Online Security", "online_security"]
            ),
            "Tech Support": self._find_column(
                ["TechSupport", "Tech Support", "tech_support"]
            ),
        }

        summary: dict[str, dict[str, int]] = {}
        for label, column in service_columns.items():
            counts = self.df[column].value_counts(dropna=False)
            summary[label] = {str(index): int(value) for index, value in counts.items()}

        logger.info("Computed service summary for %d service categories.", len(summary))
        return summary

    def customer_segments(self) -> dict[str, dict[str, int]]:
        """Calculate customer counts grouped by key demographic segments.

        Returns:
            Dictionary containing grouped counts for Gender, Senior Citizen,
            Partner, and Dependents.
        """
        segment_columns = {
            "Gender": self._find_column(["gender", "Gender"]),
            "Senior Citizen": self._find_column(
                ["SeniorCitizen", "Senior Citizen", "senior_citizen"]
            ),
            "Partner": self._find_column(["Partner", "partner"]),
            "Dependents": self._find_column(["Dependents", "dependents"]),
        }

        summary: dict[str, dict[str, int]] = {}
        for label, column in segment_columns.items():
            counts = self.df[column].value_counts(dropna=False)
            summary[label] = {str(index): int(value) for index, value in counts.items()}

        logger.info("Computed customer segment summary for %d segments.", len(summary))
        return summary

    def _find_column(self, candidates: list[str], required: bool = True) -> str | None:
        """Find the first matching column from a list of candidates.

        Args:
            candidates: Possible column names.
            required: Whether to raise an error if no match is found.

        Returns:
            The matched column name, or ``None`` if not required and no match exists.

        Raises:
            KeyError: If no matching column is found and ``required`` is True.
        """
        for candidate in candidates:
            if candidate in self.df.columns:
                return candidate

        if required:
            logger.error("Required column not found. Tried candidates: %s", candidates)
            raise KeyError(f"Required column not found. Expected one of: {candidates}")

        logger.warning("Optional column not found. Tried candidates: %s", candidates)
        return None

    def _get_churn_indicator(self) -> pd.Series:
        """Convert churn column values into a binary indicator series.

        Returns:
            Binary pandas Series where 1 indicates churn and 0 indicates active.

        Raises:
            KeyError: If the churn column is not present.
        """
        churn_col = self._find_column(["Churn", "churn"])
        churn_values = self.df[churn_col]

        if pd.api.types.is_bool_dtype(churn_values):
            indicator = churn_values.fillna(False).astype(int)
            logger.debug("Churn column detected as boolean.")
            return indicator

        if pd.api.types.is_numeric_dtype(churn_values):
            indicator = churn_values.fillna(0).astype(int)
            indicator = indicator.clip(lower=0, upper=1)
            logger.debug("Churn column detected as numeric.")
            return indicator

        normalized = churn_values.astype(str).str.strip().str.lower()
        indicator = normalized.isin({"yes", "true", "1", "churned"}).astype(int)
        logger.debug("Churn column detected as string/categorical.")
        return indicator
