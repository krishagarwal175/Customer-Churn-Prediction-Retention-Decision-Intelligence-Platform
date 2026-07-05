from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import pandas as pd

__all__ = ["BusinessEDA", "DEFAULT_EDA_SCHEMA"]

# Default semantic column mapping for the active IBM Telco Customer Churn
# Status dataset. BusinessEDA is schema-driven: these defaults keep the
# single-argument constructor working, while callers may inject an override
# (e.g. ``config["schema"]["eda"]``) to point at a different dataset layout.
DEFAULT_EDA_SCHEMA: dict[str, Any] = {
    "churn_column": "Churn Label",
    "positive_churn_value": "Yes",
    "monthly_charges_column": "Monthly Charges",
    "total_charges_column": "Total Charges",
    "tenure_column": "Tenure Months",
    "contract_column": "Contract",
    "month_to_month_value": "Month-to-month",
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
    "segment_columns": (
        "Senior Citizen",
        "Partner",
        "Dependents",
        "Paperless Billing",
        "Payment Method",
    ),
}


class BusinessEDA:
    """Compute business-focused exploratory summaries for churn datasets.

    The analyzer is schema-driven: every business column is resolved from a
    semantic mapping (logical role -> actual column name) rather than being
    hardcoded, so it works across dataset variants without code changes. It is
    computation only and never mutates the input DataFrame.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        schema: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize the BusinessEDA analyzer.

        Args:
            data: Input customer churn dataset.
            schema: Optional semantic column mapping overriding
                :data:`DEFAULT_EDA_SCHEMA`. Typically ``config["schema"]["eda"]``.

        Raises:
            ValueError: If the input is not a DataFrame or is empty.
        """
        self.logger = logging.getLogger(__name__)

        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame.")

        if data.empty:
            raise ValueError("Input dataframe must not be empty.")

        resolved: dict[str, Any] = {**DEFAULT_EDA_SCHEMA, **dict(schema or {})}

        self.data: pd.DataFrame = data.copy(deep=True)

        self._churn_column: str = resolved["churn_column"]
        self._positive_churn_value: str = str(resolved["positive_churn_value"])
        self._monthly_charges_column: str = resolved["monthly_charges_column"]
        self._total_charges_column: str = resolved["total_charges_column"]
        self._tenure_column: str = resolved["tenure_column"]
        self._contract_column: str = resolved["contract_column"]
        self._month_to_month_value: str = resolved["month_to_month_value"]
        self._service_columns: list[str] = list(resolved["service_columns"])
        self._segment_columns: list[str] = list(resolved["segment_columns"])

    def churn_summary(self) -> dict[str, float | int]:
        """Compute top-level churn summary metrics.

        Returns:
            Dictionary containing total customers, churned customers,
            active customers, and churn rate.

        Raises:
            ValueError: If the required churn column is missing.
        """
        self._validate_columns([self._churn_column])

        total_customers = int(len(self.data))
        churned_customers = int(self._count_churned(self.data))
        active_customers = int(total_customers - churned_customers)
        churn_rate = float(self._compute_churn_rate(self.data))

        return {
            "total_customers": total_customers,
            "churned_customers": churned_customers,
            "active_customers": active_customers,
            "churn_rate": churn_rate,
        }

    def revenue_summary(self) -> dict[str, float]:
        """Compute revenue and charge summary metrics.

        Returns:
            Dictionary containing total revenue, average total charges,
            average monthly charges, churned customer revenue, and retained
            customer revenue.

        Raises:
            ValueError: If required columns are missing.
        """
        self._validate_columns(
            [
                self._churn_column,
                self._monthly_charges_column,
                self._total_charges_column,
            ]
        )

        churned_mask = self._is_churned(self.data)
        retained_mask = ~churned_mask

        total_column = self._total_charges_column
        monthly_column = self._monthly_charges_column

        total_revenue = float(self.data[total_column].sum())
        average_total_charges = float(self.data[total_column].mean())
        average_monthly_charges = float(self.data[monthly_column].mean())
        churned_customer_revenue = float(
            self.data.loc[churned_mask, total_column].sum()
        )
        retained_customer_revenue = float(
            self.data.loc[retained_mask, total_column].sum()
        )

        return {
            "total_revenue": total_revenue,
            "average_total_charges": average_total_charges,
            "average_monthly_charges": average_monthly_charges,
            "churned_customer_revenue": churned_customer_revenue,
            "retained_customer_revenue": retained_customer_revenue,
        }

    def contract_analysis(self) -> pd.DataFrame:
        """Analyze churn and charges by contract type.

        Returns:
            DataFrame indexed by contract category with:
                - customer_count
                - churn_rate
                - average_monthly_charge
                - average_tenure

        Raises:
            ValueError: If required columns are missing.
        """
        contract_column = self._contract_column
        monthly_column = self._monthly_charges_column
        tenure_column = self._tenure_column

        self._validate_columns(
            [self._churn_column, contract_column, monthly_column, tenure_column]
        )

        contract_frame = self.data.loc[
            :, [contract_column, self._churn_column, monthly_column, tenure_column]
        ].copy()
        contract_frame["_is_churned"] = self._is_churned(contract_frame).astype(float)

        result = (
            contract_frame.groupby(contract_column, dropna=False)
            .agg(
                customer_count=(self._churn_column, "size"),
                churn_rate=("_is_churned", "mean"),
                average_monthly_charge=(monthly_column, "mean"),
                average_tenure=(tenure_column, "mean"),
            )
            .rename_axis(contract_column)
        )

        result["customer_count"] = result["customer_count"].astype(int)
        result["churn_rate"] = result["churn_rate"].astype(float)
        result["average_monthly_charge"] = result["average_monthly_charge"].astype(
            float
        )
        result["average_tenure"] = result["average_tenure"].astype(float)

        return result

    def tenure_analysis(self) -> dict[str, Any]:
        """Compute tenure summary statistics and churn by tenure bucket.

        Returns:
            Dictionary containing summary statistics for tenure and a dataframe
            of churn rate by predefined tenure buckets.

        Raises:
            ValueError: If required columns are missing.
        """
        self._validate_columns([self._churn_column, self._tenure_column])

        tenure_series = self.data[self._tenure_column]

        tenure_bucket_df = self._tenure_bucket_analysis()

        return {
            "mean": float(tenure_series.mean()),
            "median": float(tenure_series.median()),
            "min": int(tenure_series.min()),
            "max": int(tenure_series.max()),
            "churn_rate_by_bucket": tenure_bucket_df,
        }

    def service_analysis(self) -> dict[str, pd.DataFrame]:
        """Analyze service-related columns for customer count and churn rate.

        Automatically inspects the configured service columns if present.

        Returns:
            Dictionary mapping column name to grouped dataframe.
        """
        available_columns = self._existing_columns(self._service_columns)
        analyses: dict[str, pd.DataFrame] = {}

        for column in available_columns:
            analyses[column] = self._grouped_customer_churn_analysis(column)

        return analyses

    def customer_segmentation(self) -> dict[str, pd.DataFrame]:
        """Analyze churn across the configured customer segmentation columns.

        Returns:
            Dictionary mapping segmentation column name to grouped dataframe.
        """
        available_columns = self._existing_columns(self._segment_columns)
        analyses: dict[str, pd.DataFrame] = {}

        for column in available_columns:
            analyses[column] = self._grouped_customer_churn_analysis(column)

        return analyses

    def executive_dashboard(self) -> dict[str, float | int]:
        """Return high-level executive KPI summary.

        Returns:
            Dictionary with customers, churn rate, average monthly charge,
            average total charge, average tenure, and month-to-month churn
            when contract data is available.

        Raises:
            ValueError: If required KPI columns are missing.
        """
        self._validate_columns(
            [
                self._churn_column,
                self._monthly_charges_column,
                self._total_charges_column,
                self._tenure_column,
            ]
        )

        dashboard: dict[str, float | int] = {
            "customers": int(len(self.data)),
            "churn_rate": float(self._compute_churn_rate(self.data)),
            "average_monthly_charge": float(
                self.data[self._monthly_charges_column].mean()
            ),
            "average_total_charge": float(self.data[self._total_charges_column].mean()),
            "average_tenure": float(self.data[self._tenure_column].mean()),
        }

        if self._contract_column in self.data.columns:
            month_to_month_mask = (
                self.data[self._contract_column] == self._month_to_month_value
            )
            month_to_month_df = self.data.loc[month_to_month_mask]
            dashboard["month_to_month_churn"] = float(
                self._compute_churn_rate(month_to_month_df)
            )

        return dashboard

    def generate_report(self) -> dict[str, Any]:
        """Generate a consolidated business EDA report.

        Returns:
            Dictionary containing all business analysis outputs.
        """
        return {
            "churn": self.churn_summary(),
            "revenue": self.revenue_summary(),
            "contracts": self.contract_analysis(),
            "tenure": self.tenure_analysis(),
            "services": self.service_analysis(),
            "segments": self.customer_segmentation(),
            "executive": self.executive_dashboard(),
        }

    def _validate_columns(self, required_columns: list[str]) -> None:
        """Validate that required columns exist in the dataset.

        Args:
            required_columns: Column names required for a calculation.

        Raises:
            ValueError: If one or more required columns are missing.
        """
        missing_columns = [
            column for column in required_columns if column not in self.data.columns
        ]
        if missing_columns:
            raise ValueError(
                f"Missing required columns: {', '.join(sorted(missing_columns))}"
            )

    def _existing_columns(self, columns: list[str]) -> list[str]:
        """Return only columns present in the dataset.

        Args:
            columns: Candidate column names.

        Returns:
            List of columns that exist in the dataset.
        """
        return [column for column in columns if column in self.data.columns]

    def _is_churned(self, df: pd.DataFrame) -> pd.Series:
        """Return boolean mask for churned customers.

        Args:
            df: Dataframe to evaluate.

        Returns:
            Boolean Series where True indicates churned customers.
        """
        positive_value = self._positive_churn_value.strip().lower()
        return (
            df[self._churn_column].astype(str).str.strip().str.lower() == positive_value
        )

    def _count_churned(self, df: pd.DataFrame) -> int:
        """Count churned customers in a dataframe.

        Args:
            df: Dataframe to evaluate.

        Returns:
            Number of churned customers.
        """
        return int(self._is_churned(df).sum())

    def _compute_churn_rate(self, df: pd.DataFrame) -> float:
        """Compute churn rate for a dataframe.

        Args:
            df: Dataframe containing a churn column.

        Returns:
            Churn rate as a float between 0 and 1. Returns 0.0 for empty input.
        """
        if df.empty:
            return 0.0

        churned_count = self._count_churned(df)
        return float(churned_count / len(df))

    def _grouped_customer_churn_analysis(self, column: str) -> pd.DataFrame:
        """Compute customer count and churn rate grouped by a column.

        Args:
            column: Column used for grouping.

        Returns:
            Grouped dataframe with customer count and churn rate.
        """
        self._validate_columns([self._churn_column, column])

        analysis_frame = self.data.loc[:, [column, self._churn_column]].copy()
        analysis_frame["_is_churned"] = self._is_churned(analysis_frame).astype(float)

        result = (
            analysis_frame.groupby(column, dropna=False)
            .agg(
                customer_count=(self._churn_column, "size"),
                churn_rate=("_is_churned", "mean"),
            )
            .rename_axis(column)
        )

        result["customer_count"] = result["customer_count"].astype(int)
        result["churn_rate"] = result["churn_rate"].astype(float)

        return result

    def _tenure_bucket_analysis(self) -> pd.DataFrame:
        """Compute customer count and churn rate by tenure bucket.

        Returns:
            DataFrame indexed by tenure bucket.
        """
        bucket_labels = ["0-12", "13-24", "25-48", "49-60", "61-72"]
        bucket_edges = [-1, 12, 24, 48, 60, 72]

        temp_df = self.data.loc[:, [self._tenure_column, self._churn_column]].copy()
        temp_df["tenure_bucket"] = pd.cut(
            temp_df[self._tenure_column],
            bins=bucket_edges,
            labels=bucket_labels,
            include_lowest=True,
        )
        temp_df["_is_churned"] = self._is_churned(temp_df).astype(float)

        result = (
            temp_df.groupby("tenure_bucket", dropna=False, observed=False)
            .agg(
                customer_count=(self._churn_column, "size"),
                churn_rate=("_is_churned", "mean"),
            )
            .rename_axis("tenure_bucket")
        )

        result["customer_count"] = result["customer_count"].astype(int)
        result["churn_rate"] = result["churn_rate"].astype(float)

        return result
