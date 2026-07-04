"""Deterministic cleaning for churn preprocessing."""

from __future__ import annotations

import logging
from typing import Any
from typing import Final

import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype

LOGGER_NAME: Final = __name__

__all__ = ["DataCleaner"]


class DataCleaner:
    """Apply deterministic, non-mutating cleaning to a churn dataset.
    This cleaner performs only deterministic preprocessing steps:
    - strip whitespace from column names
    - strip whitespace from string values
    - replace empty strings with ``pd.NA``
    - safely convert the configured "Total Charges" column to numeric
    - remove duplicate rows
    - remove identifier columns
    - remove leakage columns
    - preserve the target column
    - validate required columns remain
    - validate duplicate customer IDs when an identifier column exists

    The input DataFrame is never mutated.
    """

    def __init__(
        self,
        config: Any,
        schema: Any,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the cleaner.

        Args:
            config: Configuration object or mapping.
            schema: Schema object or mapping describing target, required,
                identifier, and leakage columns.
            logger: Optional logger instance.

        Raises:
            TypeError: If config or schema is None.
            ValueError: If required configuration values are invalid.
        """
        if config is None:
            raise TypeError("`config` must not be None.")
        if schema is None:
            raise TypeError("`schema` must not be None.")

        self._config = config
        self._schema = schema
        self._logger = logger or logging.getLogger(__name__)
        self._report: dict[str, Any] = {}

        total_charges_column = self._get_config(
            "total_charges_column",
            default="Total Charges",
        )
        if (
            not isinstance(total_charges_column, str)
            or not total_charges_column.strip()
        ):
            raise ValueError(
                "Configuration field 'total_charges_column' must be a non-empty string."
            )

        target_column = self._get_schema("target_column")
        if target_column is not None and not isinstance(target_column, str):
            raise ValueError("Schema field 'target_column' must be a string.")

        for field_name in ("required_columns", "identifier_columns", "leakage_columns"):
            value = self._get_schema(field_name, default=[])
            if isinstance(value, str):
                raise ValueError(
                    f"Schema field '{field_name}' must be a list-like collection, not a string."
                )
            if value is not None:
                try:
                    list(value)
                except TypeError as exc:
                    raise ValueError(
                        f"Schema field '{field_name}' must be a list-like collection."
                    ) from exc

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the input DataFrame and return a new DataFrame.

        Args:
            df: Input dataset.

        Returns:
            A new cleaned DataFrame.

        Raises:
            TypeError: If df is not a pandas DataFrame.
            ValueError: If the DataFrame is invalid or required columns are
               missing after cleaning.
        """
        validated_df = self._validate_dataframe(df)
        cleaned_df = validated_df.copy(deep=True)

        self._report = {
            "rows_before": int(len(cleaned_df)),
            "rows_after": None,
            "duplicates_removed": 0,
            "columns_removed": [],
            "missing_values_created": 0,
            "warnings": [],
        }

        self._logger.info(
            "Starting data cleaning: rows=%d, columns=%d",
            len(cleaned_df),
            len(cleaned_df.columns),
        )

        cleaned_df = self._normalize_column_names(cleaned_df)
        cleaned_df = self._clean_string_values(cleaned_df)
        cleaned_df = self._convert_total_charges(cleaned_df)

        rows_before_dedup = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        self._report["duplicates_removed"] = int(rows_before_dedup - len(cleaned_df))
        self._logger.info(
            "Removed duplicate rows: %d",
            self._report["duplicates_removed"],
        )

        cleaned_df = self._remove_columns(cleaned_df, schema_key="identifier_columns")
        cleaned_df = self._remove_columns(cleaned_df, schema_key="leakage_columns")
        cleaned_df = self._validate_output(cleaned_df)

        self._report["rows_after"] = int(len(cleaned_df))

        self._logger.info(
            "Completed data cleaning: rows_before=%d, rows_after=%d, "
            "duplicates_removed=%d, columns_removed=%d, missing_values_created=%d",
            self._report["rows_before"],
            self._report["rows_after"],
            self._report["duplicates_removed"],
            len(self._report["columns_removed"]),
            self._report["missing_values_created"],
        )

        return cleaned_df

    def get_report(self) -> dict[str, Any]:
        """Return the most recent cleaning report.

        Returns:
            A shallow copy of the internal cleaning report dictionary.
        """
        return self._report.copy()

    def _validate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate the input DataFrame.

        Args:
            df: Input object expected to be a pandas DataFrame.

        Returns:
            The validated DataFrame.

        Raises:
            TypeError: If the input is not a pandas DataFrame.
            ValueError: If the DataFrame is empty or has duplicate columns.
        """
        if not isinstance(df, pd.DataFrame):
            self._logger.error(
                "Input validation failed: expected pandas DataFrame, got %s.",
                type(df).__name__,
            )
            raise TypeError(
                f"`df` must be a pandas DataFrame, got {type(df).__name__}."
            )

        if len(df) == 0:
            self._logger.error("Input validation failed: DataFrame has zero rows.")
            raise ValueError("Input DataFrame must contain at least one row.")

        if df.columns.has_duplicates:
            duplicate_columns = df.columns[df.columns.duplicated()].tolist()
            self._logger.error(
                "Input validation failed: duplicate column names found: %s",
                duplicate_columns,
            )
            raise ValueError(
                "Input DataFrame contains duplicate column names: "
                f"{duplicate_columns}"
            )

        return df

    def _normalize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Strip whitespace from column names.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with normalized column names.

        Raises:
            ValueError: If normalization creates duplicate column names.
        """
        original_columns = list(df.columns)
        normalized_columns = [
            column.strip() if isinstance(column, str) else column
            for column in original_columns
        ]

        if len(set(normalized_columns)) != len(normalized_columns):
            duplicates: list[Any] = []
            seen: set[Any] = set()
            for column in normalized_columns:
                if column in seen and column not in duplicates:
                    duplicates.append(column)
                seen.add(column)

            self._logger.error(
                "Column normalization failed: duplicate columns created: %s",
                duplicates,
            )
            raise ValueError(
                "Column name normalization created duplicate columns: " f"{duplicates}"
            )

        renamed_df = df.rename(columns=dict(zip(original_columns, normalized_columns)))

        renamed_count = sum(
            before != after
            for before, after in zip(original_columns, normalized_columns)
        )
        self._logger.info("Normalized column names: %d renamed.", renamed_count)

        return renamed_df

    def _clean_string_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Strip whitespace from string values and replace empty strings with NA.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with cleaned string values.
        """
        cleaned_df = df.copy(deep=True)
        missing_values_created = 0

        for column in cleaned_df.columns:
            series = cleaned_df[column]
            if not (is_object_dtype(series) or is_string_dtype(series)):
                continue

            stripped = series.map(
                lambda value: value.strip() if isinstance(value, str) else value
            )
            empty_mask = stripped.eq("").fillna(False)
            missing_values_created += int(empty_mask.sum())
            cleaned_df[column] = stripped.mask(empty_mask, pd.NA)

        self._report["missing_values_created"] += missing_values_created
        self._logger.info(
            "Cleaned string values; empty strings converted to NA=%d.",
            missing_values_created,
        )

        return cleaned_df

    def _convert_total_charges(self, df: pd.DataFrame) -> pd.DataFrame:
        """Safely convert the Total Charges column to numeric when present.

        Args:
           df: Input DataFrame.

        Returns:
            DataFrame with the Total Charges column converted when present.
        """
        cleaned_df = df.copy(deep=True)
        column_name = self._get_config(
            "total_charges_column",
            default="Total Charges",
        )
        column_name = column_name.strip()

        if column_name not in cleaned_df.columns:
            warning = (
                f"Configured Total Charges column '{column_name}' not found; "
                "numeric conversion skipped."
            )
            self._report["warnings"].append(warning)
            self._logger.warning(warning)
            return cleaned_df

        series_before = cleaned_df[column_name]
        missing_before = int(series_before.isna().sum())
        converted = pd.to_numeric(series_before, errors="coerce")
        missing_after = int(converted.isna().sum())
        created_missing = max(0, missing_after - missing_before)

        cleaned_df[column_name] = converted
        self._report["missing_values_created"] += created_missing

        self._logger.info(
            "Converted '%s' to numeric; additional missing values created=%d.",
            column_name,
            created_missing,
        )

        return cleaned_df

    def _validate_output(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate that required columns remain after cleaning.

        Also validates duplicate customer IDs if an identifier column exists.

        Args:
            df: Cleaned DataFrame.

        Returns:
            The validated DataFrame.

        Raises:
            ValueError: If required columns are missing, target is missing,
                or duplicate customer IDs are found.
        """
        required_columns = list(self._get_schema("required_columns", default=[]))
        target_column = self._get_schema("target_column")
        identifier_columns = list(self._get_schema("identifier_columns", default=[]))

        missing_required = [
            column for column in required_columns if column not in df.columns
        ]
        if missing_required:
            self._logger.error(
                "Output validation failed: required columns missing: %s",
                missing_required,
            )
            raise ValueError(
                "Required columns are missing after cleaning: " f"{missing_required}"
            )

        if target_column is not None and target_column not in df.columns:
            self._logger.error(
                "Output validation failed: target column missing: %s",
                target_column,
            )
            raise ValueError(
                f"Target column '{target_column}' is missing after cleaning."
            )

        identifier_column = next(
            (column for column in identifier_columns if column in df.columns),
            None,
        )
        if identifier_column is not None and df[identifier_column].duplicated().any():
            duplicate_count = int(df[identifier_column].duplicated().sum())
            self._logger.error(
                "Output validation failed: duplicate customer IDs found in '%s': %d",
                identifier_column,
                duplicate_count,
            )
            raise ValueError(
                f"Duplicate customer IDs found in identifier column "
                f"'{identifier_column}': {duplicate_count}"
            )

        self._logger.info(
            "Output validation succeeded: rows=%d, columns=%d.",
            len(df),
            len(df.columns),
        )
        return df

    def _remove_columns(self, df: pd.DataFrame, schema_key: str) -> pd.DataFrame:
        """Remove schema-defined columns while preserving the target column.

        Args:
            df: Input DataFrame.
            schema_key: Schema key containing columns to remove.

        Returns:
            DataFrame with the requested columns removed.
        """
        columns = list(self._get_schema(schema_key, default=[]))
        target_column = self._get_schema("target_column")

        removable = [
            column
            for column in columns
            if column in df.columns and column != target_column
        ]
        if not removable:
            self._logger.info("No %s removed.", schema_key.replace("_", " "))
            return df

        cleaned_df = df.drop(columns=removable)
        self._report["columns_removed"].extend(removable)
        self._logger.info(
            "Removed %s: %d",
            schema_key.replace("_", " "),
            len(removable),
        )
        return cleaned_df

    def _get_config(self, key: str, default: Any = None) -> Any:
        """Return a configuration value from a mapping or object."""
        if isinstance(self._config, dict):
            return self._config.get(key, default)
        return getattr(self._config, key, default)

    def _get_schema(self, key: str, default: Any = None) -> Any:
        """Return a schema value from a mapping or object."""
        if isinstance(self._schema, dict):
            return self._schema.get(key, default)
        return getattr(self._schema, key, default)
