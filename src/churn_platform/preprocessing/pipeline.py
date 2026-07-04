"""Reusable preprocessing pipeline orchestration."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline

from churn_platform.data.loader import load_dataset
from churn_platform.preprocessing.cleaning import DataCleaner, define_binary_target
from churn_platform.preprocessing.encoding import build_categorical_encoder
from churn_platform.preprocessing.preprocessing_config import (
    CleaningConfig,
    PreprocessingConfig,
    load_preprocessing_config,
)
from churn_platform.preprocessing.scaling import build_scaler
from churn_platform.preprocessing.splitting import split_train_validation_test
from churn_platform.utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)


@dataclass
class FeatureGroups:
    """Columns separated by preprocessing role."""

    features: list[str]
    target: str
    identifiers: list[str]
    leakage_columns: list[str]
    reporting_columns: list[str]
    removed_columns: list[str]


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Select model-eligible features from a cleaned dataframe."""

    def __init__(self, config: CleaningConfig | None = None) -> None:
        self.config = config or CleaningConfig()
        self.feature_columns_: list[str] = []
        self.removed_columns_: list[str] = []
        self.feature_groups_: FeatureGroups | None = None

    def fit(self, X: pd.DataFrame, y: Any = None) -> "FeatureSelector":
        """Determine reusable model feature columns from training data."""

        configured_removed = set(self.config.identifier_columns)
        configured_removed.update(self.config.leakage_columns)
        configured_removed.add(self.config.target_column)
        if not self.config.use_reporting_features:
            configured_removed.update(self.config.reporting_columns)

        self.removed_columns_ = [
            column for column in X.columns if column in configured_removed
        ]
        self.feature_columns_ = [
            column for column in X.columns if column not in configured_removed
        ]
        self.feature_groups_ = FeatureGroups(
            features=self.feature_columns_,
            target=self.config.target_column,
            identifiers=[
                column
                for column in self.config.identifier_columns
                if column in X.columns
            ],
            leakage_columns=[
                column for column in self.config.leakage_columns if column in X.columns
            ],
            reporting_columns=[
                column
                for column in self.config.reporting_columns
                if column in X.columns
            ],
            removed_columns=self.removed_columns_,
        )
        LOGGER.info(
            "FeatureSelector fitted: retained=%s removed=%s",
            len(self.feature_columns_),
            len(self.removed_columns_),
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return the feature columns learned during fit."""

        missing_features = [
            column for column in self.feature_columns_ if column not in X.columns
        ]
        if missing_features:
            raise KeyError(
                f"Input data is missing required feature columns: {missing_features}"
            )
        return X.loc[:, self.feature_columns_].copy()


class ChurnPreprocessor:
    """Reusable preprocessing object for training and future inference."""

    def __init__(self, config: PreprocessingConfig) -> None:
        self.config = config
        self.pipeline = build_preprocessing_pipeline(config.cleaning)

    def fit(self, dataframe: pd.DataFrame) -> "ChurnPreprocessor":
        """Fit preprocessing on a labeled training dataframe."""

        y = define_binary_target(dataframe, self.config.cleaning)
        self.pipeline.fit(dataframe, y)
        return self

    def transform(self, dataframe: pd.DataFrame) -> Any:
        """Transform records into a machine-learning-ready matrix."""

        return self.pipeline.transform(dataframe)

    def fit_transform(self, dataframe: pd.DataFrame) -> Any:
        """Fit preprocessing and transform the same labeled dataframe."""

        y = define_binary_target(dataframe, self.config.cleaning)
        return self.pipeline.fit_transform(dataframe, y)

    def get_feature_names(self) -> list[str]:
        """Return output feature names after encoding and scaling."""

        transformer = self.pipeline.named_steps["column_processing"]
        names = transformer.get_feature_names_out()
        return [str(name) for name in names]


def build_preprocessing_pipeline(config: CleaningConfig) -> Pipeline:
    """Build the sklearn preprocessing pipeline."""

    numeric_pipeline = Pipeline(steps=[("scaler", build_scaler(config))])
    categorical_pipeline = Pipeline(
        steps=[("encoder", build_categorical_encoder(config))]
    )

    column_processing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                make_column_selector(dtype_include="number"),
            ),
            (
                "categorical",
                categorical_pipeline,
                make_column_selector(dtype_include=["object", "string", "category"]),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return Pipeline(
        steps=[
            ("cleaning", DataCleaner(config)),
            ("feature_selection", FeatureSelector(config)),
            ("column_processing", column_processing),
        ]
    )


def run_preprocessing(
    config_path: str | Path = "config/preprocessing_config.yaml",
) -> int:
    """Run the complete preprocessing workflow and persist outputs."""

    config = load_preprocessing_config(config_path)
    configure_logging(config.runtime.log_level)

    LOGGER.info("Starting preprocessing pipeline")
    dataframe = load_dataset(config.paths.input_dataset)
    target = define_binary_target(dataframe, config.cleaning)

    X_train, X_validation, X_test, y_train, y_validation, y_test = (
        split_train_validation_test(
            dataframe,
            target,
            config.split,
        )
    )

    preprocessor = ChurnPreprocessor(config)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_validation_processed = preprocessor.transform(X_validation)
    X_test_processed = preprocessor.transform(X_test)

    config.paths.processed_data_dir.mkdir(parents=True, exist_ok=True)
    config.paths.preprocessor_artifact.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, config.paths.preprocessor_artifact)
    LOGGER.info(
        "Saved fitted preprocessing artifact to %s", config.paths.preprocessor_artifact
    )

    processed_payload = {
        "X_train": X_train_processed,
        "X_validation": X_validation_processed,
        "X_test": X_test_processed,
        "y_train": y_train.to_numpy(),
        "y_validation": y_validation.to_numpy(),
        "y_test": y_test.to_numpy(),
        "feature_names": preprocessor.get_feature_names(),
    }
    processed_output = config.paths.processed_data_dir / "preprocessed_splits.joblib"
    joblib.dump(processed_payload, processed_output)
    LOGGER.info("Saved processed train/validation/test arrays to %s", processed_output)

    report = build_preprocessing_report(
        preprocessor=preprocessor,
        config=config,
        split_sizes={
            "train": len(X_train),
            "validation": len(X_validation),
            "test": len(X_test),
        },
    )
    config.paths.preprocessing_report.parent.mkdir(parents=True, exist_ok=True)
    config.paths.preprocessing_report.write_text(report, encoding="utf-8")
    LOGGER.info("Saved preprocessing report to %s", config.paths.preprocessing_report)
    LOGGER.info("Preprocessing pipeline completed successfully")
    return 0


def build_preprocessing_report(
    preprocessor: ChurnPreprocessor,
    config: PreprocessingConfig,
    split_sizes: dict[str, int],
) -> str:
    """Build a Markdown report describing preprocessing decisions."""

    selector = preprocessor.pipeline.named_steps["feature_selection"]
    feature_groups = selector.feature_groups_
    if feature_groups is None:
        raise RuntimeError("Feature selector has not been fitted.")

    feature_names = preprocessor.get_feature_names()
    lines = [
        "# Preprocessing Report",
        "",
        "## Target Definition",
        "",
        f"- Target column: `{config.cleaning.target_column}`",
        f"- Positive class: `{config.cleaning.positive_target_label}` mapped to `1`",
        f"- Negative class: `{config.cleaning.negative_target_label}` mapped to `0`",
        "- Duplicate target columns are excluded from model features.",
        "",
        "## Features Retained",
        "",
        _format_list(feature_groups.features),
        "",
        "## Features Removed",
        "",
        "| Column | Reason |",
        "|---|---|",
    ]

    for column in feature_groups.removed_columns:
        lines.append(f"| {column} | {_removal_reason(column, config.cleaning)} |")

    lines.extend(
        [
            "",
            "## Missing Value Handling",
            "",
            "| Column Group | Strategy | Rationale |",
            "|---|---|---|",
            "| Text/categorical columns | Trim whitespace, convert empty strings to missing, fill with configured category | Preserves rows and supports unseen inference records |",
            "| `Total Charges` | Convert to numeric; fill missing zero-tenure records with configured zero value | Blank total charges occur for new customers with no accumulated charges |",
            "| Other numeric columns | Fill with training-set median by default | Robust to outliers and avoids dropping rows |",
            "",
            "## Encoding Strategy",
            "",
            f"- Encoder: `{config.cleaning.encoder}`",
            f"- Unknown category handling: `{config.cleaning.handle_unknown}`",
            "- Categorical columns are detected automatically after cleaning and feature selection.",
            "",
            "## Scaling Strategy",
            "",
            f"- Scaling enabled: `{config.cleaning.scale_numeric}`",
            f"- Scaler: `{config.cleaning.scaler}`",
            "- Numerical columns are detected automatically after cleaning and feature selection.",
            "",
            "## Split Strategy",
            "",
            f"- Train size: `{config.split.train_size}`",
            f"- Validation size: `{config.split.validation_size}`",
            f"- Test size: `{config.split.test_size}`",
            f"- Stratified: `{config.split.stratify}`",
            f"- Random state: `{config.split.random_state}`",
            "",
            "## Output Summary",
            "",
            f"- Final transformed feature count: `{len(feature_names)}`",
            f"- Train rows: `{split_sizes['train']}`",
            f"- Validation rows: `{split_sizes['validation']}`",
            f"- Test rows: `{split_sizes['test']}`",
        ]
    )

    return "\n".join(lines) + "\n"


def _format_list(values: list[str]) -> str:
    if not values:
        return "None"
    return "\n".join(f"- `{value}`" for value in values)


def _removal_reason(column: str, config: CleaningConfig) -> str:
    if column in config.identifier_columns:
        return "Identifier column; useful for lookup but not generalizable as a model feature."
    if column in config.leakage_columns:
        return "Target leakage or post-event field; excluded from training features."
    if column == config.target_column:
        return "Selected target variable; used as label, not as model input."
    if column in config.reporting_columns:
        return "Reporting-only field excluded by configuration."
    return "Excluded by preprocessing configuration."


def parse_args() -> argparse.Namespace:
    """Parse preprocessing CLI arguments."""

    parser = argparse.ArgumentParser(description="Run churn preprocessing pipeline.")
    parser.add_argument(
        "--config",
        default="config/preprocessing_config.yaml",
        help="Path to preprocessing configuration.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    raise SystemExit(run_preprocessing(args.config))


if __name__ == "__main__":
    main()
