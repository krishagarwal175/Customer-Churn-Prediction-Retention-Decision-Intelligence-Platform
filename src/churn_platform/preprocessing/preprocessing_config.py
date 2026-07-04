"""Configuration objects for the preprocessing pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PathConfig:
    """Filesystem paths used by preprocessing."""

    input_dataset: Path = Path("data/interim/Telco_customer_churn_validated.xlsx")
    processed_data_dir: Path = Path("data/processed")
    preprocessor_artifact: Path = Path(
        "artifacts/preprocessors/preprocessing_pipeline.joblib"
    )
    preprocessing_report: Path = Path("reports/preprocessing_report.md")


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime behavior for preprocessing."""

    random_seed: int = 42
    log_level: str = "INFO"


@dataclass(frozen=True)
class CleaningConfig:
    """Cleaning, feature selection, and transformer configuration."""

    target_column: str = "Churn Label"
    positive_target_label: str = "Yes"
    negative_target_label: str = "No"
    identifier_columns: tuple[str, ...] = ("CustomerID",)
    leakage_columns: tuple[str, ...] = (
        "Churn Value",
        "Churn Score",
        "Churn Reason",
        "Churn Category",
        "Customer Status",
    )
    reporting_columns: tuple[str, ...] = ("Count", "Country", "State", "Lat Long")
    use_reporting_features: bool = False
    categorical_fill_value: str = "Unknown"
    numeric_imputation_strategy: str = "median"
    total_charges_column: str = "Total Charges"
    tenure_column: str = "Tenure Months"
    zero_tenure_total_charges_value: float = 0.0
    encoder: str = "onehot"
    handle_unknown: str = "ignore"
    scale_numeric: bool = True
    scaler: str = "standard"


@dataclass(frozen=True)
class SplitConfig:
    """Train, validation, and test split configuration."""

    train_size: float = 0.70
    validation_size: float = 0.15
    test_size: float = 0.15
    random_state: int = 42
    stratify: bool = True


@dataclass(frozen=True)
class PreprocessingConfig:
    """Top-level preprocessing configuration."""

    paths: PathConfig = field(default_factory=PathConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)
    split: SplitConfig = field(default_factory=SplitConfig)


def load_preprocessing_config(config_path: str | Path) -> PreprocessingConfig:
    """Load preprocessing configuration from a simple YAML file.

    The parser supports the two-level YAML structure used by this project,
    including inline lists such as ``[CustomerID, Churn Score]``. Keeping this
    reader local avoids making preprocessing execution depend on optional YAML
    packages before the virtual environment is installed.
    """

    raw_config = _read_simple_yaml(config_path)
    paths = raw_config.get("paths", {})
    runtime = raw_config.get("runtime", {})
    preprocessing = raw_config.get("preprocessing", {})
    split = raw_config.get("split", {})

    return PreprocessingConfig(
        paths=PathConfig(
            input_dataset=Path(paths.get("input_dataset", PathConfig.input_dataset)),
            processed_data_dir=Path(
                paths.get("processed_data_dir", PathConfig.processed_data_dir)
            ),
            preprocessor_artifact=Path(
                paths.get("preprocessor_artifact", PathConfig.preprocessor_artifact)
            ),
            preprocessing_report=Path(
                paths.get("preprocessing_report", PathConfig.preprocessing_report)
            ),
        ),
        runtime=RuntimeConfig(
            random_seed=int(runtime.get("random_seed", RuntimeConfig.random_seed)),
            log_level=str(runtime.get("log_level", RuntimeConfig.log_level)),
        ),
        cleaning=CleaningConfig(
            target_column=str(
                preprocessing.get("target_column", CleaningConfig.target_column)
            ),
            positive_target_label=str(
                preprocessing.get(
                    "positive_target_label", CleaningConfig.positive_target_label
                )
            ),
            negative_target_label=str(
                preprocessing.get(
                    "negative_target_label", CleaningConfig.negative_target_label
                )
            ),
            identifier_columns=tuple(
                preprocessing.get(
                    "identifier_columns", CleaningConfig.identifier_columns
                )
            ),
            leakage_columns=tuple(
                preprocessing.get("leakage_columns", CleaningConfig.leakage_columns)
            ),
            reporting_columns=tuple(
                preprocessing.get("reporting_columns", CleaningConfig.reporting_columns)
            ),
            use_reporting_features=bool(
                preprocessing.get(
                    "use_reporting_features", CleaningConfig.use_reporting_features
                )
            ),
            categorical_fill_value=str(
                preprocessing.get(
                    "categorical_fill_value", CleaningConfig.categorical_fill_value
                )
            ),
            numeric_imputation_strategy=str(
                preprocessing.get(
                    "numeric_imputation_strategy",
                    CleaningConfig.numeric_imputation_strategy,
                )
            ),
            total_charges_column=str(
                preprocessing.get(
                    "total_charges_column", CleaningConfig.total_charges_column
                )
            ),
            tenure_column=str(
                preprocessing.get("tenure_column", CleaningConfig.tenure_column)
            ),
            zero_tenure_total_charges_value=float(
                preprocessing.get(
                    "zero_tenure_total_charges_value",
                    CleaningConfig.zero_tenure_total_charges_value,
                )
            ),
            encoder=str(preprocessing.get("encoder", CleaningConfig.encoder)),
            handle_unknown=str(
                preprocessing.get("handle_unknown", CleaningConfig.handle_unknown)
            ),
            scale_numeric=bool(
                preprocessing.get("scale_numeric", CleaningConfig.scale_numeric)
            ),
            scaler=str(preprocessing.get("scaler", CleaningConfig.scaler)),
        ),
        split=SplitConfig(
            train_size=float(split.get("train_size", SplitConfig.train_size)),
            validation_size=float(
                split.get("validation_size", SplitConfig.validation_size)
            ),
            test_size=float(split.get("test_size", SplitConfig.test_size)),
            random_state=int(split.get("random_state", SplitConfig.random_state)),
            stratify=bool(split.get("stratify", SplitConfig.stratify)),
        ),
    )


def _read_simple_yaml(config_path: str | Path) -> dict[str, dict[str, Any]]:
    path = Path(config_path)
    config: dict[str, dict[str, Any]] = {}
    current_section: str | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", maxsplit=1)[0].rstrip()
        if not line.strip():
            continue

        if not line.startswith(" ") and line.endswith(":"):
            current_section = line[:-1].strip()
            config[current_section] = {}
            continue

        if current_section and line.startswith("  ") and ":" in line:
            key, value = line.strip().split(":", maxsplit=1)
            config[current_section][key.strip()] = _parse_value(value.strip())

    return config


def _parse_value(value: str) -> Any:
    if value.startswith("[") and value.endswith("]"):
        contents = value[1:-1].strip()
        if not contents:
            return []
        return [item.strip().strip("\"'") for item in contents.split(",")]
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("\"'")
