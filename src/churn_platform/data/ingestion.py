"""Command-line orchestration for Milestone 2 data ingestion."""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path
from typing import Any

from churn_platform.data.data_dictionary import save_data_dictionary
from churn_platform.data.loader import load_dataset
from churn_platform.data.profiler import build_profile_report, save_profile_report
from churn_platform.data.validator import save_validation_result, validate_dataset
from churn_platform.utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load the project YAML configuration for ingestion.

    The project configuration intentionally uses a simple two-level YAML
    structure. This lightweight reader keeps ingestion runnable before optional
    environment dependencies are installed.
    """

    path = Path(config_path)
    config: dict[str, Any] = {}
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
            config[current_section][key.strip()] = _parse_config_value(value.strip())

    return config


def _parse_config_value(value: str) -> Any:
    """Parse simple scalar values from the project YAML file."""

    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.isdigit():
        return int(value)
    try:
        return float(value)
    except ValueError:
        return value


def run_ingestion(config_path: str | Path = "config/config.yaml") -> int:
    """Run the dataset ingestion and validation pipeline.

    Args:
        config_path: Path to the project configuration file.

    Returns:
        Process exit code. Returns 0 when validation passes, otherwise 1.
    """

    config = load_config(config_path)
    runtime_config = config.get("runtime", {})
    configure_logging(runtime_config.get("log_level", "INFO"))

    paths = config.get("paths", {})
    raw_dataset_file = Path(
        paths.get("raw_dataset_file", "data/raw/Telco_customer_churn.xlsx")
    )
    validated_dataset_file = Path(
        paths.get(
            "validated_dataset_file", "data/interim/Telco_customer_churn_validated.xlsx"
        )
    )
    profile_report = Path(paths.get("profile_report", "reports/data_profile.md"))
    validation_report = Path(
        paths.get("validation_report", "reports/validation_results.json")
    )

    LOGGER.info("Starting ingestion pipeline")
    dataframe = load_dataset(raw_dataset_file)
    validation_result = validate_dataset(dataframe)
    save_validation_result(validation_result, validation_report)
    save_data_dictionary("docs/data_dictionary.md")

    profile = build_profile_report(dataframe)
    save_profile_report(profile, profile_report)

    if not validation_result.passed:
        LOGGER.error("Validation failed; validated interim dataset was not saved")
        return 1

    validated_dataset_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(raw_dataset_file, validated_dataset_file)
    LOGGER.info("Validated dataset copy saved to %s", validated_dataset_file)
    LOGGER.info("Ingestion pipeline completed successfully")
    return 0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Run raw dataset ingestion and validation."
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to the project YAML configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    raise SystemExit(run_ingestion(args.config))


if __name__ == "__main__":
    main()
