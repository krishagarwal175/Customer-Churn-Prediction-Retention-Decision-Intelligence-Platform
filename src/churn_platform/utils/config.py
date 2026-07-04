from __future__ import annotations

import logging
from pathlib import Path

from typing import Any

import yaml

logger = logging.getLogger(__name__)

__all__ = ["load_config"]


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file into a dictionary.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        A dictionary containing the parsed configuration.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        IsADirectoryError: If the provided path points to a directory.
        ValueError: If the YAML is invalid or the parsed content is not a
            dictionary.
        OSError: If the file cannot be read.
    """
    path = Path(config_path).expanduser().resolve()

    logger.info("Loading configuration from %s", path)

    if not path.exists():
        logger.error("Configuration file does not exist: %s", path)
        raise FileNotFoundError(f"Configuration file not found: {path}")

    if path.is_dir():
        logger.error("Configuration path points to a directory, not a file: %s", path)
        raise IsADirectoryError(f"Configuration path is a directory: {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        logger.error("Invalid YAML syntax in configuration file: %s", path)
        raise ValueError(f"Invalid YAML syntax in configuration file: {path}") from exc
    except OSError as exc:
        logger.error("Failed to read configuration file: %s", path)
        raise OSError(f"Failed to read configuration file: {path}") from exc

    if data is None:
        logger.warning("Configuration file is empty: %s", path)
        return {}

    if not isinstance(data, dict):
        logger.error(
            "Configuration content must be a dictionary at the top level: %s", path
        )
        raise ValueError(
            "Configuration content must be a dictionary at the top level: " f"{path}"
        )

    logger.info("Configuration loaded successfully from %s", path)
    return data
