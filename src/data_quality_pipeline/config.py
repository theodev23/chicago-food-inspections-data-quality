"""Load and validate the pipeline configuration."""

from pathlib import Path
from typing import Any

import yaml

type ConfigData = dict[str, Any]

_REQUIRED_SECTIONS = frozenset(
    {
        "project",
        "source",
        "paths",
        "ingestion",
        "output",
    }
)


class ConfigurationError(Exception):
    """Raised when the pipeline configuration is missing or invalid."""


def load_config(config_path: str | Path) -> ConfigData:
    """Load a YAML configuration file and validate its main sections.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        The parsed configuration dictionary.

    Raises:
        ConfigurationError: If the file is missing, unreadable, malformed,
            or does not contain the required sections.
    """
    path = Path(config_path)

    if not path.is_file():
        raise ConfigurationError(f"Configuration file not found: {path}")

    try:
        with path.open(encoding="utf-8") as config_file:
            loaded_config = yaml.safe_load(config_file)
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML configuration file: {path}") from exc
    except OSError as exc:
        raise ConfigurationError(f"Unable to read configuration file: {path}") from exc

    if not isinstance(loaded_config, dict):
        raise ConfigurationError("The configuration root must be a YAML mapping.")

    missing_sections = _REQUIRED_SECTIONS.difference(loaded_config)

    if missing_sections:
        formatted_sections = ", ".join(sorted(missing_sections))
        raise ConfigurationError(
            f"Missing required configuration sections: {formatted_sections}"
        )

    return loaded_config
