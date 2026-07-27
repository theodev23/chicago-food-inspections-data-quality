"""Unit tests for pipeline configuration loading."""

from pathlib import Path

import pytest

from data_quality_pipeline.config import ConfigurationError, load_config


def test_load_config_returns_valid_mapping(tmp_path: Path) -> None:
    """A valid configuration should be loaded as a dictionary."""
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text(
        """
project: {}
source: {}
paths: {}
ingestion: {}
output: {}
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config["project"] == {}
    assert config["source"] == {}
    assert config["paths"] == {}
    assert config["ingestion"] == {}
    assert config["output"] == {}


def test_load_config_raises_error_when_file_is_missing(
    tmp_path: Path,
) -> None:
    """A missing configuration file should raise a clear error."""
    missing_path = tmp_path / "missing.yaml"

    with pytest.raises(
        ConfigurationError,
        match="Configuration file not found",
    ):
        load_config(missing_path)


def test_load_config_raises_error_for_invalid_yaml(
    tmp_path: Path,
) -> None:
    """Malformed YAML should raise a configuration error."""
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("project: [", encoding="utf-8")

    with pytest.raises(
        ConfigurationError,
        match="Invalid YAML configuration file",
    ):
        load_config(config_path)


def test_load_config_rejects_non_mapping_root(tmp_path: Path) -> None:
    """The YAML root must be a key-value mapping."""
    config_path = tmp_path / "list.yaml"
    config_path.write_text("- project\n- source\n", encoding="utf-8")

    with pytest.raises(
        ConfigurationError,
        match="configuration root must be a YAML mapping",
    ):
        load_config(config_path)


def test_load_config_reports_missing_sections(tmp_path: Path) -> None:
    """Missing required sections should be listed in the error."""
    config_path = tmp_path / "incomplete.yaml"
    config_path.write_text(
        """
project:
  name: example
source: {}
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError) as error:
        load_config(config_path)

    message = str(error.value)

    assert "Missing required configuration sections" in message
    assert "paths" in message
    assert "ingestion" in message
    assert "output" in message
