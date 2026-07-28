"""Unit tests for data contract loading and validation."""

from pathlib import Path
from typing import Any

import pytest
import yaml

from data_quality_pipeline.contract import (
    DataContractError,
    load_data_contract,
)


def _valid_contract() -> dict[str, Any]:
    """Return a minimal structurally valid data contract."""
    return {
        "contract": {
            "name": "example_contract",
            "version": "1.0.0",
            "primary_key": "inspection_id",
            "source_format": "csv",
        },
        "source_schema": {
            "expected_columns": [
                "inspection_id",
                "results",
            ],
        },
        "columns": {
            "inspection_id": {
                "target_name": "inspection_id",
                "target_type": "int64",
                "nullable": False,
            },
            "results": {
                "target_name": "inspection_result",
                "target_type": "string",
                "nullable": False,
            },
        },
        "record_rules": [
            {
                "id": "record_rule",
                "description": "Example record rule.",
                "severity": "error",
                "action": "quarantine",
            },
        ],
        "batch_rules": [
            {
                "id": "batch_rule",
                "description": "Example batch rule.",
                "severity": "error",
                "action": "fail_batch",
            },
        ],
    }


def _write_contract(
    tmp_path: Path,
    contract: dict[str, Any],
) -> Path:
    """Write a synthetic contract to a temporary YAML file."""
    contract_path = tmp_path / "data_contract.yaml"
    contract_path.write_text(
        yaml.safe_dump(
            contract,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return contract_path


def test_load_data_contract_returns_valid_mapping(
    tmp_path: Path,
) -> None:
    """A structurally valid contract should be loaded."""
    contract_path = _write_contract(
        tmp_path,
        _valid_contract(),
    )

    contract = load_data_contract(contract_path)

    assert contract["contract"]["name"] == "example_contract"
    assert contract["contract"]["primary_key"] == "inspection_id"


def test_load_data_contract_raises_error_when_file_is_missing(
    tmp_path: Path,
) -> None:
    """A missing contract file should raise a clear error."""
    missing_path = tmp_path / "missing.yaml"

    with pytest.raises(
        DataContractError,
        match="Data contract file not found",
    ):
        load_data_contract(missing_path)


def test_load_data_contract_raises_error_for_invalid_yaml(
    tmp_path: Path,
) -> None:
    """Malformed YAML should raise a data contract error."""
    contract_path = tmp_path / "invalid.yaml"
    contract_path.write_text("contract: [", encoding="utf-8")

    with pytest.raises(
        DataContractError,
        match="Invalid YAML data contract",
    ):
        load_data_contract(contract_path)


def test_load_data_contract_rejects_non_mapping_root(
    tmp_path: Path,
) -> None:
    """The YAML root must be a key-value mapping."""
    contract_path = tmp_path / "list.yaml"
    contract_path.write_text(
        "- contract\n- source_schema\n",
        encoding="utf-8",
    )

    with pytest.raises(
        DataContractError,
        match="data contract root must be a YAML mapping",
    ):
        load_data_contract(contract_path)


def test_load_data_contract_reports_missing_sections(
    tmp_path: Path,
) -> None:
    """Missing required top-level sections should be reported."""
    contract_path = tmp_path / "incomplete.yaml"
    contract_path.write_text(
        """
contract:
  name: example
source_schema:
  expected_columns:
    - inspection_id
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(DataContractError) as error:
        load_data_contract(contract_path)

    message = str(error.value)

    assert "Missing required data contract sections" in message
    assert "columns" in message
    assert "record_rules" in message
    assert "batch_rules" in message


def test_load_data_contract_rejects_schema_definition_mismatch(
    tmp_path: Path,
) -> None:
    """Every expected source column must have one definition."""
    contract = _valid_contract()
    contract["columns"].pop("results")
    contract_path = _write_contract(tmp_path, contract)

    with pytest.raises(
        DataContractError,
        match="Source schema and column definitions do not match",
    ):
        load_data_contract(contract_path)


def test_load_data_contract_rejects_unknown_primary_key(
    tmp_path: Path,
) -> None:
    """The primary key must exist in the source schema."""
    contract = _valid_contract()
    contract["contract"]["primary_key"] = "unknown_column"
    contract_path = _write_contract(tmp_path, contract)

    with pytest.raises(
        DataContractError,
        match="Primary key is not defined in the source schema",
    ):
        load_data_contract(contract_path)


def test_load_data_contract_rejects_duplicate_target_names(
    tmp_path: Path,
) -> None:
    """Two source columns must not share the same target name."""
    contract = _valid_contract()
    contract["columns"]["results"]["target_name"] = "inspection_id"
    contract_path = _write_contract(tmp_path, contract)

    with pytest.raises(
        DataContractError,
        match="Column target names must be unique",
    ):
        load_data_contract(contract_path)


def test_load_data_contract_rejects_duplicate_rule_ids(
    tmp_path: Path,
) -> None:
    """Rule identifiers must be unique across all rule sections."""
    contract = _valid_contract()
    contract["batch_rules"][0]["id"] = "record_rule"
    contract_path = _write_contract(tmp_path, contract)

    with pytest.raises(
        DataContractError,
        match="Data contract rule identifiers must be unique",
    ):
        load_data_contract(contract_path)
