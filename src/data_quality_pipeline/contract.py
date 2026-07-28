"""Load and validate the data contract."""

from pathlib import Path
from typing import Any

import yaml

type ContractData = dict[str, Any]

_REQUIRED_SECTIONS = frozenset(
    {
        "contract",
        "source_schema",
        "columns",
        "record_rules",
        "batch_rules",
    }
)


class DataContractError(Exception):
    """Raised when the data contract is missing or structurally invalid."""


def load_data_contract(contract_path: str | Path) -> ContractData:
    """Load a YAML data contract and validate its main invariants.

    Args:
        contract_path: Path to the YAML data contract.

    Returns:
        The parsed and validated data contract.

    Raises:
        DataContractError: If the contract is missing, malformed,
            unreadable, or structurally inconsistent.
    """
    path = Path(contract_path)

    if not path.is_file():
        raise DataContractError(f"Data contract file not found: {path}")

    try:
        with path.open(encoding="utf-8") as contract_file:
            loaded_contract = yaml.safe_load(contract_file)
    except yaml.YAMLError as exc:
        raise DataContractError(f"Invalid YAML data contract: {path}") from exc
    except OSError as exc:
        raise DataContractError(f"Unable to read data contract: {path}") from exc

    if not isinstance(loaded_contract, dict):
        raise DataContractError("The data contract root must be a YAML mapping.")

    missing_sections = _REQUIRED_SECTIONS.difference(loaded_contract)

    if missing_sections:
        formatted_sections = ", ".join(sorted(missing_sections))
        raise DataContractError(
            f"Missing required data contract sections: {formatted_sections}"
        )

    _validate_schema(loaded_contract)
    _validate_rule_ids(loaded_contract)

    return loaded_contract


def _validate_schema(contract: ContractData) -> None:
    """Validate column definitions and primary-key consistency."""
    source_schema = contract["source_schema"]
    columns = contract["columns"]
    contract_metadata = contract["contract"]

    if not isinstance(source_schema, dict):
        raise DataContractError("The source_schema section must be a YAML mapping.")

    expected_columns = source_schema.get("expected_columns")

    if not isinstance(expected_columns, list) or not expected_columns:
        raise DataContractError(
            "source_schema.expected_columns must be a non-empty list."
        )

    if not all(isinstance(column, str) for column in expected_columns):
        raise DataContractError("Every expected source column must be a string.")

    if len(expected_columns) != len(set(expected_columns)):
        raise DataContractError(
            "source_schema.expected_columns contains duplicate names."
        )

    if not isinstance(columns, dict) or not columns:
        raise DataContractError("The columns section must be a non-empty YAML mapping.")

    expected_set = set(expected_columns)
    defined_set = set(columns)

    if expected_set != defined_set:
        missing_definitions = sorted(expected_set - defined_set)
        unexpected_definitions = sorted(defined_set - expected_set)

        raise DataContractError(
            "Source schema and column definitions do not match. "
            f"Missing definitions: {missing_definitions}. "
            f"Unexpected definitions: {unexpected_definitions}."
        )

    primary_key = contract_metadata.get("primary_key")

    if primary_key not in expected_set:
        raise DataContractError(
            f"Primary key is not defined in the source schema: {primary_key}"
        )

    target_names = [definition.get("target_name") for definition in columns.values()]

    if not all(
        isinstance(target_name, str) and target_name for target_name in target_names
    ):
        raise DataContractError(
            "Every column definition must contain a non-empty target_name."
        )

    if len(target_names) != len(set(target_names)):
        raise DataContractError("Column target names must be unique.")


def _validate_rule_ids(contract: ContractData) -> None:
    """Validate record-rule and batch-rule identifiers."""
    all_rule_ids: list[str] = []

    for section_name in ("record_rules", "batch_rules"):
        rules = contract[section_name]

        if not isinstance(rules, list):
            raise DataContractError(f"The {section_name} section must be a YAML list.")

        for rule in rules:
            if not isinstance(rule, dict):
                raise DataContractError(
                    f"Every rule in {section_name} must be a mapping."
                )

            rule_id = rule.get("id")

            if not isinstance(rule_id, str) or not rule_id:
                raise DataContractError(
                    f"Every rule in {section_name} must have a non-empty id."
                )

            all_rule_ids.append(rule_id)

    if len(all_rule_ids) != len(set(all_rule_ids)):
        raise DataContractError("Data contract rule identifiers must be unique.")
