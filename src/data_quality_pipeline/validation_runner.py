"""Run contract-driven record validation for one incoming batch."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from data_quality_pipeline.duplicates import (
    DuplicateDetectionError,
    ExactDuplicateRecord,
    find_exact_duplicates,
)
from data_quality_pipeline.validation import (
    RecordIssue,
    find_coordinate_issues,
    find_inspection_date_issues,
    find_inspection_id_issues,
    find_inspection_result_issues,
    find_license_issues,
    find_risk_issues,
    find_string_pattern_issues,
)


@dataclass(frozen=True, slots=True)
class BatchValidationResult:
    """Contain all deterministic validation findings for one source batch."""

    issues: tuple[RecordIssue, ...]
    exact_duplicates: tuple[ExactDuplicateRecord, ...]
    duplicate_detection_skipped: bool

    @property
    def errors(self) -> tuple[RecordIssue, ...]:
        """Return blocking data quality issues."""
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[RecordIssue, ...]:
        """Return non-blocking data quality findings."""
        return tuple(issue for issue in self.issues if issue.severity == "warning")

    @property
    def has_errors(self) -> bool:
        """Indicate whether the batch contains a blocking issue."""
        return bool(self.errors)


class BatchValidationError(Exception):
    """Raised when contract-driven batch validation cannot run safely."""


def validate_batch_records(
    data: pd.DataFrame,
    *,
    batch_year: int,
    contract: Mapping[str, Any],
) -> BatchValidationResult:
    """Run all current record-level rules for one incoming batch.

    Exact duplicate detection requires a fully valid primary key. It is
    skipped when primary-key validation produces any issue, while all other
    applicable validators still run.

    Args:
        data: Raw source records.
        batch_year: Year encoded in the incoming filename.
        contract: Validated data contract.

    Returns:
        Immutable validation findings and exact duplicate references.

    Raises:
        BatchValidationError: If exact duplicate detection fails after the
            primary key has passed validation.
    """
    contract_metadata = contract["contract"]
    source_schema = contract["source_schema"]
    columns = contract["columns"]

    primary_key = contract_metadata["primary_key"]
    source_columns = source_schema["expected_columns"]

    inspection_id_contract = columns[primary_key]
    inspection_date_contract = columns["inspection_date"]
    result_contract = columns["results"]
    license_contract = columns["license_"]
    risk_contract = columns["risk"]
    state_contract = columns["state"]
    zip_contract = columns["zip"]
    latitude_contract = columns["latitude"]
    longitude_contract = columns["longitude"]

    inspection_id_issues = find_inspection_id_issues(
        data,
        primary_key=primary_key,
        minimum=inspection_id_contract["validations"]["minimum"],
    )

    issues: list[RecordIssue] = list(inspection_id_issues)

    issues.extend(
        find_inspection_date_issues(
            data,
            batch_year=batch_year,
            accepted_formats=inspection_date_contract["accepted_formats"],
            primary_key=primary_key,
        )
    )

    issues.extend(
        find_inspection_result_issues(
            data,
            allowed_values=result_contract["allowed_values"],
            primary_key=primary_key,
        )
    )

    issues.extend(
        find_coordinate_issues(
            data,
            primary_key=primary_key,
            latitude_minimum=latitude_contract["validations"]["minimum"],
            latitude_maximum=latitude_contract["validations"]["maximum"],
            longitude_minimum=longitude_contract["validations"]["minimum"],
            longitude_maximum=longitude_contract["validations"]["maximum"],
        )
    )

    for column_name, column_contract in (
        ("state", state_contract),
        ("zip", zip_contract),
    ):
        issues.extend(
            find_string_pattern_issues(
                data,
                column=column_name,
                pattern=column_contract["validations"]["pattern"],
                nullable=column_contract["nullable"],
                primary_key=primary_key,
            )
        )

    issues.extend(
        find_license_issues(
            data,
            nullable=license_contract["nullable"],
            zero_as_null=license_contract["cleaning"]["zero_as_null"],
            format_when_present=(
                license_contract["validations"]["format_when_present"]
            ),
            primary_key=primary_key,
        )
    )

    issues.extend(
        find_risk_issues(
            data,
            allowed_values=risk_contract["allowed_values"],
            unknown_value_severity=risk_contract["unknown_value_severity"],
            primary_key=primary_key,
        )
    )

    exact_duplicates: tuple[ExactDuplicateRecord, ...] = ()
    duplicate_detection_skipped = bool(inspection_id_issues)

    if not duplicate_detection_skipped:
        try:
            exact_duplicates = find_exact_duplicates(
                data,
                primary_key=primary_key,
                source_columns=source_columns,
            )
        except DuplicateDetectionError as exc:
            raise BatchValidationError(
                "Exact duplicate detection failed after primary-key "
                "validation succeeded."
            ) from exc

        duplicate_rule = _find_record_rule(
            contract,
            rule_id="exact_duplicate_record",
        )

        inspection_ids = data[primary_key].astype("string").reset_index(drop=True)
        source_row_by_id = {
            int(inspection_id): position + 2
            for position, inspection_id in enumerate(inspection_ids)
        }

        for duplicate in exact_duplicates:
            issues.append(
                RecordIssue(
                    source_row_number=source_row_by_id[duplicate.inspection_id],
                    inspection_id=str(duplicate.inspection_id),
                    rule_id=duplicate_rule["id"],
                    column=primary_key,
                    value=str(duplicate.inspection_id),
                    message=(
                        "Record is an exact duplicate of "
                        f"{primary_key} "
                        f"{duplicate.duplicate_of_inspection_id}."
                    ),
                    severity=duplicate_rule["severity"],
                )
            )

    ordered_issues = tuple(
        sorted(
            issues,
            key=lambda issue: (
                issue.source_row_number,
                issue.rule_id,
                issue.column,
                issue.value,
            ),
        )
    )

    return BatchValidationResult(
        issues=ordered_issues,
        exact_duplicates=exact_duplicates,
        duplicate_detection_skipped=duplicate_detection_skipped,
    )


def _find_record_rule(
    contract: Mapping[str, Any],
    *,
    rule_id: str,
) -> Mapping[str, Any]:
    """Return one named record rule from the validated contract."""
    for rule in contract["record_rules"]:
        if rule["id"] == rule_id:
            return rule

    raise BatchValidationError(f"Data contract is missing record rule: {rule_id}")
