"""Build issue-level quarantine records from rejected source rows."""

from collections.abc import Sequence

import pandas as pd

from data_quality_pipeline.validation import RecordIssue
from data_quality_pipeline.validation_runner import BatchValidationResult

QUARANTINE_METADATA_COLUMNS: tuple[str, ...] = (
    "dq_source_row_number",
    "dq_rule_id",
    "dq_column",
    "dq_value",
    "dq_message",
    "dq_severity",
    "dq_duplicate_of_inspection_id",
    "dq_batch_year",
)


class QuarantineBuildError(Exception):
    """Raised when rejected records cannot be mapped to source rows safely."""


def build_quarantine_records(
    data: pd.DataFrame,
    *,
    validation_result: BatchValidationResult,
    batch_year: int,
    primary_key: str,
) -> pd.DataFrame:
    """Build one quarantine row for each blocking validation issue.

    Every output row contains the original source columns followed by
    diagnostic metadata prefixed with ``dq_``. Warnings are excluded.

    ``RecordIssue.source_row_number`` uses the physical CSV line number,
    including the header. Therefore, source records are selected with
    ``source_row_number - 2``.

    Args:
        data: Raw source records in their original column order.
        validation_result: Complete record-level validation result.
        batch_year: Year derived from the incoming filename.
        primary_key: Source primary-key column used to verify row mapping.

    Returns:
        A new DataFrame containing one row per blocking issue.

    Raises:
        QuarantineBuildError: If schema, row numbers, identifiers, or
            duplicate references are inconsistent.
    """
    _validate_batch_year(batch_year)
    _validate_source_schema(data, primary_key=primary_key)

    errors = sorted(
        validation_result.errors,
        key=_issue_sort_key,
    )
    duplicate_references = _build_duplicate_reference_map(validation_result)

    positions: list[int] = []
    duplicate_of_values: list[int | None] = []

    for issue in errors:
        position = _source_position(
            issue,
            source_row_count=len(data),
        )
        source_identifier = _string_value(data.iloc[position][primary_key])

        if source_identifier != issue.inspection_id:
            raise QuarantineBuildError(
                "Validation issue does not match its source row: "
                f"CSV line {issue.source_row_number}, "
                f"expected {issue.inspection_id!r}, "
                f"found {source_identifier!r}."
            )

        duplicate_of = duplicate_references.get(issue.inspection_id)

        if issue.rule_id == "exact_duplicate_record" and duplicate_of is None:
            raise QuarantineBuildError(
                "Exact duplicate issue is missing its retained-record "
                f"reference: inspection_id {issue.inspection_id!r}."
            )

        positions.append(position)
        duplicate_of_values.append(duplicate_of)

    quarantine = data.iloc[positions].copy().reset_index(drop=True)

    quarantine["dq_source_row_number"] = pd.Series(
        [issue.source_row_number for issue in errors],
        dtype="int64",
    )
    quarantine["dq_rule_id"] = _string_series([issue.rule_id for issue in errors])
    quarantine["dq_column"] = _string_series([issue.column for issue in errors])
    quarantine["dq_value"] = _string_series([issue.value for issue in errors])
    quarantine["dq_message"] = _string_series([issue.message for issue in errors])
    quarantine["dq_severity"] = _string_series([issue.severity for issue in errors])
    quarantine["dq_duplicate_of_inspection_id"] = pd.Series(
        duplicate_of_values,
        dtype="Int64",
    )
    quarantine["dq_batch_year"] = pd.Series(
        [batch_year] * len(errors),
        dtype="int16",
    )

    return quarantine


def _validate_batch_year(batch_year: int) -> None:
    """Validate the batch partition year."""
    if (
        isinstance(batch_year, bool)
        or not isinstance(batch_year, int)
        or not 1 <= batch_year <= 9999
    ):
        raise QuarantineBuildError("Batch year must be an integer between 1 and 9999.")


def _validate_source_schema(
    data: pd.DataFrame,
    *,
    primary_key: str,
) -> None:
    """Validate source columns required for deterministic row mapping."""
    if not data.columns.is_unique:
        raise QuarantineBuildError("Source data contains duplicate column names.")

    if not isinstance(primary_key, str) or not primary_key:
        raise QuarantineBuildError("Primary key must be a non-empty string.")

    if primary_key not in data.columns:
        raise QuarantineBuildError(f"Source data is missing primary key: {primary_key}")

    collisions = sorted(set(data.columns).intersection(QUARANTINE_METADATA_COLUMNS))

    if collisions:
        raise QuarantineBuildError(
            f"Source columns conflict with quarantine metadata: {collisions}"
        )


def _build_duplicate_reference_map(
    validation_result: BatchValidationResult,
) -> dict[str, int]:
    """Map rejected duplicate identifiers to retained identifiers."""
    references: dict[str, int] = {}

    for duplicate in validation_result.exact_duplicates:
        inspection_id = str(duplicate.inspection_id)
        retained_id = duplicate.duplicate_of_inspection_id
        existing = references.get(inspection_id)

        if existing is not None and existing != retained_id:
            raise QuarantineBuildError(
                f"Conflicting duplicate references for inspection_id {inspection_id!r}."
            )

        references[inspection_id] = retained_id

    return references


def _source_position(
    issue: RecordIssue,
    *,
    source_row_count: int,
) -> int:
    """Convert a physical CSV line number to a DataFrame position."""
    position = issue.source_row_number - 2

    if not 0 <= position < source_row_count:
        raise QuarantineBuildError(
            "Validation issue references an unavailable CSV line: "
            f"{issue.source_row_number}."
        )

    return position


def _issue_sort_key(
    issue: RecordIssue,
) -> tuple[int, str, str, str]:
    """Return a deterministic quarantine ordering key."""
    return (
        issue.source_row_number,
        issue.rule_id,
        issue.column,
        issue.inspection_id,
    )


def _string_value(value: object) -> str:
    """Convert a raw source identifier to its validation representation."""
    if pd.isna(value):
        return ""

    return str(value)


def _string_series(values: Sequence[str]) -> pd.Series:
    """Create a pandas string column with stable empty-output typing."""
    return pd.Series(values, dtype="string")
