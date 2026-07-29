"""Validate source records and describe row-level quality issues."""

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

_INT64_MAX_TEXT = "9223372036854775807"


@dataclass(frozen=True, slots=True)
class RecordIssue:
    """Describe one data quality issue attached to a source row."""

    source_row_number: int
    inspection_id: str
    rule_id: str
    column: str
    value: str
    message: str


class RecordValidationError(Exception):
    """Raised when record-level validation cannot be performed safely."""


def find_inspection_date_issues(
    data: pd.DataFrame,
    *,
    batch_year: int,
    accepted_formats: Sequence[str],
    primary_key: str = "inspection_id",
    date_column: str = "inspection_date",
) -> tuple[RecordIssue, ...]:
    """Find missing, malformed, or incorrectly partitioned inspection dates.

    Args:
        data: Raw source records.
        batch_year: Year encoded in the incoming batch filename.
        accepted_formats: Date formats allowed by the data contract.
        primary_key: Source column identifying each inspection.
        date_column: Source inspection-date column.

    Returns:
        Immutable issues ordered by their source CSV row number.

    Raises:
        RecordValidationError: If required columns or validation parameters
            are missing or invalid.
    """
    if (
        isinstance(batch_year, bool)
        or not isinstance(batch_year, int)
        or not 1 <= batch_year <= 9999
    ):
        raise RecordValidationError(
            f"Batch year must be an integer between 1 and 9999: {batch_year}"
        )

    formats = list(accepted_formats)

    if not formats:
        raise RecordValidationError(
            "At least one accepted inspection date format is required."
        )

    if not all(
        isinstance(date_format, str) and date_format.strip() for date_format in formats
    ):
        raise RecordValidationError(
            "Accepted inspection date formats must be non-empty strings."
        )

    formats = [date_format.strip() for date_format in formats]

    required_columns = [primary_key, date_column]
    missing_columns = [
        column for column in required_columns if column not in data.columns
    ]

    if missing_columns:
        raise RecordValidationError(
            f"Data is missing required validation columns: {missing_columns}"
        )

    inspection_ids = data[primary_key].astype("string").reset_index(drop=True)
    date_values = data[date_column].astype("string").reset_index(drop=True)

    blank_dates = date_values.isna() | date_values.str.strip().eq("").fillna(True)

    parsed_dates = pd.Series(
        pd.NaT,
        index=date_values.index,
        dtype="datetime64[ns]",
    )

    for date_format in formats:
        remaining = ~blank_dates & parsed_dates.isna()

        if not remaining.any():
            break

        parsed_dates.loc[remaining] = pd.to_datetime(
            date_values.loc[remaining],
            format=date_format,
            errors="coerce",
        )

    issues: list[RecordIssue] = []

    for position in range(len(data)):
        source_row_number = position + 2

        inspection_id_value = inspection_ids.iat[position]
        date_value = date_values.iat[position]

        inspection_id = "" if pd.isna(inspection_id_value) else str(inspection_id_value)
        raw_date = "" if pd.isna(date_value) else str(date_value)

        if bool(blank_dates.iat[position]):
            issues.append(
                RecordIssue(
                    source_row_number=source_row_number,
                    inspection_id=inspection_id,
                    rule_id="inspection_date_required",
                    column=date_column,
                    value=raw_date,
                    message="Inspection date is required.",
                )
            )
            continue

        parsed_date = parsed_dates.iat[position]

        if pd.isna(parsed_date):
            issues.append(
                RecordIssue(
                    source_row_number=source_row_number,
                    inspection_id=inspection_id,
                    rule_id="inspection_date_format",
                    column=date_column,
                    value=raw_date,
                    message=("Inspection date does not match an accepted format."),
                )
            )
            continue

        inspection_year = int(parsed_date.year)

        if inspection_year != batch_year:
            issues.append(
                RecordIssue(
                    source_row_number=source_row_number,
                    inspection_id=inspection_id,
                    rule_id="inspection_year_matches_filename",
                    column=date_column,
                    value=raw_date,
                    message=(
                        f"Inspection year {inspection_year} does not match "
                        f"batch year {batch_year}."
                    ),
                )
            )

    return tuple(issues)


def find_inspection_id_issues(
    data: pd.DataFrame,
    *,
    primary_key: str = "inspection_id",
    minimum: int = 1,
) -> tuple[RecordIssue, ...]:
    """Find missing, malformed, invalid, or duplicated inspection IDs.

    Args:
        data: Raw source records.
        primary_key: Source column identifying each inspection.
        minimum: Lowest numeric identifier accepted by the contract.

    Returns:
        Immutable issues ordered by their source CSV row number.

    Raises:
        RecordValidationError: If the validation parameters or schema
            prevent inspection ID validation.
    """
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
        raise RecordValidationError(
            f"Inspection ID minimum must be a positive integer: {minimum}"
        )

    if primary_key not in data.columns:
        raise RecordValidationError(
            f"Data is missing required validation column: {primary_key}"
        )

    values = data[primary_key].astype("string").reset_index(drop=True)

    parsed_keys: list[int | None] = [None] * len(data)
    row_issues: list[RecordIssue | None] = [None] * len(data)

    for position, value in enumerate(values):
        source_row_number = position + 2
        raw_value = "" if pd.isna(value) else str(value)

        if not raw_value.strip():
            row_issues[position] = RecordIssue(
                source_row_number=source_row_number,
                inspection_id=raw_value,
                rule_id="inspection_id_required",
                column=primary_key,
                value=raw_value,
                message="Inspection ID is required.",
            )
            continue

        if re.fullmatch(r"[0-9]+", raw_value) is None:
            row_issues[position] = RecordIssue(
                source_row_number=source_row_number,
                inspection_id=raw_value,
                rule_id="inspection_id_format",
                column=primary_key,
                value=raw_value,
                message="Inspection ID must contain digits only.",
            )
            continue

        canonical_digits = raw_value.lstrip("0") or "0"

        exceeds_int64 = len(canonical_digits) > len(_INT64_MAX_TEXT) or (
            len(canonical_digits) == len(_INT64_MAX_TEXT)
            and canonical_digits > _INT64_MAX_TEXT
        )

        if exceeds_int64:
            row_issues[position] = RecordIssue(
                source_row_number=source_row_number,
                inspection_id=raw_value,
                rule_id="inspection_id_int64",
                column=primary_key,
                value=raw_value,
                message="Inspection ID cannot be represented as int64.",
            )
            continue

        numeric_value = int(canonical_digits)

        if numeric_value < minimum:
            row_issues[position] = RecordIssue(
                source_row_number=source_row_number,
                inspection_id=raw_value,
                rule_id="inspection_id_minimum",
                column=primary_key,
                value=raw_value,
                message=(f"Inspection ID must be greater than or equal to {minimum}."),
            )
            continue

        parsed_keys[position] = numeric_value

    key_counts = Counter(key for key in parsed_keys if key is not None)

    for position, numeric_value in enumerate(parsed_keys):
        if row_issues[position] is not None:
            continue

        if numeric_value is None or key_counts[numeric_value] == 1:
            continue

        raw_value = str(values.iat[position])

        row_issues[position] = RecordIssue(
            source_row_number=position + 2,
            inspection_id=raw_value,
            rule_id="inspection_id_unique",
            column=primary_key,
            value=raw_value,
            message=(
                f"Inspection ID {numeric_value} appears more than once in the batch."
            ),
        )

    return tuple(issue for issue in row_issues if issue is not None)


def find_inspection_result_issues(
    data: pd.DataFrame,
    *,
    allowed_values: Sequence[str],
    primary_key: str = "inspection_id",
    result_column: str = "results",
) -> tuple[RecordIssue, ...]:
    """Find missing or unknown inspection results.

    Args:
        data: Raw source records.
        allowed_values: Exact result values accepted by the data contract.
        primary_key: Source column identifying each inspection.
        result_column: Source inspection-result column.

    Returns:
        Immutable issues ordered by their source CSV row number.

    Raises:
        RecordValidationError: If the allowed values or required columns
            prevent inspection-result validation.
    """
    if isinstance(allowed_values, (str, bytes)):
        raise RecordValidationError(
            "Allowed inspection results must be a sequence of strings."
        )

    values = list(allowed_values)

    if not values:
        raise RecordValidationError(
            "At least one allowed inspection result is required."
        )

    if not all(isinstance(value, str) and value.strip() for value in values):
        raise RecordValidationError(
            "Allowed inspection results must be non-empty strings."
        )

    if len(values) != len(set(values)):
        raise RecordValidationError(
            "Allowed inspection results contain duplicate values."
        )

    required_columns = [primary_key, result_column]
    missing_columns = [
        column for column in required_columns if column not in data.columns
    ]

    if missing_columns:
        raise RecordValidationError(
            f"Data is missing required validation columns: {missing_columns}"
        )

    allowed_set = set(values)

    inspection_ids = data[primary_key].astype("string").reset_index(drop=True)
    result_values = data[result_column].astype("string").reset_index(drop=True)

    issues: list[RecordIssue] = []

    for position in range(len(data)):
        inspection_id_value = inspection_ids.iat[position]
        result_value = result_values.iat[position]

        inspection_id = "" if pd.isna(inspection_id_value) else str(inspection_id_value)
        raw_result = "" if pd.isna(result_value) else str(result_value)

        if not raw_result.strip():
            issues.append(
                RecordIssue(
                    source_row_number=position + 2,
                    inspection_id=inspection_id,
                    rule_id="inspection_result_required",
                    column=result_column,
                    value=raw_result,
                    message="Inspection result is required.",
                )
            )
            continue

        if raw_result not in allowed_set:
            issues.append(
                RecordIssue(
                    source_row_number=position + 2,
                    inspection_id=inspection_id,
                    rule_id="inspection_result_allowed_values",
                    column=result_column,
                    value=raw_result,
                    message=("Inspection result is not an allowed value."),
                )
            )

    return tuple(issues)
