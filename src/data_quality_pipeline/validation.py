"""Validate source records and describe row-level quality issues."""

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd


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
