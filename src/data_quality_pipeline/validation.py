"""Validate source records and describe row-level quality issues."""

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Literal

import pandas as pd

_INT64_MAX_TEXT = "9223372036854775807"


IssueSeverity = Literal["error", "warning"]


@dataclass(frozen=True, slots=True)
class RecordIssue:
    """Describe one data quality issue attached to a source row."""

    source_row_number: int
    inspection_id: str
    rule_id: str
    column: str
    value: str
    message: str
    severity: IssueSeverity = "error"

    def __post_init__(self) -> None:
        """Reject unsupported issue severities."""
        if not isinstance(self.severity, str) or self.severity not in (
            "error",
            "warning",
        ):
            raise ValueError("Record issue severity must be 'error' or 'warning'.")


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


def find_coordinate_issues(
    data: pd.DataFrame,
    *,
    primary_key: str = "inspection_id",
    latitude_column: str = "latitude",
    longitude_column: str = "longitude",
    latitude_minimum: float = -90,
    latitude_maximum: float = 90,
    longitude_minimum: float = -180,
    longitude_maximum: float = 180,
) -> tuple[RecordIssue, ...]:
    """Find inconsistent, malformed, or out-of-range coordinates.

    Args:
        data: Raw source records.
        primary_key: Source column identifying each inspection.
        latitude_column: Source latitude column.
        longitude_column: Source longitude column.
        latitude_minimum: Lowest latitude accepted by the contract.
        latitude_maximum: Highest latitude accepted by the contract.
        longitude_minimum: Lowest longitude accepted by the contract.
        longitude_maximum: Highest longitude accepted by the contract.

    Returns:
        Immutable issues ordered by source CSV row and validation rule.

    Raises:
        RecordValidationError: If coordinate bounds or required columns
            prevent validation.
    """
    latitude_bounds = _validate_coordinate_bounds(
        minimum=latitude_minimum,
        maximum=latitude_maximum,
        coordinate_name="Latitude",
    )
    longitude_bounds = _validate_coordinate_bounds(
        minimum=longitude_minimum,
        maximum=longitude_maximum,
        coordinate_name="Longitude",
    )

    required_columns = [
        primary_key,
        latitude_column,
        longitude_column,
    ]
    missing_columns = [
        column for column in required_columns if column not in data.columns
    ]

    if missing_columns:
        raise RecordValidationError(
            f"Data is missing required validation columns: {missing_columns}"
        )

    inspection_ids = data[primary_key].astype("string").reset_index(drop=True)
    latitude_values = data[latitude_column].astype("string").reset_index(drop=True)
    longitude_values = data[longitude_column].astype("string").reset_index(drop=True)

    issues: list[RecordIssue] = []

    for position in range(len(data)):
        source_row_number = position + 2

        inspection_id_value = inspection_ids.iat[position]
        latitude_value = latitude_values.iat[position]
        longitude_value = longitude_values.iat[position]

        inspection_id = "" if pd.isna(inspection_id_value) else str(inspection_id_value)
        raw_latitude = "" if pd.isna(latitude_value) else str(latitude_value)
        raw_longitude = "" if pd.isna(longitude_value) else str(longitude_value)

        latitude_blank = not raw_latitude.strip()
        longitude_blank = not raw_longitude.strip()

        if latitude_blank != longitude_blank:
            issues.append(
                RecordIssue(
                    source_row_number=source_row_number,
                    inspection_id=inspection_id,
                    rule_id="coordinate_pair_consistency",
                    column=f"{latitude_column},{longitude_column}",
                    value=(
                        f"{latitude_column}={raw_latitude!r}; "
                        f"{longitude_column}={raw_longitude!r}"
                    ),
                    message=(
                        "Latitude and longitude must both be present or both be null."
                    ),
                )
            )

        if not latitude_blank:
            issues.extend(
                _find_coordinate_value_issues(
                    source_row_number=source_row_number,
                    inspection_id=inspection_id,
                    column=latitude_column,
                    raw_value=raw_latitude,
                    coordinate_name="Latitude",
                    minimum=latitude_bounds[0],
                    maximum=latitude_bounds[1],
                )
            )

        if not longitude_blank:
            issues.extend(
                _find_coordinate_value_issues(
                    source_row_number=source_row_number,
                    inspection_id=inspection_id,
                    column=longitude_column,
                    raw_value=raw_longitude,
                    coordinate_name="Longitude",
                    minimum=longitude_bounds[0],
                    maximum=longitude_bounds[1],
                )
            )

    return tuple(issues)


def _validate_coordinate_bounds(
    *,
    minimum: float,
    maximum: float,
    coordinate_name: str,
) -> tuple[float, float]:
    """Validate and normalize one configured coordinate range."""
    if (
        isinstance(minimum, bool)
        or not isinstance(minimum, (int, float))
        or isinstance(maximum, bool)
        or not isinstance(maximum, (int, float))
    ):
        raise RecordValidationError(f"{coordinate_name} bounds must be numeric.")

    normalized_minimum = float(minimum)
    normalized_maximum = float(maximum)

    if not isfinite(normalized_minimum) or not isfinite(normalized_maximum):
        raise RecordValidationError(f"{coordinate_name} bounds must be finite.")

    if normalized_minimum >= normalized_maximum:
        raise RecordValidationError(
            f"{coordinate_name} minimum must be lower than its maximum."
        )

    return normalized_minimum, normalized_maximum


def _find_coordinate_value_issues(
    *,
    source_row_number: int,
    inspection_id: str,
    column: str,
    raw_value: str,
    coordinate_name: str,
    minimum: float,
    maximum: float,
) -> list[RecordIssue]:
    """Validate one non-blank coordinate value."""
    try:
        numeric_value = float(raw_value)
    except ValueError:
        numeric_value = float("nan")

    if not isfinite(numeric_value):
        return [
            RecordIssue(
                source_row_number=source_row_number,
                inspection_id=inspection_id,
                rule_id=f"{column}_numeric",
                column=column,
                value=raw_value,
                message=f"{coordinate_name} must be a finite number.",
            )
        ]

    if not minimum <= numeric_value <= maximum:
        return [
            RecordIssue(
                source_row_number=source_row_number,
                inspection_id=inspection_id,
                rule_id=f"{column}_range",
                column=column,
                value=raw_value,
                message=(
                    f"{coordinate_name} must be between {minimum:g} and {maximum:g}."
                ),
            )
        ]

    return []


def find_string_pattern_issues(
    data: pd.DataFrame,
    *,
    column: str,
    pattern: str,
    nullable: bool,
    primary_key: str = "inspection_id",
) -> tuple[RecordIssue, ...]:
    """Find missing or incorrectly formatted string values.

    Args:
        data: Raw source records.
        column: Source column validated against the pattern.
        pattern: Regular expression that must match the entire value.
        nullable: Whether blank values are accepted.
        primary_key: Source column identifying each inspection.

    Returns:
        Immutable issues ordered by their source CSV row number.

    Raises:
        RecordValidationError: If validation parameters or required columns
            prevent pattern validation.
    """
    if not isinstance(column, str) or not column.strip():
        raise RecordValidationError(
            "Pattern validation column must be a non-empty string."
        )

    if not isinstance(primary_key, str) or not primary_key.strip():
        raise RecordValidationError(
            "Pattern validation primary key must be a non-empty string."
        )

    if not isinstance(pattern, str) or not pattern:
        raise RecordValidationError("Validation pattern must be a non-empty string.")

    if not isinstance(nullable, bool):
        raise RecordValidationError(
            "Pattern validation nullable setting must be boolean."
        )

    try:
        compiled_pattern = re.compile(pattern)
    except re.error as exc:
        raise RecordValidationError(
            f"Invalid validation pattern for {column}: {pattern}"
        ) from exc

    required_columns = [primary_key, column]
    missing_columns = [
        required_column
        for required_column in required_columns
        if required_column not in data.columns
    ]

    if missing_columns:
        raise RecordValidationError(
            f"Data is missing required validation columns: {missing_columns}"
        )

    inspection_ids = data[primary_key].astype("string").reset_index(drop=True)
    values = data[column].astype("string").reset_index(drop=True)

    issues: list[RecordIssue] = []

    for position in range(len(data)):
        inspection_id_value = inspection_ids.iat[position]
        value = values.iat[position]

        inspection_id = "" if pd.isna(inspection_id_value) else str(inspection_id_value)
        raw_value = "" if pd.isna(value) else str(value)

        if not raw_value.strip():
            if not nullable:
                issues.append(
                    RecordIssue(
                        source_row_number=position + 2,
                        inspection_id=inspection_id,
                        rule_id=f"{column}_required",
                        column=column,
                        value=raw_value,
                        message=f"{column} is required.",
                    )
                )

            continue

        if compiled_pattern.fullmatch(raw_value) is None:
            issues.append(
                RecordIssue(
                    source_row_number=position + 2,
                    inspection_id=inspection_id,
                    rule_id=f"{column}_pattern",
                    column=column,
                    value=raw_value,
                    message=(f"{column} does not match the required pattern."),
                )
            )

    return tuple(issues)
