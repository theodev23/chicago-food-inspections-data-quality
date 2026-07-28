"""Unit tests for record-level data quality validation."""

from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from data_quality_pipeline.validation import (
    RecordIssue,
    RecordValidationError,
    find_inspection_date_issues,
    find_inspection_id_issues,
)


def _frame(rows: list[dict[str, str]]) -> pd.DataFrame:
    """Create a source-like DataFrame using pandas string columns."""
    return pd.DataFrame(rows, dtype="string")


def test_find_inspection_date_issues_returns_empty_tuple_for_valid_dates() -> None:
    """Valid inspection dates from the batch year should produce no issues."""
    data = _frame(
        [
            {
                "inspection_id": "1",
                "inspection_date": "2019-01-15T00:00:00.000",
            },
            {
                "inspection_id": "2",
                "inspection_date": "2019-12-31T00:00:00.000",
            },
        ]
    )

    issues = find_inspection_date_issues(
        data,
        batch_year=2019,
        accepted_formats=["%Y-%m-%dT%H:%M:%S.%f"],
    )

    assert issues == ()


def test_find_inspection_date_issues_classifies_row_failures() -> None:
    """Missing, malformed, and wrong-year dates should be distinguished."""
    data = _frame(
        [
            {
                "inspection_id": "10",
                "inspection_date": "",
            },
            {
                "inspection_id": "11",
                "inspection_date": "not-a-date",
            },
            {
                "inspection_id": "12",
                "inspection_date": "2020-02-03T00:00:00.000",
            },
            {
                "inspection_id": "13",
                "inspection_date": "2019-04-05T00:00:00.000",
            },
        ]
    )
    data.index = [50, 60, 70, 80]

    issues = find_inspection_date_issues(
        data,
        batch_year=2019,
        accepted_formats=["%Y-%m-%dT%H:%M:%S.%f"],
    )

    assert issues == (
        RecordIssue(
            source_row_number=2,
            inspection_id="10",
            rule_id="inspection_date_required",
            column="inspection_date",
            value="",
            message="Inspection date is required.",
        ),
        RecordIssue(
            source_row_number=3,
            inspection_id="11",
            rule_id="inspection_date_format",
            column="inspection_date",
            value="not-a-date",
            message="Inspection date does not match an accepted format.",
        ),
        RecordIssue(
            source_row_number=4,
            inspection_id="12",
            rule_id="inspection_year_matches_filename",
            column="inspection_date",
            value="2020-02-03T00:00:00.000",
            message=("Inspection year 2020 does not match batch year 2019."),
        ),
    )


def test_find_inspection_date_issues_supports_multiple_formats() -> None:
    """A date may match any format explicitly accepted by the contract."""
    data = _frame(
        [
            {
                "inspection_id": "1",
                "inspection_date": "2019-05-06",
            },
            {
                "inspection_id": "2",
                "inspection_date": "07/08/2019",
            },
        ]
    )

    issues = find_inspection_date_issues(
        data,
        batch_year=2019,
        accepted_formats=[
            "%Y-%m-%d",
            "%d/%m/%Y",
        ],
    )

    assert issues == ()


def test_record_issue_is_immutable() -> None:
    """A recorded data quality issue must not change after creation."""
    issue = RecordIssue(
        source_row_number=2,
        inspection_id="1",
        rule_id="inspection_date_required",
        column="inspection_date",
        value="",
        message="Inspection date is required.",
    )

    with pytest.raises(FrozenInstanceError):
        issue.source_row_number = 3


@pytest.mark.parametrize(
    "batch_year",
    [
        True,
        0,
        10000,
        2019.0,
    ],
)
def test_find_inspection_date_issues_rejects_invalid_batch_year(
    batch_year: object,
) -> None:
    """The batch year must be a valid four-digit-compatible integer."""
    data = _frame(
        [
            {
                "inspection_id": "1",
                "inspection_date": "2019-01-01",
            }
        ]
    )

    with pytest.raises(
        RecordValidationError,
        match="Batch year must be an integer between 1 and 9999",
    ):
        find_inspection_date_issues(
            data,
            batch_year=batch_year,
            accepted_formats=["%Y-%m-%d"],
        )


def test_find_inspection_date_issues_requires_accepted_format() -> None:
    """At least one date format must be supplied."""
    data = _frame(
        [
            {
                "inspection_id": "1",
                "inspection_date": "2019-01-01",
            }
        ]
    )

    with pytest.raises(
        RecordValidationError,
        match="At least one accepted inspection date format is required",
    ):
        find_inspection_date_issues(
            data,
            batch_year=2019,
            accepted_formats=[],
        )


@pytest.mark.parametrize(
    "accepted_formats",
    [
        ["%Y-%m-%d", "   "],
        ["%Y-%m-%d", 123],
    ],
)
def test_find_inspection_date_issues_rejects_invalid_format_entries(
    accepted_formats: list[object],
) -> None:
    """Accepted date formats must be non-empty strings."""
    data = _frame(
        [
            {
                "inspection_id": "1",
                "inspection_date": "2019-01-01",
            }
        ]
    )

    with pytest.raises(
        RecordValidationError,
        match="Accepted inspection date formats must be non-empty strings",
    ):
        find_inspection_date_issues(
            data,
            batch_year=2019,
            accepted_formats=accepted_formats,
        )


def test_find_inspection_date_issues_rejects_missing_columns() -> None:
    """Both the primary key and date column are required."""
    data = _frame(
        [
            {
                "other_column": "value",
            }
        ]
    )

    with pytest.raises(
        RecordValidationError,
        match="Data is missing required validation columns",
    ) as error:
        find_inspection_date_issues(
            data,
            batch_year=2019,
            accepted_formats=["%Y-%m-%d"],
        )

    message = str(error.value)

    assert "inspection_id" in message
    assert "inspection_date" in message


def test_find_inspection_id_issues_returns_empty_tuple_for_valid_ids() -> None:
    """Valid unique identifiers should produce no issues."""
    data = _frame(
        [
            {"inspection_id": "10"},
            {"inspection_id": "11"},
            {"inspection_id": "9223372036854775807"},
        ]
    )

    issues = find_inspection_id_issues(
        data,
        minimum=10,
    )

    assert issues == ()


def test_find_inspection_id_issues_classifies_invalid_values() -> None:
    """Missing, malformed, overflowing, and low IDs should be distinguished."""
    data = _frame(
        [
            {"inspection_id": ""},
            {"inspection_id": "12A"},
            {"inspection_id": "9223372036854775808"},
            {"inspection_id": "0"},
            {"inspection_id": "15"},
        ]
    )
    data.index = [50, 60, 70, 80, 90]

    issues = find_inspection_id_issues(
        data,
        minimum=1,
    )

    assert issues == (
        RecordIssue(
            source_row_number=2,
            inspection_id="",
            rule_id="inspection_id_required",
            column="inspection_id",
            value="",
            message="Inspection ID is required.",
        ),
        RecordIssue(
            source_row_number=3,
            inspection_id="12A",
            rule_id="inspection_id_format",
            column="inspection_id",
            value="12A",
            message="Inspection ID must contain digits only.",
        ),
        RecordIssue(
            source_row_number=4,
            inspection_id="9223372036854775808",
            rule_id="inspection_id_int64",
            column="inspection_id",
            value="9223372036854775808",
            message="Inspection ID cannot be represented as int64.",
        ),
        RecordIssue(
            source_row_number=5,
            inspection_id="0",
            rule_id="inspection_id_minimum",
            column="inspection_id",
            value="0",
            message="Inspection ID must be greater than or equal to 1.",
        ),
    )


def test_find_inspection_id_issues_flags_all_canonical_duplicates() -> None:
    """Equivalent numeric IDs should all be reported as duplicates."""
    data = _frame(
        [
            {"inspection_id": "001"},
            {"inspection_id": "1"},
            {"inspection_id": "2"},
            {"inspection_id": "002"},
            {"inspection_id": "3"},
        ]
    )

    issues = find_inspection_id_issues(data)

    assert issues == (
        RecordIssue(
            source_row_number=2,
            inspection_id="001",
            rule_id="inspection_id_unique",
            column="inspection_id",
            value="001",
            message="Inspection ID 1 appears more than once in the batch.",
        ),
        RecordIssue(
            source_row_number=3,
            inspection_id="1",
            rule_id="inspection_id_unique",
            column="inspection_id",
            value="1",
            message="Inspection ID 1 appears more than once in the batch.",
        ),
        RecordIssue(
            source_row_number=4,
            inspection_id="2",
            rule_id="inspection_id_unique",
            column="inspection_id",
            value="2",
            message="Inspection ID 2 appears more than once in the batch.",
        ),
        RecordIssue(
            source_row_number=5,
            inspection_id="002",
            rule_id="inspection_id_unique",
            column="inspection_id",
            value="002",
            message="Inspection ID 2 appears more than once in the batch.",
        ),
    )


def test_find_inspection_id_issues_supports_custom_primary_key() -> None:
    """The validator should support the primary key declared by a contract."""
    data = _frame(
        [
            {"custom_id": "10"},
            {"custom_id": "11"},
        ]
    )

    issues = find_inspection_id_issues(
        data,
        primary_key="custom_id",
        minimum=10,
    )

    assert issues == ()


@pytest.mark.parametrize(
    "minimum",
    [
        True,
        0,
        -1,
        1.0,
        "1",
    ],
)
def test_find_inspection_id_issues_rejects_invalid_minimum(
    minimum: object,
) -> None:
    """The configured minimum must be a positive integer."""
    data = _frame([{"inspection_id": "1"}])

    with pytest.raises(
        RecordValidationError,
        match="Inspection ID minimum must be a positive integer",
    ):
        find_inspection_id_issues(
            data,
            minimum=minimum,
        )


def test_find_inspection_id_issues_rejects_missing_primary_key() -> None:
    """The configured primary-key column must exist."""
    data = _frame([{"other_column": "1"}])

    with pytest.raises(
        RecordValidationError,
        match="Data is missing required validation column: inspection_id",
    ):
        find_inspection_id_issues(data)
