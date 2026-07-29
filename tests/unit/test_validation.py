"""Unit tests for record-level data quality validation."""

from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from data_quality_pipeline.validation import (
    RecordIssue,
    RecordValidationError,
    find_coordinate_issues,
    find_inspection_date_issues,
    find_inspection_id_issues,
    find_inspection_result_issues,
    find_string_pattern_issues,
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


def test_find_inspection_result_issues_returns_empty_tuple_for_valid_values() -> None:
    """Allowed inspection results should produce no issues."""
    data = _frame(
        [
            {
                "inspection_id": "1",
                "results": "Pass",
            },
            {
                "inspection_id": "2",
                "results": "Fail",
            },
            {
                "inspection_id": "3",
                "results": "Pass w/ Conditions",
            },
        ]
    )

    issues = find_inspection_result_issues(
        data,
        allowed_values=[
            "Pass",
            "Fail",
            "Pass w/ Conditions",
        ],
    )

    assert issues == ()


def test_find_inspection_result_issues_classifies_row_failures() -> None:
    """Missing and unknown inspection results should be distinguished."""
    data = _frame(
        [
            {
                "inspection_id": "10",
                "results": "",
            },
            {
                "inspection_id": "11",
                "results": "Pending",
            },
            {
                "inspection_id": "12",
                "results": "Pass",
            },
        ]
    )
    data.index = [50, 60, 70]

    issues = find_inspection_result_issues(
        data,
        allowed_values=["Pass", "Fail"],
    )

    assert issues == (
        RecordIssue(
            source_row_number=2,
            inspection_id="10",
            rule_id="inspection_result_required",
            column="results",
            value="",
            message="Inspection result is required.",
        ),
        RecordIssue(
            source_row_number=3,
            inspection_id="11",
            rule_id="inspection_result_allowed_values",
            column="results",
            value="Pending",
            message="Inspection result is not an allowed value.",
        ),
    )


def test_find_inspection_result_issues_uses_exact_matching() -> None:
    """Case and surrounding whitespace must not be normalized silently."""
    data = _frame(
        [
            {
                "inspection_id": "1",
                "results": "pass",
            },
            {
                "inspection_id": "2",
                "results": "Pass ",
            },
        ]
    )

    issues = find_inspection_result_issues(
        data,
        allowed_values=["Pass"],
    )

    assert [issue.rule_id for issue in issues] == [
        "inspection_result_allowed_values",
        "inspection_result_allowed_values",
    ]
    assert [issue.value for issue in issues] == [
        "pass",
        "Pass ",
    ]


def test_find_inspection_result_issues_supports_custom_columns() -> None:
    """The validator should support contract-defined source column names."""
    data = _frame(
        [
            {
                "custom_id": "1",
                "custom_result": "Accepted",
            }
        ]
    )

    issues = find_inspection_result_issues(
        data,
        allowed_values=["Accepted"],
        primary_key="custom_id",
        result_column="custom_result",
    )

    assert issues == ()


def test_find_inspection_result_issues_rejects_string_allowed_values() -> None:
    """One string must not be interpreted as a sequence of categories."""
    data = _frame(
        [
            {
                "inspection_id": "1",
                "results": "Pass",
            }
        ]
    )

    with pytest.raises(
        RecordValidationError,
        match="Allowed inspection results must be a sequence of strings",
    ):
        find_inspection_result_issues(
            data,
            allowed_values="Pass",
        )


def test_find_inspection_result_issues_requires_allowed_value() -> None:
    """At least one accepted result category must be configured."""
    data = _frame(
        [
            {
                "inspection_id": "1",
                "results": "Pass",
            }
        ]
    )

    with pytest.raises(
        RecordValidationError,
        match="At least one allowed inspection result is required",
    ):
        find_inspection_result_issues(
            data,
            allowed_values=[],
        )


@pytest.mark.parametrize(
    "allowed_values",
    [
        ["Pass", "   "],
        ["Pass", 123],
    ],
)
def test_find_inspection_result_issues_rejects_invalid_allowed_entries(
    allowed_values: list[object],
) -> None:
    """Allowed result categories must be non-empty strings."""
    data = _frame(
        [
            {
                "inspection_id": "1",
                "results": "Pass",
            }
        ]
    )

    with pytest.raises(
        RecordValidationError,
        match="Allowed inspection results must be non-empty strings",
    ):
        find_inspection_result_issues(
            data,
            allowed_values=allowed_values,
        )


def test_find_inspection_result_issues_rejects_duplicate_allowed_values() -> None:
    """The contract must not declare the same result category twice."""
    data = _frame(
        [
            {
                "inspection_id": "1",
                "results": "Pass",
            }
        ]
    )

    with pytest.raises(
        RecordValidationError,
        match="Allowed inspection results contain duplicate values",
    ):
        find_inspection_result_issues(
            data,
            allowed_values=["Pass", "Pass"],
        )


def test_find_inspection_result_issues_rejects_missing_columns() -> None:
    """Both the primary key and result column are required."""
    data = _frame([{"other_column": "value"}])

    with pytest.raises(
        RecordValidationError,
        match="Data is missing required validation columns",
    ) as error:
        find_inspection_result_issues(
            data,
            allowed_values=["Pass"],
        )

    message = str(error.value)

    assert "inspection_id" in message
    assert "results" in message


def test_find_coordinate_issues_accepts_valid_and_missing_pairs() -> None:
    """Complete coordinate pairs and fully blank pairs should be accepted."""
    data = _frame(
        [
            {
                "inspection_id": "1",
                "latitude": "41.881",
                "longitude": "-87.627",
            },
            {
                "inspection_id": "2",
                "latitude": "",
                "longitude": "",
            },
            {
                "inspection_id": "3",
                "latitude": "   ",
                "longitude": "   ",
            },
            {
                "inspection_id": "4",
                "latitude": "-90",
                "longitude": "180",
            },
        ]
    )

    issues = find_coordinate_issues(data)

    assert issues == ()


def test_find_coordinate_issues_classifies_coordinate_failures() -> None:
    """Pair, numeric, and range failures should be distinguished."""
    data = _frame(
        [
            {
                "inspection_id": "10",
                "latitude": "",
                "longitude": "-87.6",
            },
            {
                "inspection_id": "11",
                "latitude": "north",
                "longitude": "-87.6",
            },
            {
                "inspection_id": "12",
                "latitude": "91",
                "longitude": "-87.6",
            },
            {
                "inspection_id": "13",
                "latitude": "41.8",
                "longitude": "west",
            },
            {
                "inspection_id": "14",
                "latitude": "41.8",
                "longitude": "-181",
            },
        ]
    )
    data.index = [50, 60, 70, 80, 90]

    issues = find_coordinate_issues(data)

    assert issues == (
        RecordIssue(
            source_row_number=2,
            inspection_id="10",
            rule_id="coordinate_pair_consistency",
            column="latitude,longitude",
            value="latitude=''; longitude='-87.6'",
            message=("Latitude and longitude must both be present or both be null."),
        ),
        RecordIssue(
            source_row_number=3,
            inspection_id="11",
            rule_id="latitude_numeric",
            column="latitude",
            value="north",
            message="Latitude must be a finite number.",
        ),
        RecordIssue(
            source_row_number=4,
            inspection_id="12",
            rule_id="latitude_range",
            column="latitude",
            value="91",
            message="Latitude must be between -90 and 90.",
        ),
        RecordIssue(
            source_row_number=5,
            inspection_id="13",
            rule_id="longitude_numeric",
            column="longitude",
            value="west",
            message="Longitude must be a finite number.",
        ),
        RecordIssue(
            source_row_number=6,
            inspection_id="14",
            rule_id="longitude_range",
            column="longitude",
            value="-181",
            message="Longitude must be between -180 and 180.",
        ),
    )


def test_find_coordinate_issues_can_report_multiple_issues_for_one_row() -> None:
    """Pair inconsistency and invalid content may coexist on one row."""
    data = _frame(
        [
            {
                "inspection_id": "15",
                "latitude": "",
                "longitude": "west",
            },
            {
                "inspection_id": "16",
                "latitude": "NaN",
                "longitude": "inf",
            },
        ]
    )

    issues = find_coordinate_issues(data)

    assert [issue.rule_id for issue in issues] == [
        "coordinate_pair_consistency",
        "longitude_numeric",
        "latitude_numeric",
        "longitude_numeric",
    ]
    assert [issue.source_row_number for issue in issues] == [
        2,
        2,
        3,
        3,
    ]


def test_find_coordinate_issues_supports_custom_columns_and_bounds() -> None:
    """Coordinate names and ranges should come from the contract."""
    data = _frame(
        [
            {
                "custom_id": "1",
                "custom_latitude": "5",
                "custom_longitude": "15",
            }
        ]
    )

    issues = find_coordinate_issues(
        data,
        primary_key="custom_id",
        latitude_column="custom_latitude",
        longitude_column="custom_longitude",
        latitude_minimum=0,
        latitude_maximum=10,
        longitude_minimum=10,
        longitude_maximum=20,
    )

    assert issues == ()


@pytest.mark.parametrize(
    ("bounds", "message"),
    [
        (
            {"latitude_minimum": "low"},
            "Latitude bounds must be numeric",
        ),
        (
            {"latitude_maximum": True},
            "Latitude bounds must be numeric",
        ),
        (
            {"longitude_minimum": None},
            "Longitude bounds must be numeric",
        ),
        (
            {"longitude_maximum": False},
            "Longitude bounds must be numeric",
        ),
    ],
)
def test_find_coordinate_issues_rejects_non_numeric_bounds(
    bounds: dict[str, object],
    message: str,
) -> None:
    """Configured coordinate bounds must be numeric and not boolean."""
    data = _frame(
        [
            {
                "inspection_id": "1",
                "latitude": "41.8",
                "longitude": "-87.6",
            }
        ]
    )

    with pytest.raises(
        RecordValidationError,
        match=message,
    ):
        find_coordinate_issues(
            data,
            **bounds,
        )


@pytest.mark.parametrize(
    ("bounds", "message"),
    [
        (
            {"latitude_minimum": float("-inf")},
            "Latitude bounds must be finite",
        ),
        (
            {"longitude_maximum": float("nan")},
            "Longitude bounds must be finite",
        ),
    ],
)
def test_find_coordinate_issues_rejects_non_finite_bounds(
    bounds: dict[str, object],
    message: str,
) -> None:
    """Configured coordinate bounds must be finite."""
    data = _frame(
        [
            {
                "inspection_id": "1",
                "latitude": "41.8",
                "longitude": "-87.6",
            }
        ]
    )

    with pytest.raises(
        RecordValidationError,
        match=message,
    ):
        find_coordinate_issues(
            data,
            **bounds,
        )


@pytest.mark.parametrize(
    ("bounds", "message"),
    [
        (
            {
                "latitude_minimum": 90,
                "latitude_maximum": 90,
            },
            "Latitude minimum must be lower than its maximum",
        ),
        (
            {
                "longitude_minimum": 180,
                "longitude_maximum": -180,
            },
            "Longitude minimum must be lower than its maximum",
        ),
    ],
)
def test_find_coordinate_issues_rejects_invalid_bound_order(
    bounds: dict[str, object],
    message: str,
) -> None:
    """Each configured minimum must be lower than its maximum."""
    data = _frame(
        [
            {
                "inspection_id": "1",
                "latitude": "41.8",
                "longitude": "-87.6",
            }
        ]
    )

    with pytest.raises(
        RecordValidationError,
        match=message,
    ):
        find_coordinate_issues(
            data,
            **bounds,
        )


def test_find_coordinate_issues_rejects_missing_columns() -> None:
    """The primary key, latitude, and longitude columns are required."""
    data = _frame([{"other_column": "value"}])

    with pytest.raises(
        RecordValidationError,
        match="Data is missing required validation columns",
    ) as error:
        find_coordinate_issues(data)

    message = str(error.value)

    assert "inspection_id" in message
    assert "latitude" in message
    assert "longitude" in message


def test_find_string_pattern_issues_accepts_valid_and_nullable_values() -> None:
    """Valid strings and permitted blank values should produce no issues."""
    data = _frame(
        [
            {
                "inspection_id": "1",
                "state": "IL",
            },
            {
                "inspection_id": "2",
                "state": "",
            },
            {
                "inspection_id": "3",
                "state": "NY",
            },
        ]
    )

    issues = find_string_pattern_issues(
        data,
        column="state",
        pattern=r"^[A-Z]{2}$",
        nullable=True,
    )

    assert issues == ()


def test_find_string_pattern_issues_classifies_row_failures() -> None:
    """Missing required values and pattern failures should be distinguished."""
    data = _frame(
        [
            {
                "inspection_id": "10",
                "state": "",
            },
            {
                "inspection_id": "11",
                "state": "Illinois",
            },
            {
                "inspection_id": "12",
                "state": "IL",
            },
        ]
    )
    data.index = [50, 60, 70]

    issues = find_string_pattern_issues(
        data,
        column="state",
        pattern=r"^[A-Z]{2}$",
        nullable=False,
    )

    assert issues == (
        RecordIssue(
            source_row_number=2,
            inspection_id="10",
            rule_id="state_required",
            column="state",
            value="",
            message="state is required.",
        ),
        RecordIssue(
            source_row_number=3,
            inspection_id="11",
            rule_id="state_pattern",
            column="state",
            value="Illinois",
            message="state does not match the required pattern.",
        ),
    )


def test_find_string_pattern_issues_requires_full_match() -> None:
    """A substring match must not satisfy the contractual pattern."""
    data = _frame(
        [
            {
                "inspection_id": "1",
                "state": "IL",
            },
            {
                "inspection_id": "2",
                "state": "ILL",
            },
            {
                "inspection_id": "3",
                "state": " IL",
            },
        ]
    )

    issues = find_string_pattern_issues(
        data,
        column="state",
        pattern=r"[A-Z]{2}",
        nullable=False,
    )

    assert [issue.value for issue in issues] == [
        "ILL",
        " IL",
    ]
    assert all(issue.rule_id == "state_pattern" for issue in issues)


def test_find_string_pattern_issues_supports_custom_columns() -> None:
    """Column names should come from the calling contract."""
    data = _frame(
        [
            {
                "custom_id": "1",
                "postal_code": "60601",
            }
        ]
    )

    issues = find_string_pattern_issues(
        data,
        column="postal_code",
        pattern=r"[0-9]{5}",
        nullable=False,
        primary_key="custom_id",
    )

    assert issues == ()


@pytest.mark.parametrize(
    "column",
    [
        "",
        "   ",
        123,
    ],
)
def test_find_string_pattern_issues_rejects_invalid_column(
    column: object,
) -> None:
    """The validated column name must be a non-empty string."""
    data = _frame(
        [
            {
                "inspection_id": "1",
                "state": "IL",
            }
        ]
    )

    with pytest.raises(
        RecordValidationError,
        match="Pattern validation column must be a non-empty string",
    ):
        find_string_pattern_issues(
            data,
            column=column,
            pattern=r"[A-Z]{2}",
            nullable=False,
        )


@pytest.mark.parametrize(
    "primary_key",
    [
        "",
        "   ",
        123,
    ],
)
def test_find_string_pattern_issues_rejects_invalid_primary_key(
    primary_key: object,
) -> None:
    """The primary-key column name must be a non-empty string."""
    data = _frame(
        [
            {
                "inspection_id": "1",
                "state": "IL",
            }
        ]
    )

    with pytest.raises(
        RecordValidationError,
        match="Pattern validation primary key must be a non-empty string",
    ):
        find_string_pattern_issues(
            data,
            column="state",
            pattern=r"[A-Z]{2}",
            nullable=False,
            primary_key=primary_key,
        )


@pytest.mark.parametrize(
    "pattern",
    [
        "",
        None,
        123,
    ],
)
def test_find_string_pattern_issues_rejects_invalid_pattern(
    pattern: object,
) -> None:
    """The configured regular expression must be a non-empty string."""
    data = _frame(
        [
            {
                "inspection_id": "1",
                "state": "IL",
            }
        ]
    )

    with pytest.raises(
        RecordValidationError,
        match="Validation pattern must be a non-empty string",
    ):
        find_string_pattern_issues(
            data,
            column="state",
            pattern=pattern,
            nullable=False,
        )


@pytest.mark.parametrize(
    "nullable",
    [
        1,
        "true",
    ],
)
def test_find_string_pattern_issues_rejects_invalid_nullable_setting(
    nullable: object,
) -> None:
    """The nullable setting must be a real boolean value."""
    data = _frame(
        [
            {
                "inspection_id": "1",
                "state": "IL",
            }
        ]
    )

    with pytest.raises(
        RecordValidationError,
        match="Pattern validation nullable setting must be boolean",
    ):
        find_string_pattern_issues(
            data,
            column="state",
            pattern=r"[A-Z]{2}",
            nullable=nullable,
        )


def test_find_string_pattern_issues_rejects_invalid_regex() -> None:
    """A malformed regular expression should raise a validation error."""
    data = _frame(
        [
            {
                "inspection_id": "1",
                "state": "IL",
            }
        ]
    )

    with pytest.raises(
        RecordValidationError,
        match="Invalid validation pattern for state",
    ):
        find_string_pattern_issues(
            data,
            column="state",
            pattern="[",
            nullable=False,
        )


def test_find_string_pattern_issues_rejects_missing_columns() -> None:
    """The primary key and validated source column are required."""
    data = _frame([{"other_column": "value"}])

    with pytest.raises(
        RecordValidationError,
        match="Data is missing required validation columns",
    ) as error:
        find_string_pattern_issues(
            data,
            column="state",
            pattern=r"[A-Z]{2}",
            nullable=True,
        )

    message = str(error.value)

    assert "inspection_id" in message
    assert "state" in message


def test_record_issue_defaults_to_error_severity() -> None:
    """Existing validators should classify issues as errors by default."""
    issue = RecordIssue(
        source_row_number=2,
        inspection_id="1",
        rule_id="inspection_id_required",
        column="inspection_id",
        value="",
        message="inspection_id is required.",
    )

    assert issue.severity == "error"


def test_record_issue_accepts_warning_severity() -> None:
    """Non-blocking quality findings should support warning severity."""
    issue = RecordIssue(
        source_row_number=2,
        inspection_id="1",
        rule_id="license_zero",
        column="license_",
        value="0",
        message="license_ uses the zero sentinel.",
        severity="warning",
    )

    assert issue.severity == "warning"


@pytest.mark.parametrize(
    "severity",
    [
        "",
        "critical",
        1,
        None,
    ],
)
def test_record_issue_rejects_invalid_severity(
    severity: object,
) -> None:
    """Only error and warning should be accepted as issue severities."""
    with pytest.raises(
        ValueError,
        match="Record issue severity must be 'error' or 'warning'",
    ):
        RecordIssue(
            source_row_number=2,
            inspection_id="1",
            rule_id="example_rule",
            column="example_column",
            value="example_value",
            message="Example issue.",
            severity=severity,
        )
