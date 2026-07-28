"""Unit tests for exact duplicate detection."""

from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from data_quality_pipeline.duplicates import (
    DuplicateDetectionError,
    ExactDuplicateRecord,
    find_exact_duplicates,
)

_SOURCE_COLUMNS = [
    "inspection_id",
    "dba_name",
    "results",
]


def _frame(rows: list[dict[str, str]]) -> pd.DataFrame:
    """Create a source-like DataFrame using pandas string columns."""
    return pd.DataFrame(rows, dtype="string")


def test_find_exact_duplicates_keeps_lowest_primary_key() -> None:
    """Every duplicate group should retain its lowest numeric key."""
    data = _frame(
        [
            {
                "inspection_id": "30",
                "dba_name": "Restaurant A",
                "results": "Pass",
            },
            {
                "inspection_id": "10",
                "dba_name": "Restaurant A",
                "results": "Pass",
            },
            {
                "inspection_id": "20",
                "dba_name": "Restaurant A",
                "results": "Pass",
            },
            {
                "inspection_id": "9",
                "dba_name": "Restaurant B",
                "results": "Fail",
            },
            {
                "inspection_id": "7",
                "dba_name": "Restaurant B",
                "results": "Fail",
            },
            {
                "inspection_id": "100",
                "dba_name": "Restaurant C",
                "results": "Pass",
            },
        ]
    )

    duplicates = find_exact_duplicates(
        data,
        primary_key="inspection_id",
        source_columns=_SOURCE_COLUMNS,
    )

    assert duplicates == (
        ExactDuplicateRecord(
            inspection_id=9,
            duplicate_of_inspection_id=7,
        ),
        ExactDuplicateRecord(
            inspection_id=20,
            duplicate_of_inspection_id=10,
        ),
        ExactDuplicateRecord(
            inspection_id=30,
            duplicate_of_inspection_id=10,
        ),
    )


def test_find_exact_duplicates_returns_empty_tuple_without_matches() -> None:
    """Records differing on a source field must not be marked as duplicates."""
    data = _frame(
        [
            {
                "inspection_id": "1",
                "dba_name": "Restaurant A",
                "results": "Pass",
            },
            {
                "inspection_id": "2",
                "dba_name": "Restaurant A",
                "results": "Fail",
            },
            {
                "inspection_id": "3",
                "dba_name": "Restaurant B",
                "results": "Pass",
            },
        ]
    )

    duplicates = find_exact_duplicates(
        data,
        primary_key="inspection_id",
        source_columns=_SOURCE_COLUMNS,
    )

    assert duplicates == ()


def test_exact_duplicate_record_is_immutable() -> None:
    """A duplicate mapping must not change after detection."""
    duplicate = ExactDuplicateRecord(
        inspection_id=2,
        duplicate_of_inspection_id=1,
    )

    with pytest.raises(FrozenInstanceError):
        duplicate.inspection_id = 3


def test_find_exact_duplicates_rejects_empty_source_columns() -> None:
    """At least one source column must be provided."""
    data = _frame([{"inspection_id": "1"}])

    with pytest.raises(
        DuplicateDetectionError,
        match="Source columns must not be empty",
    ):
        find_exact_duplicates(
            data,
            primary_key="inspection_id",
            source_columns=[],
        )


def test_find_exact_duplicates_rejects_duplicate_source_names() -> None:
    """A source column must not appear more than once."""
    data = _frame(
        [
            {
                "inspection_id": "1",
                "dba_name": "Restaurant A",
            }
        ]
    )

    with pytest.raises(
        DuplicateDetectionError,
        match="Source columns contain duplicate names",
    ):
        find_exact_duplicates(
            data,
            primary_key="inspection_id",
            source_columns=[
                "inspection_id",
                "dba_name",
                "dba_name",
            ],
        )


def test_find_exact_duplicates_requires_primary_key_in_schema() -> None:
    """The primary key must be part of the declared source schema."""
    data = _frame(
        [
            {
                "inspection_id": "1",
                "dba_name": "Restaurant A",
            }
        ]
    )

    with pytest.raises(
        DuplicateDetectionError,
        match="Primary key is not present in source columns",
    ):
        find_exact_duplicates(
            data,
            primary_key="inspection_id",
            source_columns=["dba_name"],
        )


def test_find_exact_duplicates_rejects_missing_data_columns() -> None:
    """The DataFrame must contain every declared source column."""
    data = _frame(
        [
            {
                "inspection_id": "1",
                "dba_name": "Restaurant A",
            }
        ]
    )

    with pytest.raises(
        DuplicateDetectionError,
        match="Data is missing required source columns",
    ):
        find_exact_duplicates(
            data,
            primary_key="inspection_id",
            source_columns=_SOURCE_COLUMNS,
        )


def test_find_exact_duplicates_rejects_non_numeric_primary_key() -> None:
    """Primary-key values must contain digits only."""
    data = _frame(
        [
            {
                "inspection_id": "invalid",
                "dba_name": "Restaurant A",
                "results": "Pass",
            }
        ]
    )

    with pytest.raises(
        DuplicateDetectionError,
        match="Primary key contains non-numeric values",
    ):
        find_exact_duplicates(
            data,
            primary_key="inspection_id",
            source_columns=_SOURCE_COLUMNS,
        )


def test_find_exact_duplicates_rejects_duplicate_primary_keys() -> None:
    """Duplicate primary keys prevent deterministic record identity."""
    data = _frame(
        [
            {
                "inspection_id": "1",
                "dba_name": "Restaurant A",
                "results": "Pass",
            },
            {
                "inspection_id": "1",
                "dba_name": "Restaurant B",
                "results": "Fail",
            },
        ]
    )

    with pytest.raises(
        DuplicateDetectionError,
        match="Primary key contains duplicate values",
    ):
        find_exact_duplicates(
            data,
            primary_key="inspection_id",
            source_columns=_SOURCE_COLUMNS,
        )


def test_find_exact_duplicates_requires_comparison_column() -> None:
    """Duplicate detection requires a field other than the primary key."""
    data = _frame([{"inspection_id": "1"}])

    with pytest.raises(
        DuplicateDetectionError,
        match="At least one non-primary-key source column is required",
    ):
        find_exact_duplicates(
            data,
            primary_key="inspection_id",
            source_columns=["inspection_id"],
        )
