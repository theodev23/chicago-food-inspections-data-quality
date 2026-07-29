"""Unit tests for building issue-level quarantine records."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from data_quality_pipeline.duplicates import ExactDuplicateRecord
from data_quality_pipeline.quarantine import (
    QUARANTINE_METADATA_COLUMNS,
    QuarantineBuildError,
    build_quarantine_records,
)
from data_quality_pipeline.validation import RecordIssue
from data_quality_pipeline.validation_runner import BatchValidationResult


def _source_frame() -> pd.DataFrame:
    """Build a minimal raw source frame."""
    return pd.DataFrame(
        {
            "inspection_id": pd.Series(
                ["10", "20"],
                dtype="string",
            ),
            "dba_name": pd.Series(
                ["FIRST RESTAURANT", "SECOND RESTAURANT"],
                dtype="string",
            ),
        }
    )


def _issue(
    *,
    source_row_number: int = 2,
    inspection_id: str = "10",
    rule_id: str = "invalid_value",
    column: str = "dba_name",
    value: str = "invalid",
    message: str = "Value is invalid.",
    severity: str = "error",
) -> RecordIssue:
    """Build one validation issue."""
    return RecordIssue(
        source_row_number=source_row_number,
        inspection_id=inspection_id,
        rule_id=rule_id,
        column=column,
        value=value,
        message=message,
        severity=severity,
    )


def _validation_result(
    *,
    issues: tuple[RecordIssue, ...] = (),
    exact_duplicates: tuple[ExactDuplicateRecord, ...] = (),
) -> BatchValidationResult:
    """Build a validation result for quarantine tests."""
    return BatchValidationResult(
        issues=issues,
        exact_duplicates=exact_duplicates,
        duplicate_detection_skipped=False,
    )


def _build(
    data: pd.DataFrame,
    *,
    validation_result: BatchValidationResult,
    batch_year: int = 2019,
    primary_key: object = "inspection_id",
) -> pd.DataFrame:
    """Build quarantine output with the standard test configuration."""
    return build_quarantine_records(
        data,
        validation_result=validation_result,
        batch_year=batch_year,
        primary_key=primary_key,
    )


def test_build_quarantine_records_preserves_source_and_adds_metadata() -> None:
    """Each blocking issue should retain its raw source record."""
    data = _source_frame()
    issue = _issue(
        source_row_number=3,
        inspection_id="20",
        rule_id="invalid_result",
        column="results",
        value="UNKNOWN",
        message="Inspection result is invalid.",
    )

    quarantine = _build(
        data,
        validation_result=_validation_result(
            issues=(issue,),
        ),
    )

    assert quarantine.columns.tolist() == [
        *data.columns.tolist(),
        *QUARANTINE_METADATA_COLUMNS,
    ]
    assert quarantine.shape == (1, 10)
    assert quarantine.loc[0, "inspection_id"] == "20"
    assert quarantine.loc[0, "dba_name"] == "SECOND RESTAURANT"
    assert quarantine.loc[0, "dq_source_row_number"] == 3
    assert quarantine.loc[0, "dq_rule_id"] == "invalid_result"
    assert quarantine.loc[0, "dq_column"] == "results"
    assert quarantine.loc[0, "dq_value"] == "UNKNOWN"
    assert quarantine.loc[0, "dq_message"] == "Inspection result is invalid."
    assert quarantine.loc[0, "dq_severity"] == "error"
    assert pd.isna(quarantine.loc[0, "dq_duplicate_of_inspection_id"])
    assert quarantine.loc[0, "dq_batch_year"] == 2019


def test_build_quarantine_records_excludes_warnings() -> None:
    """Warnings must not cause source records to be quarantined."""
    warning = _issue(
        source_row_number=2,
        inspection_id="10",
        rule_id="missing_risk",
        column="risk",
        value="",
        severity="warning",
    )
    error = _issue(
        source_row_number=3,
        inspection_id="20",
        rule_id="invalid_result",
        column="results",
    )

    quarantine = _build(
        _source_frame(),
        validation_result=_validation_result(
            issues=(warning, error),
        ),
    )

    assert quarantine["inspection_id"].tolist() == ["20"]
    assert quarantine["dq_severity"].tolist() == ["error"]


def test_build_quarantine_records_repeats_rows_with_multiple_errors() -> None:
    """One source row should appear once for each blocking issue."""
    first_issue = _issue(
        rule_id="invalid_city",
        column="city",
    )
    second_issue = _issue(
        rule_id="invalid_state",
        column="state",
    )

    quarantine = _build(
        _source_frame(),
        validation_result=_validation_result(
            issues=(first_issue, second_issue),
        ),
    )

    assert quarantine["inspection_id"].tolist() == ["10", "10"]
    assert quarantine["dba_name"].tolist() == [
        "FIRST RESTAURANT",
        "FIRST RESTAURANT",
    ]
    assert quarantine["dq_rule_id"].tolist() == [
        "invalid_city",
        "invalid_state",
    ]


def test_build_quarantine_records_orders_issues_deterministically() -> None:
    """Output ordering should not depend on validation insertion order."""
    issues = (
        _issue(
            source_row_number=3,
            inspection_id="20",
            rule_id="z_rule",
            column="zip",
        ),
        _issue(
            source_row_number=2,
            inspection_id="10",
            rule_id="b_rule",
            column="state",
        ),
        _issue(
            source_row_number=2,
            inspection_id="10",
            rule_id="a_rule",
            column="zip",
        ),
    )

    quarantine = _build(
        _source_frame(),
        validation_result=_validation_result(issues=issues),
    )

    assert quarantine["inspection_id"].tolist() == [
        "10",
        "10",
        "20",
    ]
    assert quarantine["dq_rule_id"].tolist() == [
        "a_rule",
        "b_rule",
        "z_rule",
    ]


def test_build_quarantine_records_does_not_modify_source_data() -> None:
    """Building diagnostic records must not mutate the raw batch."""
    data = _source_frame()
    original = data.copy(deep=True)

    _build(
        data,
        validation_result=_validation_result(
            issues=(_issue(),),
        ),
    )

    assert_frame_equal(data, original)


def test_build_quarantine_records_supports_no_errors() -> None:
    """A clean batch should produce an empty, schema-stable frame."""
    data = _source_frame()

    quarantine = _build(
        data,
        validation_result=_validation_result(),
    )

    assert quarantine.empty
    assert quarantine.columns.tolist() == [
        *data.columns.tolist(),
        *QUARANTINE_METADATA_COLUMNS,
    ]
    assert str(quarantine["dq_source_row_number"].dtype) == "int64"
    assert str(quarantine["dq_duplicate_of_inspection_id"].dtype) == "Int64"
    assert str(quarantine["dq_batch_year"].dtype) == "int16"


@pytest.mark.parametrize(
    "batch_year",
    [
        True,
        0,
        10000,
        2019.0,
    ],
)
def test_build_quarantine_records_rejects_invalid_batch_year(
    batch_year: object,
) -> None:
    """The quarantine batch year must be a valid integer year."""
    with pytest.raises(
        QuarantineBuildError,
        match="Batch year must be an integer between 1 and 9999",
    ):
        _build(
            _source_frame(),
            validation_result=_validation_result(),
            batch_year=batch_year,
        )


def test_build_quarantine_records_rejects_duplicate_source_columns() -> None:
    """Ambiguous source schemas should fail before row mapping."""
    data = _source_frame()
    data.columns = [
        "inspection_id",
        "inspection_id",
    ]

    with pytest.raises(
        QuarantineBuildError,
        match="Source data contains duplicate column names",
    ):
        _build(
            data,
            validation_result=_validation_result(),
        )


@pytest.mark.parametrize(
    "primary_key",
    [
        "",
        None,
        123,
    ],
)
def test_build_quarantine_records_rejects_invalid_primary_key(
    primary_key: object,
) -> None:
    """The primary-key configuration must be a non-empty string."""
    with pytest.raises(
        QuarantineBuildError,
        match="Primary key must be a non-empty string",
    ):
        _build(
            _source_frame(),
            validation_result=_validation_result(),
            primary_key=primary_key,
        )


def test_build_quarantine_records_requires_primary_key_column() -> None:
    """The source frame must contain its configured primary key."""
    data = _source_frame().drop(columns=["inspection_id"])

    with pytest.raises(
        QuarantineBuildError,
        match="Source data is missing primary key: inspection_id",
    ):
        _build(
            data,
            validation_result=_validation_result(),
        )


def test_build_quarantine_records_rejects_metadata_column_collision() -> None:
    """Source columns must not overwrite diagnostic metadata."""
    data = _source_frame()
    data["dq_rule_id"] = "source value"

    with pytest.raises(
        QuarantineBuildError,
        match="Source columns conflict with quarantine metadata",
    ):
        _build(
            data,
            validation_result=_validation_result(),
        )


@pytest.mark.parametrize(
    "source_row_number",
    [
        1,
        4,
    ],
)
def test_build_quarantine_records_rejects_unavailable_csv_line(
    source_row_number: int,
) -> None:
    """Issue line numbers must resolve to an existing source row."""
    issue = _issue(source_row_number=source_row_number)

    with pytest.raises(
        QuarantineBuildError,
        match=("Validation issue references an unavailable CSV line"),
    ):
        _build(
            _source_frame(),
            validation_result=_validation_result(
                issues=(issue,),
            ),
        )


def test_build_quarantine_records_rejects_identifier_mismatch() -> None:
    """Issue identifiers must match their referenced source rows."""
    issue = _issue(
        source_row_number=2,
        inspection_id="999",
    )

    with pytest.raises(
        QuarantineBuildError,
        match="Validation issue does not match its source row",
    ):
        _build(
            _source_frame(),
            validation_result=_validation_result(
                issues=(issue,),
            ),
        )


def test_build_quarantine_records_requires_duplicate_reference() -> None:
    """Exact duplicate issues must identify the retained record."""
    issue = _issue(
        source_row_number=3,
        inspection_id="20",
        rule_id="exact_duplicate_record",
        column="inspection_id",
    )

    with pytest.raises(
        QuarantineBuildError,
        match=("Exact duplicate issue is missing its retained-record reference"),
    ):
        _build(
            _source_frame(),
            validation_result=_validation_result(
                issues=(issue,),
            ),
        )


def test_build_quarantine_records_adds_duplicate_reference() -> None:
    """Duplicate diagnostics should include the retained identifier."""
    issue = _issue(
        source_row_number=3,
        inspection_id="20",
        rule_id="exact_duplicate_record",
        column="inspection_id",
    )
    duplicate = ExactDuplicateRecord(
        inspection_id=20,
        duplicate_of_inspection_id=10,
    )

    quarantine = _build(
        _source_frame(),
        validation_result=_validation_result(
            issues=(issue,),
            exact_duplicates=(duplicate,),
        ),
    )

    assert (
        quarantine.loc[
            0,
            "dq_duplicate_of_inspection_id",
        ]
        == 10
    )
    assert str(quarantine["dq_duplicate_of_inspection_id"].dtype) == "Int64"


def test_build_quarantine_records_rejects_conflicting_references() -> None:
    """One rejected duplicate cannot reference two retained records."""
    duplicates = (
        ExactDuplicateRecord(
            inspection_id=20,
            duplicate_of_inspection_id=10,
        ),
        ExactDuplicateRecord(
            inspection_id=20,
            duplicate_of_inspection_id=11,
        ),
    )

    with pytest.raises(
        QuarantineBuildError,
        match=("Conflicting duplicate references for inspection_id '20'"),
    ):
        _build(
            _source_frame(),
            validation_result=_validation_result(
                exact_duplicates=duplicates,
            ),
        )
