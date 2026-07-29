"""Unit tests for contract-driven batch record validation."""

from copy import deepcopy
from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

import data_quality_pipeline.validation_runner as validation_runner_module
from data_quality_pipeline.contract import load_data_contract
from data_quality_pipeline.duplicates import (
    DuplicateDetectionError,
    ExactDuplicateRecord,
)
from data_quality_pipeline.validation import RecordIssue
from data_quality_pipeline.validation_runner import (
    BatchValidationError,
    BatchValidationResult,
    validate_batch_records,
)


def _contract() -> dict[str, object]:
    """Load an independent contract instance for one test."""
    return deepcopy(load_data_contract("config/data_contract.yaml"))


def _valid_row(
    inspection_id: str = "1",
    **overrides: str,
) -> dict[str, str]:
    """Build one source record satisfying the current data contract."""
    row = {
        "inspection_id": inspection_id,
        "dba_name": "EXAMPLE RESTAURANT",
        "aka_name": "",
        "license_": "1234567",
        "facility_type": "Restaurant",
        "risk": "Risk 1 (High)",
        "address": "1 MAIN ST",
        "city": "CHICAGO",
        "state": "IL",
        "zip": "60601",
        "inspection_date": "2019-01-01T00:00:00.000",
        "inspection_type": "Canvass",
        "results": "Pass",
        "violations": "",
        "latitude": "41.881832",
        "longitude": "-87.623177",
        "location": "(41.881832, -87.623177)",
    }
    row.update(overrides)

    return row


def _frame(*rows: dict[str, str]) -> pd.DataFrame:
    """Create a source frame while preserving contractual column order."""
    contract = load_data_contract("config/data_contract.yaml")

    return pd.DataFrame(
        rows,
        columns=contract["source_schema"]["expected_columns"],
        dtype="string",
    )


def test_batch_validation_result_exposes_severity_views() -> None:
    """The result should separate errors and warnings deterministically."""
    error = RecordIssue(
        source_row_number=2,
        inspection_id="1",
        rule_id="example_error",
        column="state",
        value="Illinois",
        message="Example error.",
    )
    warning = RecordIssue(
        source_row_number=3,
        inspection_id="2",
        rule_id="example_warning",
        column="risk",
        value="",
        message="Example warning.",
        severity="warning",
    )

    result = BatchValidationResult(
        issues=(error, warning),
        exact_duplicates=(),
        duplicate_detection_skipped=False,
    )

    assert result.errors == (error,)
    assert result.warnings == (warning,)
    assert result.has_errors is True


def test_batch_validation_result_is_immutable() -> None:
    """Validation results must not change after construction."""
    result = BatchValidationResult(
        issues=(),
        exact_duplicates=(),
        duplicate_detection_skipped=False,
    )

    with pytest.raises(FrozenInstanceError):
        result.issues = ()


def test_validate_batch_records_accepts_valid_record() -> None:
    """A fully valid source record should produce no findings."""
    result = validate_batch_records(
        _frame(_valid_row()),
        batch_year=2019,
        contract=_contract(),
    )

    assert result.issues == ()
    assert result.errors == ()
    assert result.warnings == ()
    assert result.exact_duplicates == ()
    assert result.duplicate_detection_skipped is False
    assert result.has_errors is False


def test_validate_batch_records_collects_business_warnings() -> None:
    """Configured non-blocking business anomalies should remain warnings."""
    data = _frame(
        _valid_row(
            license_="0",
            risk="",
        )
    )

    result = validate_batch_records(
        data,
        batch_year=2019,
        contract=_contract(),
    )

    assert [issue.rule_id for issue in result.issues] == [
        "license_zero_sentinel",
        "risk_missing",
    ]
    assert result.errors == ()
    assert len(result.warnings) == 2
    assert all(issue.severity == "warning" for issue in result.warnings)
    assert result.has_errors is False


def test_validate_batch_records_reports_exact_duplicate_reference() -> None:
    """Exact duplicates should identify the lowest retained inspection ID."""
    data = _frame(
        _valid_row(inspection_id="2"),
        _valid_row(inspection_id="1"),
    )

    result = validate_batch_records(
        data,
        batch_year=2019,
        contract=_contract(),
    )

    assert result.exact_duplicates == (
        ExactDuplicateRecord(
            inspection_id=2,
            duplicate_of_inspection_id=1,
        ),
    )
    assert result.issues == (
        RecordIssue(
            source_row_number=2,
            inspection_id="2",
            rule_id="exact_duplicate_record",
            column="inspection_id",
            value="2",
            message=("Record is an exact duplicate of inspection_id 1."),
            severity="error",
        ),
    )
    assert result.has_errors is True
    assert result.duplicate_detection_skipped is False


def test_validate_batch_records_skips_duplicates_for_invalid_primary_key() -> None:
    """Duplicate detection should not run with an invalid primary key."""
    data = _frame(
        _valid_row(inspection_id="invalid"),
        _valid_row(inspection_id="2"),
    )

    result = validate_batch_records(
        data,
        batch_year=2019,
        contract=_contract(),
    )

    assert result.duplicate_detection_skipped is True
    assert result.exact_duplicates == ()
    assert result.has_errors is True
    assert all(issue.rule_id != "exact_duplicate_record" for issue in result.issues)


def test_validate_batch_records_orders_issues_by_source_row() -> None:
    """Findings from separate validators should follow source row order."""
    data = _frame(
        _valid_row(
            inspection_id="1",
            license_="0",
        ),
        _valid_row(
            inspection_id="2",
            state="Illinois",
        ),
    )

    result = validate_batch_records(
        data,
        batch_year=2019,
        contract=_contract(),
    )

    assert [(issue.source_row_number, issue.rule_id) for issue in result.issues] == [
        (2, "license_zero_sentinel"),
        (3, "state_pattern"),
    ]


def test_validate_batch_records_wraps_unexpected_duplicate_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A duplicate failure after valid IDs should become a runner error."""

    def fail_duplicate_detection(
        *args: object,
        **kwargs: object,
    ) -> tuple[ExactDuplicateRecord, ...]:
        raise DuplicateDetectionError("Unexpected duplicate failure.")

    monkeypatch.setattr(
        validation_runner_module,
        "find_exact_duplicates",
        fail_duplicate_detection,
    )

    with pytest.raises(
        BatchValidationError,
        match=(
            "Exact duplicate detection failed after primary-key validation succeeded"
        ),
    ):
        validate_batch_records(
            _frame(_valid_row()),
            batch_year=2019,
            contract=_contract(),
        )


def test_validate_batch_records_requires_duplicate_rule() -> None:
    """The duplicate issue must use a rule declared by the contract."""
    contract = _contract()
    contract["record_rules"] = [
        rule
        for rule in contract["record_rules"]
        if rule["id"] != "exact_duplicate_record"
    ]

    data = _frame(
        _valid_row(inspection_id="2"),
        _valid_row(inspection_id="1"),
    )

    with pytest.raises(
        BatchValidationError,
        match=("Data contract is missing record rule: exact_duplicate_record"),
    ):
        validate_batch_records(
            data,
            batch_year=2019,
            contract=contract,
        )
