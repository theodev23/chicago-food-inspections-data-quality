"""Unit tests for annual batch pipeline orchestration."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import data_quality_pipeline.pipeline_runner as pipeline_runner_module
from data_quality_pipeline.batch import IncomingBatch
from data_quality_pipeline.curated_writer import (
    CuratedParquetWriteResult,
)
from data_quality_pipeline.pipeline_runner import (
    BatchPipelineRunError,
    BatchPipelineRunResult,
    _rejected_source_positions,
    _select_accepted_records,
    _validate_run_counts,
    run_batch_pipeline,
)
from data_quality_pipeline.quarantine_writer import (
    QuarantineParquetWriteResult,
)
from data_quality_pipeline.validation import RecordIssue
from data_quality_pipeline.validation_runner import (
    BatchValidationResult,
)


def _batch() -> IncomingBatch:
    """Build incoming-batch metadata for runner tests."""
    return IncomingBatch(
        path=Path("data/incoming/food_inspections_2019.csv"),
        year=2019,
        size_bytes=123,
        checksum="abc123",
        checksum_algorithm="sha256",
    )


def _issue(
    *,
    source_row_number: int,
    inspection_id: str,
    rule_id: str = "invalid_record",
    severity: str = "error",
) -> RecordIssue:
    """Build one record-level validation issue."""
    return RecordIssue(
        source_row_number=source_row_number,
        inspection_id=inspection_id,
        rule_id=rule_id,
        column="results",
        value="UNKNOWN",
        message="Record is invalid.",
        severity=severity,
    )


def _validation_result(
    *issues: RecordIssue,
) -> BatchValidationResult:
    """Build a batch validation result."""
    return BatchValidationResult(
        issues=issues,
        exact_duplicates=(),
        duplicate_detection_skipped=False,
    )


def _curated_write_result() -> CuratedParquetWriteResult:
    """Build curated write metadata."""
    return CuratedParquetWriteResult(
        path=Path("data/curated/inspection_year=2019/food_inspections_2019.parquet"),
        row_count=2,
        size_bytes=500,
        partition_column="inspection_year",
        partition_value=2019,
        compression="snappy",
    )


def _quarantine_write_result() -> QuarantineParquetWriteResult:
    """Build quarantine write metadata."""
    return QuarantineParquetWriteResult(
        path=Path(
            "data/quarantine/dq_batch_year=2019/"
            "food_inspections_2019_quarantine.parquet"
        ),
        row_count=1,
        size_bytes=250,
        partition_column="dq_batch_year",
        partition_value=2019,
        compression="snappy",
    )


def test_batch_pipeline_run_result_is_immutable_and_counts_issues() -> None:
    """Run metadata should expose stable error and warning counts."""
    validation = _validation_result(
        _issue(
            source_row_number=2,
            inspection_id="10",
            severity="warning",
        ),
        _issue(
            source_row_number=3,
            inspection_id="20",
        ),
    )
    result = BatchPipelineRunResult(
        batch=_batch(),
        validation=validation,
        raw_row_count=3,
        accepted_row_count=2,
        rejected_record_count=1,
        quarantine_issue_count=1,
        curated_write=_curated_write_result(),
        quarantine_write=_quarantine_write_result(),
    )

    assert result.error_count == 1
    assert result.warning_count == 1

    with pytest.raises(FrozenInstanceError):
        result.raw_row_count = 4


def test_run_batch_pipeline_orchestrates_complete_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runner should connect all components with correct arguments."""
    config = {
        "source": {
            "start_year": 2019,
            "end_year": 2025,
        },
        "ingestion": {
            "encoding": "utf-8",
            "delimiter": ",",
            "hash_algorithm": "sha256",
        },
        "output": {
            "format": "parquet",
            "compression": "snappy",
            "partition_by": ["inspection_year"],
        },
        "paths": {
            "curated": "data/curated",
            "quarantine": "data/quarantine",
        },
    }
    contract = {
        "contract": {
            "primary_key": "inspection_id",
        },
        "source_schema": {
            "expected_columns": [
                "inspection_id",
                "dba_name",
            ],
        },
    }
    batch = _batch()
    raw = pd.DataFrame(
        {
            "inspection_id": pd.Series(
                ["10", "20", "30"],
                dtype="string",
            ),
            "dba_name": pd.Series(
                ["FIRST", "SECOND", "THIRD"],
                dtype="string",
            ),
        }
    )
    warning = _issue(
        source_row_number=2,
        inspection_id="10",
        rule_id="missing_risk",
        severity="warning",
    )
    error = _issue(
        source_row_number=3,
        inspection_id="20",
    )
    validation = _validation_result(warning, error)
    quarantine = pd.DataFrame(
        {
            "inspection_id": pd.Series(
                ["20"],
                dtype="string",
            ),
            "dq_rule_id": pd.Series(
                ["invalid_record"],
                dtype="string",
            ),
        }
    )
    curated = pd.DataFrame(
        {
            "inspection_id": pd.Series(
                [10, 30],
                dtype="int64",
            ),
            "inspection_year": pd.Series(
                [2019, 2019],
                dtype="int16",
            ),
        }
    )
    call_order: list[str] = []
    captured: dict[str, object] = {}

    def fake_load_config(path: str | Path) -> dict[str, object]:
        call_order.append("load_config")
        captured["config_path"] = path
        return config

    def fake_load_contract(path: str | Path) -> dict[str, object]:
        call_order.append("load_contract")
        captured["contract_path"] = path
        return contract

    def fake_inspect(
        file_path: str | Path,
        *,
        start_year: int,
        end_year: int,
        hash_algorithm: str,
    ) -> IncomingBatch:
        call_order.append("inspect")
        captured["inspection"] = (
            file_path,
            start_year,
            end_year,
            hash_algorithm,
        )
        return batch

    def fake_read(
        incoming_batch: IncomingBatch,
        *,
        expected_columns: list[str],
        encoding: str,
        delimiter: str,
    ) -> pd.DataFrame:
        call_order.append("read")
        captured["read"] = (
            incoming_batch,
            expected_columns,
            encoding,
            delimiter,
        )
        return raw

    def fake_validate(
        data: pd.DataFrame,
        *,
        batch_year: int,
        contract: dict[str, object],
    ) -> BatchValidationResult:
        call_order.append("validate")
        assert data is raw
        captured["validation"] = (batch_year, contract)
        return validation

    def fake_build_quarantine(
        data: pd.DataFrame,
        *,
        validation_result: BatchValidationResult,
        batch_year: int,
        primary_key: str,
    ) -> pd.DataFrame:
        call_order.append("build_quarantine")
        assert data is raw
        captured["quarantine"] = (
            validation_result,
            batch_year,
            primary_key,
        )
        return quarantine

    def fake_transform(
        data: pd.DataFrame,
        *,
        batch_year: int,
        contract: dict[str, object],
    ) -> pd.DataFrame:
        call_order.append("transform")
        captured["accepted"] = data.copy(deep=True)
        captured["transformation"] = (batch_year, contract)
        return curated

    def fake_write_quarantine(
        data: pd.DataFrame,
        **kwargs: object,
    ) -> QuarantineParquetWriteResult:
        call_order.append("write_quarantine")
        assert data is quarantine
        captured["quarantine_write"] = kwargs
        return _quarantine_write_result()

    def fake_write_curated(
        data: pd.DataFrame,
        **kwargs: object,
    ) -> CuratedParquetWriteResult:
        call_order.append("write_curated")
        assert data is curated
        captured["curated_write"] = kwargs
        return _curated_write_result()

    monkeypatch.setattr(
        pipeline_runner_module,
        "load_config",
        fake_load_config,
    )
    monkeypatch.setattr(
        pipeline_runner_module,
        "load_data_contract",
        fake_load_contract,
    )
    monkeypatch.setattr(
        pipeline_runner_module,
        "inspect_incoming_batch",
        fake_inspect,
    )
    monkeypatch.setattr(
        pipeline_runner_module,
        "read_raw_batch",
        fake_read,
    )
    monkeypatch.setattr(
        pipeline_runner_module,
        "validate_batch_records",
        fake_validate,
    )
    monkeypatch.setattr(
        pipeline_runner_module,
        "build_quarantine_records",
        fake_build_quarantine,
    )
    monkeypatch.setattr(
        pipeline_runner_module,
        "transform_curated_records",
        fake_transform,
    )
    monkeypatch.setattr(
        pipeline_runner_module,
        "write_quarantine_parquet",
        fake_write_quarantine,
    )
    monkeypatch.setattr(
        pipeline_runner_module,
        "write_curated_parquet",
        fake_write_curated,
    )

    result = run_batch_pipeline(
        "incoming.csv",
        config_path="custom/pipeline.yaml",
        contract_path="custom/contract.yaml",
    )

    assert call_order == [
        "load_config",
        "load_contract",
        "inspect",
        "read",
        "validate",
        "build_quarantine",
        "transform",
        "write_quarantine",
        "write_curated",
    ]
    assert captured["config_path"] == "custom/pipeline.yaml"
    assert captured["contract_path"] == "custom/contract.yaml"
    assert captured["inspection"] == (
        "incoming.csv",
        2019,
        2025,
        "sha256",
    )
    assert captured["read"] == (
        batch,
        ["inspection_id", "dba_name"],
        "utf-8",
        ",",
    )
    assert captured["validation"] == (2019, contract)
    assert captured["quarantine"] == (
        validation,
        2019,
        "inspection_id",
    )

    accepted = captured["accepted"]
    assert isinstance(accepted, pd.DataFrame)
    assert accepted["inspection_id"].tolist() == ["10", "30"]
    assert accepted.index.tolist() == [0, 1]

    assert captured["quarantine_write"] == {
        "output_dir": "data/quarantine",
        "batch_year": 2019,
        "output_format": "parquet",
        "compression": "snappy",
    }
    assert captured["curated_write"] == {
        "output_dir": "data/curated",
        "batch_year": 2019,
        "output_format": "parquet",
        "compression": "snappy",
        "partition_by": ["inspection_year"],
    }

    assert result.batch is batch
    assert result.validation is validation
    assert result.raw_row_count == 3
    assert result.accepted_row_count == 2
    assert result.rejected_record_count == 1
    assert result.quarantine_issue_count == 1
    assert result.error_count == 1
    assert result.warning_count == 1


def test_rejected_source_positions_deduplicates_multiple_errors() -> None:
    """Several errors on one row should reject one source record."""
    validation = _validation_result(
        _issue(
            source_row_number=3,
            inspection_id="20",
            rule_id="invalid_city",
        ),
        _issue(
            source_row_number=3,
            inspection_id="20",
            rule_id="invalid_state",
        ),
    )

    positions = _rejected_source_positions(
        validation,
        source_row_count=3,
    )

    assert positions == frozenset({1})


@pytest.mark.parametrize(
    "source_row_number",
    [
        1,
        5,
    ],
)
def test_rejected_source_positions_rejects_unavailable_csv_line(
    source_row_number: int,
) -> None:
    """Blocking issues must reference existing source records."""
    validation = _validation_result(
        _issue(
            source_row_number=source_row_number,
            inspection_id="10",
        )
    )

    with pytest.raises(
        BatchPipelineRunError,
        match="Blocking issue references an unavailable CSV line",
    ):
        _rejected_source_positions(
            validation,
            source_row_count=2,
        )


def test_select_accepted_records_uses_positions_and_resets_index() -> None:
    """Accepted-row selection must use source positions, not identifiers."""
    data = pd.DataFrame(
        {
            "inspection_id": ["10", "10", "30"],
            "value": ["first", "rejected", "third"],
        }
    )
    original = data.copy(deep=True)

    accepted = _select_accepted_records(
        data,
        rejected_positions=frozenset({1}),
    )

    assert accepted.to_dict("records") == [
        {
            "inspection_id": "10",
            "value": "first",
        },
        {
            "inspection_id": "30",
            "value": "third",
        },
    ]
    assert accepted.index.tolist() == [0, 1]
    assert_frame_equal(data, original)


def test_validate_run_counts_accepts_consistent_counts() -> None:
    """Consistent record and issue counts should pass."""
    _validate_run_counts(
        raw_row_count=3,
        accepted_row_count=2,
        rejected_record_count=1,
        quarantine_issue_count=2,
        error_count=2,
    )


def test_validate_run_counts_rejects_record_count_mismatch() -> None:
    """Accepted and rejected records must reconcile to raw rows."""
    with pytest.raises(
        BatchPipelineRunError,
        match=(
            "Accepted and rejected record counts do not match the raw batch row count"
        ),
    ):
        _validate_run_counts(
            raw_row_count=3,
            accepted_row_count=1,
            rejected_record_count=1,
            quarantine_issue_count=1,
            error_count=1,
        )


def test_validate_run_counts_rejects_issue_count_mismatch() -> None:
    """Each blocking issue must produce one quarantine diagnostic."""
    with pytest.raises(
        BatchPipelineRunError,
        match=("Quarantine issue count does not match blocking validation issue count"),
    ):
        _validate_run_counts(
            raw_row_count=3,
            accepted_row_count=2,
            rejected_record_count=1,
            quarantine_issue_count=1,
            error_count=2,
        )
