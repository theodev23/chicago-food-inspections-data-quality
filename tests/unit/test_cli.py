"""Unit tests for the annual pipeline command-line interface."""

import json
from pathlib import Path

import pytest

import data_quality_pipeline.cli as cli_module
from data_quality_pipeline.batch import IncomingBatch
from data_quality_pipeline.cli import (
    build_json_summary,
    build_multi_batch_json_summary,
    build_parser,
    format_multi_batch_text_summary,
    format_text_summary,
    main,
)
from data_quality_pipeline.curated_writer import (
    CuratedParquetWriteResult,
)
from data_quality_pipeline.multi_batch_runner import (
    MultiBatchPipelineResult,
)
from data_quality_pipeline.pipeline_runner import (
    BatchPipelineRunResult,
    BatchPipelineSkipResult,
)
from data_quality_pipeline.quarantine_writer import (
    QuarantineParquetWriteResult,
)
from data_quality_pipeline.state_manifest import BatchStateManifest
from data_quality_pipeline.validation import RecordIssue
from data_quality_pipeline.validation_runner import BatchValidationResult


def _issue(
    *,
    severity: str,
    rule_id: str,
) -> RecordIssue:
    """Build one validation issue for CLI summaries."""
    return RecordIssue(
        source_row_number=2,
        inspection_id="100",
        rule_id=rule_id,
        column="results",
        value="UNKNOWN",
        message="Record issue.",
        severity=severity,
    )


def _run_result(
    *,
    duplicate_detection_skipped: bool = False,
) -> BatchPipelineRunResult:
    """Build complete successful-run metadata."""
    batch = IncomingBatch(
        path=Path("data/incoming/food_inspections_2019.csv"),
        year=2019,
        size_bytes=1000,
        checksum="abc123",
        checksum_algorithm="sha256",
    )
    validation = BatchValidationResult(
        issues=(
            _issue(
                severity="error",
                rule_id="invalid_result",
            ),
            _issue(
                severity="warning",
                rule_id="missing_risk",
            ),
        ),
        exact_duplicates=(),
        duplicate_detection_skipped=duplicate_detection_skipped,
    )
    curated_write = CuratedParquetWriteResult(
        path=Path("data/curated/inspection_year=2019/food_inspections_2019.parquet"),
        row_count=2,
        size_bytes=500,
        partition_column="inspection_year",
        partition_value=2019,
        compression="snappy",
    )
    quarantine_write = QuarantineParquetWriteResult(
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

    return BatchPipelineRunResult(
        batch=batch,
        validation=validation,
        raw_row_count=3,
        accepted_row_count=2,
        rejected_record_count=1,
        quarantine_issue_count=1,
        curated_write=curated_write,
        quarantine_write=quarantine_write,
    )


def _skip_result() -> BatchPipelineSkipResult:
    """Build complete metadata for an unchanged skipped batch."""
    batch = IncomingBatch(
        path=Path("data/incoming/food_inspections_2019.csv"),
        year=2019,
        size_bytes=1000,
        checksum="abc123",
        checksum_algorithm="sha256",
    )
    manifest = BatchStateManifest(
        schema_version=1,
        dataset="chicago_food_inspections",
        batch_year=2019,
        completed_at_utc="2026-07-30T08:00:00Z",
        source_path=str(batch.path),
        source_size_bytes=1000,
        checksum_algorithm="sha256",
        checksum="abc123",
        raw_row_count=3,
        accepted_row_count=2,
        rejected_record_count=1,
        quarantine_issue_count=1,
        error_count=1,
        warning_count=1,
        curated_path=(
            "data/curated/inspection_year=2019/food_inspections_2019.parquet"
        ),
        curated_row_count=2,
        curated_size_bytes=500,
        curated_compression="snappy",
        quarantine_path=(
            "data/quarantine/dq_batch_year=2019/"
            "food_inspections_2019_quarantine.parquet"
        ),
        quarantine_row_count=1,
        quarantine_size_bytes=250,
        quarantine_compression="snappy",
    )

    return BatchPipelineSkipResult(
        batch=batch,
        manifest=manifest,
        state_manifest_path=Path("data/state/chicago_food_inspections_2019.json"),
    )


def _multi_batch_result() -> MultiBatchPipelineResult:
    """Build aggregate metadata for processed and skipped batches."""
    processed = _run_result()
    skipped = _skip_result()

    return MultiBatchPipelineResult(
        batch_paths=(
            processed.batch.path,
            skipped.batch.path,
        ),
        results=(
            processed,
            skipped,
        ),
    )


def test_build_parser_uses_expected_defaults() -> None:
    """The parser should expose stable default configuration paths."""
    arguments = build_parser().parse_args(["data/incoming/food_inspections_2019.csv"])

    assert arguments.file_path == "data/incoming/food_inspections_2019.csv"
    assert arguments.config == "config/pipeline.yaml"
    assert arguments.contract == "config/data_contract.yaml"
    assert not arguments.all_batches
    assert not arguments.json_output


def test_build_parser_accepts_all_overrides() -> None:
    """Users should be able to replace configuration paths and output mode."""
    arguments = build_parser().parse_args(
        [
            "incoming.csv",
            "--config",
            "custom/pipeline.yaml",
            "--contract",
            "custom/contract.yaml",
            "--json",
        ]
    )

    assert arguments.file_path == "incoming.csv"
    assert arguments.config == "custom/pipeline.yaml"
    assert arguments.contract == "custom/contract.yaml"
    assert not arguments.all_batches
    assert arguments.json_output


def test_build_parser_accepts_all_batches_mode() -> None:
    """The parser should allow discovery mode without a file path."""
    arguments = build_parser().parse_args(
        [
            "--all",
            "--config",
            "custom/pipeline.yaml",
            "--contract",
            "custom/contract.yaml",
            "--json",
        ]
    )

    assert arguments.file_path is None
    assert arguments.all_batches
    assert arguments.config == "custom/pipeline.yaml"
    assert arguments.contract == "custom/contract.yaml"
    assert arguments.json_output


def test_main_rejects_missing_file_path_without_all(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """One file path should be required outside discovery mode."""
    with pytest.raises(SystemExit) as error:
        main([])

    output = capsys.readouterr()

    assert error.value.code == 2
    assert output.out == ""
    assert "file_path is required unless --all is used" in output.err


def test_main_rejects_file_path_with_all(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A specific path and discovery mode must be mutually exclusive."""
    with pytest.raises(SystemExit) as error:
        main(
            [
                "incoming.csv",
                "--all",
            ]
        )

    output = capsys.readouterr()

    assert error.value.code == 2
    assert output.out == ""
    assert "file_path cannot be used together with --all" in output.err


def test_build_json_summary_contains_complete_run_metadata() -> None:
    """The machine-readable summary should expose operational metadata."""
    summary = build_json_summary(_run_result())

    assert summary == {
        "status": "processed",
        "batch": {
            "path": "data/incoming/food_inspections_2019.csv",
            "year": 2019,
            "size_bytes": 1000,
            "checksum_algorithm": "sha256",
            "checksum": "abc123",
        },
        "validation": {
            "errors": 1,
            "warnings": 1,
            "duplicate_detection_skipped": False,
        },
        "records": {
            "raw": 3,
            "accepted": 2,
            "rejected": 1,
            "quarantine_issues": 1,
        },
        "outputs": {
            "curated": {
                "path": (
                    "data/curated/inspection_year=2019/food_inspections_2019.parquet"
                ),
                "rows": 2,
                "size_bytes": 500,
                "compression": "snappy",
            },
            "quarantine": {
                "path": (
                    "data/quarantine/dq_batch_year=2019/"
                    "food_inspections_2019_quarantine.parquet"
                ),
                "rows": 1,
                "size_bytes": 250,
                "compression": "snappy",
            },
        },
    }


def test_build_json_summary_reports_skipped_batch_state() -> None:
    """JSON output should expose persisted metadata for skipped batches."""
    summary = build_json_summary(_skip_result())

    assert summary == {
        "status": "skipped",
        "reason": "batch_state_current",
        "batch": {
            "path": ("data/incoming/food_inspections_2019.csv"),
            "year": 2019,
            "size_bytes": 1000,
            "checksum_algorithm": "sha256",
            "checksum": "abc123",
        },
        "state": {
            "manifest_path": ("data/state/chicago_food_inspections_2019.json"),
            "completed_at_utc": ("2026-07-30T08:00:00Z"),
        },
        "validation": {
            "errors": 1,
            "warnings": 1,
        },
        "records": {
            "raw": 3,
            "accepted": 2,
            "rejected": 1,
            "quarantine_issues": 1,
        },
        "outputs": {
            "curated": {
                "path": (
                    "data/curated/inspection_year=2019/food_inspections_2019.parquet"
                ),
                "rows": 2,
                "size_bytes": 500,
                "compression": "snappy",
            },
            "quarantine": {
                "path": (
                    "data/quarantine/dq_batch_year=2019/"
                    "food_inspections_2019_quarantine.parquet"
                ),
                "rows": 1,
                "size_bytes": 250,
                "compression": "snappy",
            },
        },
    }


def test_format_text_summary_reports_skipped_batch_state() -> None:
    """Text output should explain why an unchanged batch was skipped."""
    summary = format_text_summary(_skip_result())

    assert summary.startswith("Pipeline skipped: batch state is current")
    assert "[State]" in summary
    assert ("Manifest path: data/state/chicago_food_inspections_2019.json") in summary
    assert "Completed at UTC: 2026-07-30T08:00:00Z" in summary
    assert "Raw: 3" in summary
    assert "Accepted: 2" in summary
    assert "Rejected: 1" in summary
    assert "Errors: 1" in summary
    assert "Warnings: 1" in summary
    assert "Compression: snappy" in summary


def test_build_multi_batch_json_summary_contains_all_results() -> None:
    """Multi-batch JSON should aggregate counts and batch details."""
    result = _multi_batch_result()

    summary = build_multi_batch_json_summary(result)

    assert summary == {
        "status": "success",
        "mode": "all",
        "summary": {
            "discovered": 2,
            "processed": 1,
            "skipped": 1,
        },
        "batches": [
            build_json_summary(result.results[0]),
            build_json_summary(result.results[1]),
        ],
    }


def test_format_multi_batch_text_summary_lists_batch_statuses() -> None:
    """Multi-batch text should list each annual execution status."""
    summary = format_multi_batch_text_summary(_multi_batch_result())

    assert summary.startswith("Multi-batch pipeline completed successfully")
    assert "Discovered: 2" in summary
    assert "Processed: 1" in summary
    assert "Skipped: 1" in summary
    assert ("2019: processed - data/incoming/food_inspections_2019.csv") in summary
    assert ("2019: skipped - data/incoming/food_inspections_2019.csv") in summary


def test_format_multi_batch_text_summary_reports_empty_discovery() -> None:
    """An empty incoming directory should have explicit readable output."""
    summary = format_multi_batch_text_summary(
        MultiBatchPipelineResult(
            batch_paths=(),
            results=(),
        )
    )

    assert "Discovered: 0" in summary
    assert "Processed: 0" in summary
    assert "Skipped: 0" in summary
    assert "No incoming batches discovered." in summary


@pytest.mark.parametrize(
    ("duplicate_detection_skipped", "expected_status"),
    [
        (False, "Duplicate detection: completed"),
        (True, "Duplicate detection: skipped"),
    ],
)
def test_format_text_summary_reports_duplicate_detection_status(
    duplicate_detection_skipped: bool,
    expected_status: str,
) -> None:
    """The text summary should clearly report duplicate detection status."""
    summary = format_text_summary(
        _run_result(duplicate_detection_skipped=duplicate_detection_skipped)
    )

    assert summary.startswith("Pipeline completed successfully")
    assert "Raw: 3" in summary
    assert "Accepted: 2" in summary
    assert "Rejected: 1" in summary
    assert "Errors: 1" in summary
    assert "Warnings: 1" in summary
    assert expected_status in summary
    assert "Compression: snappy" in summary


def test_main_runs_pipeline_and_prints_text_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A successful default invocation should print readable output."""
    captured_arguments: dict[str, object] = {}

    def fake_run_batch_pipeline(
        file_path: str,
        *,
        config_path: str,
        contract_path: str,
    ) -> BatchPipelineRunResult:
        captured_arguments.update(
            {
                "file_path": file_path,
                "config_path": config_path,
                "contract_path": contract_path,
            }
        )
        return _run_result()

    monkeypatch.setattr(
        cli_module,
        "run_batch_pipeline",
        fake_run_batch_pipeline,
    )

    exit_code = main(["data/incoming/food_inspections_2019.csv"])
    output = capsys.readouterr()

    assert exit_code == 0
    assert captured_arguments == {
        "file_path": "data/incoming/food_inspections_2019.csv",
        "config_path": "config/pipeline.yaml",
        "contract_path": "config/data_contract.yaml",
    }
    assert "Pipeline completed successfully" in output.out
    assert "[Curated output]" in output.out
    assert "[Quarantine output]" in output.out
    assert output.err == ""


def test_main_prints_valid_json_with_custom_paths(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """JSON mode should be parseable and forward path overrides."""
    captured_arguments: dict[str, object] = {}

    def fake_run_batch_pipeline(
        file_path: str,
        *,
        config_path: str,
        contract_path: str,
    ) -> BatchPipelineRunResult:
        captured_arguments.update(
            {
                "file_path": file_path,
                "config_path": config_path,
                "contract_path": contract_path,
            }
        )
        return _run_result()

    monkeypatch.setattr(
        cli_module,
        "run_batch_pipeline",
        fake_run_batch_pipeline,
    )

    exit_code = main(
        [
            "incoming.csv",
            "--config",
            "custom/pipeline.yaml",
            "--contract",
            "custom/contract.yaml",
            "--json",
        ]
    )
    output = capsys.readouterr()
    summary = json.loads(output.out)

    assert exit_code == 0
    assert captured_arguments == {
        "file_path": "incoming.csv",
        "config_path": "custom/pipeline.yaml",
        "contract_path": "custom/contract.yaml",
    }
    assert summary["status"] == "processed"
    assert summary["batch"]["year"] == 2019
    assert summary["records"]["raw"] == 3
    assert summary["validation"]["errors"] == 1
    assert output.err == ""


def test_main_returns_one_and_reports_pipeline_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Domain failures should produce a non-zero process result."""

    def fail_run_batch_pipeline(
        _file_path: str,
        *,
        config_path: str,
        contract_path: str,
    ) -> BatchPipelineRunResult:
        del config_path, contract_path
        raise RuntimeError("Simulated pipeline failure.")

    monkeypatch.setattr(
        cli_module,
        "run_batch_pipeline",
        fail_run_batch_pipeline,
    )

    exit_code = main(["incoming.csv"])
    output = capsys.readouterr()

    assert exit_code == 1
    assert output.out == ""
    assert output.err == (
        "Pipeline failed (RuntimeError): Simulated pipeline failure.\n"
    )


def test_main_runs_all_batches_and_prints_text_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Discovery mode should print one readable aggregate summary."""
    captured_arguments: dict[str, object] = {}

    def fake_run_discovered_batches(
        *,
        config_path: str,
        contract_path: str,
    ) -> MultiBatchPipelineResult:
        captured_arguments.update(
            {
                "config_path": config_path,
                "contract_path": contract_path,
            }
        )
        return _multi_batch_result()

    def fail_single_batch(
        *_args: object,
        **_kwargs: object,
    ) -> BatchPipelineRunResult:
        raise AssertionError("Single-batch runner must not be called with --all.")

    monkeypatch.setattr(
        cli_module,
        "run_discovered_batches",
        fake_run_discovered_batches,
    )
    monkeypatch.setattr(
        cli_module,
        "run_batch_pipeline",
        fail_single_batch,
    )

    exit_code = main(["--all"])
    output = capsys.readouterr()

    assert exit_code == 0
    assert captured_arguments == {
        "config_path": "config/pipeline.yaml",
        "contract_path": "config/data_contract.yaml",
    }
    assert "Multi-batch pipeline completed successfully" in output.out
    assert "Discovered: 2" in output.out
    assert "Processed: 1" in output.out
    assert "Skipped: 1" in output.out
    assert output.err == ""


def test_main_prints_multi_batch_json_with_custom_paths(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Discovery JSON mode should forward configuration overrides."""
    captured_arguments: dict[str, object] = {}

    def fake_run_discovered_batches(
        *,
        config_path: str,
        contract_path: str,
    ) -> MultiBatchPipelineResult:
        captured_arguments.update(
            {
                "config_path": config_path,
                "contract_path": contract_path,
            }
        )
        return _multi_batch_result()

    monkeypatch.setattr(
        cli_module,
        "run_discovered_batches",
        fake_run_discovered_batches,
    )

    exit_code = main(
        [
            "--all",
            "--config",
            "custom/pipeline.yaml",
            "--contract",
            "custom/contract.yaml",
            "--json",
        ]
    )
    output = capsys.readouterr()
    summary = json.loads(output.out)

    assert exit_code == 0
    assert captured_arguments == {
        "config_path": "custom/pipeline.yaml",
        "contract_path": "custom/contract.yaml",
    }
    assert summary["status"] == "success"
    assert summary["mode"] == "all"
    assert summary["summary"] == {
        "discovered": 2,
        "processed": 1,
        "skipped": 1,
    }
    assert len(summary["batches"]) == 2
    assert output.err == ""
