"""Unit tests for deterministic multi-batch orchestration."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

import data_quality_pipeline.multi_batch_runner as runner_module
from data_quality_pipeline.batch import IncomingBatch
from data_quality_pipeline.curated_writer import (
    CuratedParquetWriteResult,
)
from data_quality_pipeline.multi_batch_runner import (
    MultiBatchPipelineResult,
    MultiBatchPipelineRunError,
    run_discovered_batches,
)
from data_quality_pipeline.pipeline_runner import (
    BatchPipelineRunResult,
    BatchPipelineSkipResult,
)
from data_quality_pipeline.quarantine_writer import (
    QuarantineParquetWriteResult,
)
from data_quality_pipeline.state_manifest import (
    BatchStateManifest,
)
from data_quality_pipeline.validation_runner import (
    BatchValidationResult,
)


def _batch(path: Path, *, year: int) -> IncomingBatch:
    """Build deterministic incoming-batch metadata."""
    return IncomingBatch(
        path=path,
        year=year,
        size_bytes=1000,
        checksum=f"checksum-{year}",
        checksum_algorithm="sha256",
    )


def _processed_result(
    path: Path,
    *,
    year: int,
) -> BatchPipelineRunResult:
    """Build one processed annual-batch result."""
    batch = _batch(path, year=year)
    validation = BatchValidationResult(
        issues=(),
        exact_duplicates=(),
        duplicate_detection_skipped=False,
    )
    curated_write = CuratedParquetWriteResult(
        path=Path(
            f"data/curated/inspection_year={year}/food_inspections_{year}.parquet"
        ),
        row_count=2,
        size_bytes=500,
        partition_column="inspection_year",
        partition_value=year,
        compression="snappy",
    )
    quarantine_write = QuarantineParquetWriteResult(
        path=Path(
            f"data/quarantine/dq_batch_year={year}/"
            f"food_inspections_{year}_quarantine.parquet"
        ),
        row_count=0,
        size_bytes=250,
        partition_column="dq_batch_year",
        partition_value=year,
        compression="snappy",
    )

    return BatchPipelineRunResult(
        batch=batch,
        validation=validation,
        raw_row_count=2,
        accepted_row_count=2,
        rejected_record_count=0,
        quarantine_issue_count=0,
        curated_write=curated_write,
        quarantine_write=quarantine_write,
    )


def _skipped_result(
    path: Path,
    *,
    year: int,
) -> BatchPipelineSkipResult:
    """Build one unchanged annual-batch result."""
    batch = _batch(path, year=year)
    manifest = BatchStateManifest(
        schema_version=1,
        dataset="chicago_food_inspections",
        batch_year=year,
        completed_at_utc="2026-07-30T08:00:00Z",
        source_path=str(path),
        source_size_bytes=1000,
        checksum_algorithm="sha256",
        checksum=f"checksum-{year}",
        raw_row_count=2,
        accepted_row_count=2,
        rejected_record_count=0,
        quarantine_issue_count=0,
        error_count=0,
        warning_count=0,
        curated_path=(
            f"data/curated/inspection_year={year}/food_inspections_{year}.parquet"
        ),
        curated_row_count=2,
        curated_size_bytes=500,
        curated_compression="snappy",
        quarantine_path=(
            f"data/quarantine/dq_batch_year={year}/"
            f"food_inspections_{year}_quarantine.parquet"
        ),
        quarantine_row_count=0,
        quarantine_size_bytes=250,
        quarantine_compression="snappy",
    )

    return BatchPipelineSkipResult(
        batch=batch,
        manifest=manifest,
        state_manifest_path=Path(f"data/state/chicago_food_inspections_{year}.json"),
    )


def _config() -> dict[str, object]:
    """Build minimal configuration for multi-batch tests."""
    return {
        "source": {
            "start_year": 2019,
            "end_year": 2025,
        },
        "ingestion": {
            "file_pattern": "food_inspections_*.csv",
        },
        "paths": {
            "incoming": "data/incoming",
        },
    }


def test_multi_batch_pipeline_result_is_immutable_and_counts_results() -> None:
    """Aggregate metadata should expose stable status counts."""
    first_path = Path("data/incoming/food_inspections_2019.csv")
    second_path = Path("data/incoming/food_inspections_2020.csv")
    result = MultiBatchPipelineResult(
        batch_paths=(first_path, second_path),
        results=(
            _processed_result(first_path, year=2019),
            _skipped_result(second_path, year=2020),
        ),
    )

    assert result.discovered_count == 2
    assert result.processed_count == 1
    assert result.skipped_count == 1

    with pytest.raises(FrozenInstanceError):
        result.batch_paths = ()


def test_run_discovered_batches_orchestrates_in_discovery_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every discovered batch should run sequentially in year order."""
    first_path = Path("data/incoming/food_inspections_2019.csv")
    second_path = Path("data/incoming/food_inspections_2020.csv")
    discovered_paths = (
        first_path,
        second_path,
    )
    processed = _processed_result(
        first_path,
        year=2019,
    )
    skipped = _skipped_result(
        second_path,
        year=2020,
    )
    call_order: list[str] = []
    captured: dict[str, object] = {}

    def fake_load_config(
        path: str | Path,
    ) -> dict[str, object]:
        call_order.append("load_config")
        captured["config_path"] = path
        return _config()

    def fake_discover(
        input_dir: str | Path,
        *,
        file_pattern: str,
        start_year: int,
        end_year: int,
    ) -> tuple[Path, ...]:
        call_order.append("discover")
        captured["discovery"] = (
            input_dir,
            file_pattern,
            start_year,
            end_year,
        )
        return discovered_paths

    def fake_run_batch(
        file_path: str | Path,
        *,
        config_path: str | Path,
        contract_path: str | Path,
    ) -> BatchPipelineRunResult | BatchPipelineSkipResult:
        path = Path(file_path)
        call_order.append(f"run_{path.stem}")
        captured.setdefault("runs", []).append(
            (
                path,
                config_path,
                contract_path,
            )
        )

        if path == first_path:
            return processed

        return skipped

    monkeypatch.setattr(
        runner_module,
        "load_config",
        fake_load_config,
    )
    monkeypatch.setattr(
        runner_module,
        "discover_incoming_batch_paths",
        fake_discover,
    )
    monkeypatch.setattr(
        runner_module,
        "run_batch_pipeline",
        fake_run_batch,
    )

    result = run_discovered_batches(
        config_path="custom/pipeline.yaml",
        contract_path="custom/contract.yaml",
    )

    assert call_order == [
        "load_config",
        "discover",
        "run_food_inspections_2019",
        "run_food_inspections_2020",
    ]
    assert captured["config_path"] == ("custom/pipeline.yaml")
    assert captured["discovery"] == (
        "data/incoming",
        "food_inspections_*.csv",
        2019,
        2025,
    )
    assert captured["runs"] == [
        (
            first_path,
            "custom/pipeline.yaml",
            "custom/contract.yaml",
        ),
        (
            second_path,
            "custom/pipeline.yaml",
            "custom/contract.yaml",
        ),
    ]
    assert result.batch_paths == discovered_paths
    assert result.results == (
        processed,
        skipped,
    )
    assert result.discovered_count == 2
    assert result.processed_count == 1
    assert result.skipped_count == 1


def test_run_discovered_batches_accepts_empty_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No discovered batches should produce an empty valid result."""
    monkeypatch.setattr(
        runner_module,
        "load_config",
        lambda _path: _config(),
    )
    monkeypatch.setattr(
        runner_module,
        "discover_incoming_batch_paths",
        lambda *_args, **_kwargs: (),
    )

    def fail_unexpected_run(
        *_args: object,
        **_kwargs: object,
    ) -> BatchPipelineRunResult:
        raise AssertionError("No annual batch should be executed.")

    monkeypatch.setattr(
        runner_module,
        "run_batch_pipeline",
        fail_unexpected_run,
    )

    result = run_discovered_batches()

    assert result.batch_paths == ()
    assert result.results == ()
    assert result.discovered_count == 0
    assert result.processed_count == 0
    assert result.skipped_count == 0


def test_run_discovered_batches_is_fail_fast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A batch failure should prevent later batches from starting."""
    paths = tuple(
        Path(f"data/incoming/food_inspections_{year}.csv")
        for year in (
            2019,
            2020,
            2021,
        )
    )
    executed: list[Path] = []

    monkeypatch.setattr(
        runner_module,
        "load_config",
        lambda _path: _config(),
    )
    monkeypatch.setattr(
        runner_module,
        "discover_incoming_batch_paths",
        lambda *_args, **_kwargs: paths,
    )

    def fake_run_batch(
        file_path: str | Path,
        **_kwargs: object,
    ) -> BatchPipelineRunResult:
        path = Path(file_path)
        executed.append(path)

        if path == paths[1]:
            raise RuntimeError("Simulated annual batch failure.")

        return _processed_result(
            path,
            year=int(path.stem[-4:]),
        )

    monkeypatch.setattr(
        runner_module,
        "run_batch_pipeline",
        fake_run_batch,
    )

    with pytest.raises(
        RuntimeError,
        match="Simulated annual batch failure",
    ):
        run_discovered_batches()

    assert executed == [
        paths[0],
        paths[1],
    ]


def test_run_discovered_batches_rejects_result_path_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A result must identify the exact discovered path submitted."""
    discovered_path = Path("data/incoming/food_inspections_2019.csv")
    different_path = Path("data/incoming/food_inspections_2020.csv")

    monkeypatch.setattr(
        runner_module,
        "load_config",
        lambda _path: _config(),
    )
    monkeypatch.setattr(
        runner_module,
        "discover_incoming_batch_paths",
        lambda *_args, **_kwargs: (discovered_path,),
    )
    monkeypatch.setattr(
        runner_module,
        "run_batch_pipeline",
        lambda *_args, **_kwargs: _processed_result(
            different_path,
            year=2020,
        ),
    )

    with pytest.raises(
        MultiBatchPipelineRunError,
        match=("Pipeline result path does not match the discovered batch path"),
    ):
        run_discovered_batches()
