"""Unit tests for incremental annual batch state manifests."""

import json
from dataclasses import FrozenInstanceError, asdict, replace
from pathlib import Path

import pytest

from data_quality_pipeline.batch import IncomingBatch
from data_quality_pipeline.curated_writer import CuratedParquetWriteResult
from data_quality_pipeline.pipeline_runner import BatchPipelineRunResult
from data_quality_pipeline.quarantine_writer import (
    QuarantineParquetWriteResult,
)
from data_quality_pipeline.state_manifest import (
    STATE_SCHEMA_VERSION,
    BatchStateManifest,
    BatchStateManifestError,
    BatchStateWriteResult,
    batch_state_manifest_path,
    build_batch_state_manifest,
    is_batch_state_current,
    read_batch_state_manifest,
    write_batch_state_manifest,
)
from data_quality_pipeline.validation import RecordIssue
from data_quality_pipeline.validation_runner import BatchValidationResult


def _issue(*, severity: str) -> RecordIssue:
    """Build one validation issue."""
    return RecordIssue(
        source_row_number=2,
        inspection_id="100",
        rule_id="test_rule",
        column="results",
        value="UNKNOWN",
        message="Test issue.",
        severity=severity,
    )


def _run_result(tmp_path: Path) -> BatchPipelineRunResult:
    """Build complete successful-run metadata backed by real files."""
    source_path = tmp_path / "food_inspections_2019.csv"
    source_path.write_bytes(b"source-data")

    curated_path = (
        tmp_path / "curated" / "inspection_year=2019" / "food_inspections_2019.parquet"
    )
    quarantine_path = (
        tmp_path
        / "quarantine"
        / "dq_batch_year=2019"
        / "food_inspections_2019_quarantine.parquet"
    )

    curated_path.parent.mkdir(parents=True)
    quarantine_path.parent.mkdir(parents=True)
    curated_path.write_bytes(b"curated-output")
    quarantine_path.write_bytes(b"quarantine-output")

    batch = IncomingBatch(
        path=source_path,
        year=2019,
        size_bytes=source_path.stat().st_size,
        checksum="abc123",
        checksum_algorithm="sha256",
    )
    validation = BatchValidationResult(
        issues=(
            _issue(severity="error"),
            _issue(severity="warning"),
        ),
        exact_duplicates=(),
        duplicate_detection_skipped=False,
    )
    curated_write = CuratedParquetWriteResult(
        path=curated_path,
        row_count=2,
        size_bytes=curated_path.stat().st_size,
        partition_column="inspection_year",
        partition_value=2019,
        compression="snappy",
    )
    quarantine_write = QuarantineParquetWriteResult(
        path=quarantine_path,
        row_count=1,
        size_bytes=quarantine_path.stat().st_size,
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


def _manifest(tmp_path: Path) -> BatchStateManifest:
    """Build a valid deterministic state manifest."""
    return build_batch_state_manifest(
        _run_result(tmp_path),
        dataset="chicago_food_inspections",
        completed_at_utc="2026-07-30T08:00:00Z",
    )


def test_state_result_models_are_immutable() -> None:
    """Manifest and write-result metadata must remain immutable."""
    manifest = BatchStateManifest(
        schema_version=1,
        dataset="dataset",
        batch_year=2019,
        completed_at_utc="2026-07-30T08:00:00Z",
        source_path="/source.csv",
        source_size_bytes=1,
        checksum_algorithm="sha256",
        checksum="abc",
        raw_row_count=1,
        accepted_row_count=1,
        rejected_record_count=0,
        quarantine_issue_count=0,
        error_count=0,
        warning_count=0,
        curated_path="/curated.parquet",
        curated_row_count=1,
        curated_size_bytes=1,
        curated_compression="snappy",
        quarantine_path="/quarantine.parquet",
        quarantine_row_count=0,
        quarantine_size_bytes=1,
        quarantine_compression="snappy",
    )
    write_result = BatchStateWriteResult(
        path=Path("state.json"),
        size_bytes=100,
    )

    with pytest.raises(FrozenInstanceError):
        manifest.batch_year = 2020

    with pytest.raises(FrozenInstanceError):
        write_result.size_bytes = 200


def test_build_batch_state_manifest_maps_successful_run(
    tmp_path: Path,
) -> None:
    """Successful-run metadata should map completely to state."""
    result = _run_result(tmp_path)

    manifest = build_batch_state_manifest(
        result,
        dataset="chicago_food_inspections",
        completed_at_utc="2026-07-30T08:00:00Z",
    )

    assert manifest.schema_version == STATE_SCHEMA_VERSION
    assert manifest.dataset == "chicago_food_inspections"
    assert manifest.batch_year == 2019
    assert manifest.completed_at_utc == "2026-07-30T08:00:00Z"
    assert manifest.source_path == str(result.batch.path.resolve())
    assert manifest.source_size_bytes == result.batch.size_bytes
    assert manifest.checksum_algorithm == "sha256"
    assert manifest.checksum == "abc123"
    assert manifest.raw_row_count == 3
    assert manifest.accepted_row_count == 2
    assert manifest.rejected_record_count == 1
    assert manifest.quarantine_issue_count == 1
    assert manifest.error_count == 1
    assert manifest.warning_count == 1
    assert manifest.curated_path == str(result.curated_write.path.resolve())
    assert manifest.curated_row_count == 2
    assert manifest.curated_size_bytes == (result.curated_write.size_bytes)
    assert manifest.quarantine_path == str(result.quarantine_write.path.resolve())
    assert manifest.quarantine_row_count == 1


def test_write_and_read_batch_state_manifest_round_trip(
    tmp_path: Path,
) -> None:
    """State should survive deterministic JSON persistence."""
    manifest = _manifest(tmp_path)
    state_dir = tmp_path / "state"

    write_result = write_batch_state_manifest(
        manifest,
        output_dir=state_dir,
    )
    reloaded = read_batch_state_manifest(
        state_dir,
        dataset="chicago_food_inspections",
        batch_year=2019,
    )

    assert write_result.path == (state_dir / "chicago_food_inspections_2019.json")
    assert write_result.path.is_file()
    assert write_result.size_bytes == write_result.path.stat().st_size
    assert reloaded == manifest

    serialized = write_result.path.read_text(encoding="utf-8")

    assert serialized.endswith("\n")
    assert json.loads(serialized) == asdict(manifest)


def test_write_batch_state_manifest_replaces_existing_state(
    tmp_path: Path,
) -> None:
    """A rerun should atomically replace the previous annual state."""
    state_dir = tmp_path / "state"
    first = _manifest(tmp_path)
    second = replace(first, warning_count=2)

    first_result = write_batch_state_manifest(
        first,
        output_dir=state_dir,
    )
    second_result = write_batch_state_manifest(
        second,
        output_dir=state_dir,
    )
    reloaded = read_batch_state_manifest(
        state_dir,
        dataset="chicago_food_inspections",
        batch_year=2019,
    )

    assert second_result.path == first_result.path
    assert reloaded == second
    assert not list(state_dir.glob("*.tmp"))


def test_read_batch_state_manifest_returns_none_when_absent(
    tmp_path: Path,
) -> None:
    """An unprocessed annual batch should have no state."""
    result = read_batch_state_manifest(
        tmp_path / "state",
        dataset="chicago_food_inspections",
        batch_year=2019,
    )

    assert result is None


def test_is_batch_state_current_accepts_matching_state(
    tmp_path: Path,
) -> None:
    """Matching fingerprints and outputs should mark a batch current."""
    result = _run_result(tmp_path)
    manifest = build_batch_state_manifest(
        result,
        dataset="chicago_food_inspections",
        completed_at_utc="2026-07-30T08:00:00Z",
    )

    assert is_batch_state_current(result.batch, manifest)


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("size_bytes", 999),
        ("checksum_algorithm", "md5"),
        ("checksum", "different"),
    ],
)
def test_is_batch_state_current_rejects_source_fingerprint_changes(
    tmp_path: Path,
    field_name: str,
    field_value: object,
) -> None:
    """Any source fingerprint change should require reprocessing."""
    result = _run_result(tmp_path)
    manifest = build_batch_state_manifest(
        result,
        dataset="chicago_food_inspections",
        completed_at_utc="2026-07-30T08:00:00Z",
    )
    changed_batch = replace(
        result.batch,
        **{field_name: field_value},
    )

    assert not is_batch_state_current(
        changed_batch,
        manifest,
    )


@pytest.mark.parametrize(
    ("field_name", "mode"),
    [
        ("curated_path", "missing"),
        ("curated_size_bytes", "size"),
        ("quarantine_path", "missing"),
        ("quarantine_size_bytes", "size"),
    ],
)
def test_is_batch_state_current_rejects_changed_outputs(
    tmp_path: Path,
    field_name: str,
    mode: str,
) -> None:
    """Missing or resized published files should invalidate state."""
    result = _run_result(tmp_path)
    manifest = build_batch_state_manifest(
        result,
        dataset="chicago_food_inspections",
        completed_at_utc="2026-07-30T08:00:00Z",
    )

    if mode == "missing":
        field_value: object = str(tmp_path / f"missing-{field_name}")
    else:
        field_value = getattr(manifest, field_name) + 1

    changed_manifest = replace(
        manifest,
        **{field_name: field_value},
    )

    assert not is_batch_state_current(
        result.batch,
        changed_manifest,
    )


@pytest.mark.parametrize(
    "dataset",
    [
        "",
        "bad dataset",
        "../escape",
        "_leading",
        None,
    ],
)
def test_batch_state_manifest_path_rejects_invalid_dataset(
    tmp_path: Path,
    dataset: object,
) -> None:
    """Dataset identifiers must be safe for deterministic filenames."""
    with pytest.raises(
        BatchStateManifestError,
        match="Dataset must contain only letters",
    ):
        batch_state_manifest_path(
            tmp_path,
            dataset=dataset,
            batch_year=2019,
        )


@pytest.mark.parametrize(
    "batch_year",
    [
        True,
        0,
        10000,
        2019.0,
    ],
)
def test_batch_state_manifest_path_rejects_invalid_year(
    tmp_path: Path,
    batch_year: object,
) -> None:
    """State filenames require a valid integer year."""
    with pytest.raises(
        BatchStateManifestError,
        match="Batch year must be an integer between 1 and 9999",
    ):
        batch_state_manifest_path(
            tmp_path,
            dataset="dataset",
            batch_year=batch_year,
        )


def test_write_batch_state_manifest_rejects_schema_version(
    tmp_path: Path,
) -> None:
    """Unsupported state schemas must fail before persistence."""
    manifest = replace(
        _manifest(tmp_path),
        schema_version=999,
    )

    with pytest.raises(
        BatchStateManifestError,
        match="Unsupported batch state schema version",
    ):
        write_batch_state_manifest(
            manifest,
            output_dir=tmp_path / "state",
        )


@pytest.mark.parametrize(
    "timestamp",
    [
        "",
        "not-a-timestamp",
        "2026-07-30T08:00:00",
    ],
)
def test_write_batch_state_manifest_rejects_invalid_timestamp(
    tmp_path: Path,
    timestamp: str,
) -> None:
    """Completion timestamps must be timezone-aware ISO-8601 values."""
    manifest = replace(
        _manifest(tmp_path),
        completed_at_utc=timestamp,
    )

    with pytest.raises(BatchStateManifestError):
        write_batch_state_manifest(
            manifest,
            output_dir=tmp_path / "state",
        )


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("source_size_bytes", -1),
        ("raw_row_count", -1),
        ("curated_size_bytes", True),
    ],
)
def test_write_batch_state_manifest_rejects_invalid_counts(
    tmp_path: Path,
    field_name: str,
    field_value: object,
) -> None:
    """Manifest counts and byte sizes must be non-negative integers."""
    manifest = replace(
        _manifest(tmp_path),
        **{field_name: field_value},
    )

    with pytest.raises(
        BatchStateManifestError,
        match="must be a non-negative integer",
    ):
        write_batch_state_manifest(
            manifest,
            output_dir=tmp_path / "state",
        )


def test_write_batch_state_manifest_requires_record_reconciliation(
    tmp_path: Path,
) -> None:
    """Accepted and rejected records must reconcile to raw rows."""
    manifest = replace(
        _manifest(tmp_path),
        accepted_row_count=1,
    )

    with pytest.raises(
        BatchStateManifestError,
        match="accepted and rejected counts",
    ):
        write_batch_state_manifest(
            manifest,
            output_dir=tmp_path / "state",
        )


def test_write_batch_state_manifest_requires_curated_reconciliation(
    tmp_path: Path,
) -> None:
    """Curated rows must equal accepted source records."""
    manifest = replace(
        _manifest(tmp_path),
        curated_row_count=1,
    )

    with pytest.raises(
        BatchStateManifestError,
        match="curated row count",
    ):
        write_batch_state_manifest(
            manifest,
            output_dir=tmp_path / "state",
        )


def test_write_batch_state_manifest_requires_quarantine_reconciliation(
    tmp_path: Path,
) -> None:
    """Quarantine rows must equal issue-level diagnostics."""
    manifest = replace(
        _manifest(tmp_path),
        quarantine_row_count=0,
    )

    with pytest.raises(
        BatchStateManifestError,
        match="quarantine row count",
    ):
        write_batch_state_manifest(
            manifest,
            output_dir=tmp_path / "state",
        )


def test_write_batch_state_manifest_requires_error_reconciliation(
    tmp_path: Path,
) -> None:
    """Every blocking issue must have a quarantine diagnostic."""
    manifest = replace(
        _manifest(tmp_path),
        error_count=0,
    )

    with pytest.raises(
        BatchStateManifestError,
        match="error count",
    ):
        write_batch_state_manifest(
            manifest,
            output_dir=tmp_path / "state",
        )


def test_write_batch_state_manifest_rejects_file_output_root(
    tmp_path: Path,
) -> None:
    """The configured state root must be a directory."""
    output_path = tmp_path / "state"
    output_path.write_text("not a directory", encoding="utf-8")

    with pytest.raises(
        BatchStateManifestError,
        match="State output path is not a directory",
    ):
        write_batch_state_manifest(
            _manifest(tmp_path),
            output_dir=output_path,
        )


@pytest.mark.parametrize(
    "serialized",
    [
        "{not valid json",
        "[]",
    ],
)
def test_read_batch_state_manifest_rejects_invalid_json(
    tmp_path: Path,
    serialized: str,
) -> None:
    """Malformed or non-object JSON must not be accepted as state."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()

    path = batch_state_manifest_path(
        state_dir,
        dataset="dataset",
        batch_year=2019,
    )
    path.write_text(serialized, encoding="utf-8")

    with pytest.raises(BatchStateManifestError):
        read_batch_state_manifest(
            state_dir,
            dataset="dataset",
            batch_year=2019,
        )


def test_read_batch_state_manifest_rejects_dataset_mismatch(
    tmp_path: Path,
) -> None:
    """Manifest content must match the requested dataset."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    manifest = replace(
        _manifest(tmp_path),
        dataset="other_dataset",
    )

    path = batch_state_manifest_path(
        state_dir,
        dataset="requested_dataset",
        batch_year=2019,
    )
    path.write_text(
        json.dumps(asdict(manifest)),
        encoding="utf-8",
    )

    with pytest.raises(
        BatchStateManifestError,
        match="dataset does not match",
    ):
        read_batch_state_manifest(
            state_dir,
            dataset="requested_dataset",
            batch_year=2019,
        )


def test_read_batch_state_manifest_rejects_year_mismatch(
    tmp_path: Path,
) -> None:
    """Manifest content must match the requested annual batch."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    manifest = replace(
        _manifest(tmp_path),
        batch_year=2020,
    )

    path = batch_state_manifest_path(
        state_dir,
        dataset="chicago_food_inspections",
        batch_year=2019,
    )
    path.write_text(
        json.dumps(asdict(manifest)),
        encoding="utf-8",
    )

    with pytest.raises(
        BatchStateManifestError,
        match="year does not match",
    ):
        read_batch_state_manifest(
            state_dir,
            dataset="chicago_food_inspections",
            batch_year=2019,
        )


def test_write_batch_state_manifest_removes_partial_temporary_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed atomic replacement should not leave temporary state."""
    original_replace = Path.replace

    def fail_temporary_replace(
        path: Path,
        target: str | Path,
    ) -> Path:
        if path.name.endswith(".tmp"):
            raise OSError("Simulated state replacement failure.")

        return original_replace(path, target)

    monkeypatch.setattr(
        Path,
        "replace",
        fail_temporary_replace,
    )

    state_dir = tmp_path / "state"
    temporary_path = state_dir / ".chicago_food_inspections_2019.json.tmp"

    with pytest.raises(
        BatchStateManifestError,
        match="Unable to write batch state manifest",
    ):
        write_batch_state_manifest(
            _manifest(tmp_path),
            output_dir=state_dir,
        )

    assert not temporary_path.exists()
