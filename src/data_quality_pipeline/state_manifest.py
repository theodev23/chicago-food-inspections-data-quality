"""Persist and evaluate incremental state for annual pipeline batches."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from data_quality_pipeline.batch import IncomingBatch
    from data_quality_pipeline.pipeline_runner import BatchPipelineRunResult


STATE_SCHEMA_VERSION = 1
_DATASET_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


@dataclass(frozen=True, slots=True)
class BatchStateManifest:
    """Describe one successfully published annual batch."""

    schema_version: int
    dataset: str
    batch_year: int
    completed_at_utc: str
    source_path: str
    source_size_bytes: int
    checksum_algorithm: str
    checksum: str
    raw_row_count: int
    accepted_row_count: int
    rejected_record_count: int
    quarantine_issue_count: int
    error_count: int
    warning_count: int
    curated_path: str
    curated_row_count: int
    curated_size_bytes: int
    curated_compression: str
    quarantine_path: str
    quarantine_row_count: int
    quarantine_size_bytes: int
    quarantine_compression: str


@dataclass(frozen=True, slots=True)
class BatchStateWriteResult:
    """Describe one successfully written state manifest."""

    path: Path
    size_bytes: int


class BatchStateManifestError(Exception):
    """Raised when incremental batch state is invalid or unavailable."""


def build_batch_state_manifest(
    result: BatchPipelineRunResult,
    *,
    dataset: str,
    completed_at_utc: str | None = None,
) -> BatchStateManifest:
    """Build state metadata from one successful pipeline run."""
    completed_at = completed_at_utc or _current_utc_timestamp()

    manifest = BatchStateManifest(
        schema_version=STATE_SCHEMA_VERSION,
        dataset=dataset,
        batch_year=result.batch.year,
        completed_at_utc=completed_at,
        source_path=str(result.batch.path.resolve()),
        source_size_bytes=result.batch.size_bytes,
        checksum_algorithm=result.batch.checksum_algorithm,
        checksum=result.batch.checksum,
        raw_row_count=result.raw_row_count,
        accepted_row_count=result.accepted_row_count,
        rejected_record_count=result.rejected_record_count,
        quarantine_issue_count=result.quarantine_issue_count,
        error_count=result.error_count,
        warning_count=result.warning_count,
        curated_path=str(result.curated_write.path.resolve()),
        curated_row_count=result.curated_write.row_count,
        curated_size_bytes=result.curated_write.size_bytes,
        curated_compression=result.curated_write.compression,
        quarantine_path=str(result.quarantine_write.path.resolve()),
        quarantine_row_count=result.quarantine_write.row_count,
        quarantine_size_bytes=result.quarantine_write.size_bytes,
        quarantine_compression=result.quarantine_write.compression,
    )

    _validate_manifest(manifest)
    return manifest


def write_batch_state_manifest(
    manifest: BatchStateManifest,
    *,
    output_dir: str | Path,
) -> BatchStateWriteResult:
    """Atomically persist one annual state manifest as JSON."""
    _validate_manifest(manifest)

    root_path = Path(output_dir)
    target_path = batch_state_manifest_path(
        root_path,
        dataset=manifest.dataset,
        batch_year=manifest.batch_year,
    )
    temporary_path = target_path.with_name(f".{target_path.name}.tmp")
    serialized = (
        json.dumps(
            asdict(manifest),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    try:
        if root_path.exists() and not root_path.is_dir():
            raise BatchStateManifestError(
                f"State output path is not a directory: {root_path}"
            )

        root_path.mkdir(parents=True, exist_ok=True)
        _remove_temporary_file(temporary_path)

        temporary_path.write_text(
            serialized,
            encoding="utf-8",
        )
        temporary_path.replace(target_path)

        size_bytes = target_path.stat().st_size
    except BatchStateManifestError:
        _remove_temporary_file(temporary_path)
        raise
    except OSError as exc:
        _remove_temporary_file(temporary_path)
        raise BatchStateManifestError(
            f"Unable to write batch state manifest: {target_path}"
        ) from exc

    return BatchStateWriteResult(
        path=target_path,
        size_bytes=size_bytes,
    )


def read_batch_state_manifest(
    output_dir: str | Path,
    *,
    dataset: str,
    batch_year: int,
) -> BatchStateManifest | None:
    """Read one annual state manifest, returning ``None`` when absent."""
    _validate_dataset(dataset)
    _validate_batch_year(batch_year)

    path = batch_state_manifest_path(
        output_dir,
        dataset=dataset,
        batch_year=batch_year,
    )

    if not path.exists():
        return None

    if not path.is_file():
        raise BatchStateManifestError(f"Batch state manifest is not a file: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))

        if not isinstance(payload, dict):
            raise BatchStateManifestError(
                f"Batch state manifest must contain a JSON object: {path}"
            )

        manifest = BatchStateManifest(**payload)
    except BatchStateManifestError:
        raise
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        raise BatchStateManifestError(
            f"Unable to read batch state manifest: {path}"
        ) from exc

    _validate_manifest(manifest)

    if manifest.dataset != dataset:
        raise BatchStateManifestError(
            "Batch state dataset does not match requested dataset: "
            f"{manifest.dataset!r} != {dataset!r}."
        )

    if manifest.batch_year != batch_year:
        raise BatchStateManifestError(
            "Batch state year does not match requested year: "
            f"{manifest.batch_year} != {batch_year}."
        )

    return manifest


def is_batch_state_current(
    batch: IncomingBatch,
    manifest: BatchStateManifest,
) -> bool:
    """Return whether source fingerprint and published outputs are current."""
    _validate_manifest(manifest)

    return (
        manifest.batch_year == batch.year
        and manifest.source_size_bytes == batch.size_bytes
        and manifest.checksum_algorithm.lower() == batch.checksum_algorithm.lower()
        and manifest.checksum == batch.checksum
        and _output_matches(
            manifest.curated_path,
            expected_size_bytes=manifest.curated_size_bytes,
        )
        and _output_matches(
            manifest.quarantine_path,
            expected_size_bytes=manifest.quarantine_size_bytes,
        )
    )


def batch_state_manifest_path(
    output_dir: str | Path,
    *,
    dataset: str,
    batch_year: int,
) -> Path:
    """Return the deterministic path for one annual state manifest."""
    _validate_dataset(dataset)
    _validate_batch_year(batch_year)

    return Path(output_dir) / f"{dataset}_{batch_year}.json"


def _validate_manifest(manifest: BatchStateManifest) -> None:
    """Validate state schema and reconciliation invariants."""
    if manifest.schema_version != STATE_SCHEMA_VERSION:
        raise BatchStateManifestError(
            f"Unsupported batch state schema version: {manifest.schema_version}."
        )

    _validate_dataset(manifest.dataset)
    _validate_batch_year(manifest.batch_year)
    _validate_timestamp(manifest.completed_at_utc)

    _validate_non_empty_string(
        manifest.source_path,
        field_name="source_path",
    )
    _validate_non_empty_string(
        manifest.checksum_algorithm,
        field_name="checksum_algorithm",
    )
    _validate_non_empty_string(
        manifest.checksum,
        field_name="checksum",
    )
    _validate_non_empty_string(
        manifest.curated_path,
        field_name="curated_path",
    )
    _validate_non_empty_string(
        manifest.curated_compression,
        field_name="curated_compression",
    )
    _validate_non_empty_string(
        manifest.quarantine_path,
        field_name="quarantine_path",
    )
    _validate_non_empty_string(
        manifest.quarantine_compression,
        field_name="quarantine_compression",
    )

    count_fields = {
        "source_size_bytes": manifest.source_size_bytes,
        "raw_row_count": manifest.raw_row_count,
        "accepted_row_count": manifest.accepted_row_count,
        "rejected_record_count": manifest.rejected_record_count,
        "quarantine_issue_count": manifest.quarantine_issue_count,
        "error_count": manifest.error_count,
        "warning_count": manifest.warning_count,
        "curated_row_count": manifest.curated_row_count,
        "curated_size_bytes": manifest.curated_size_bytes,
        "quarantine_row_count": manifest.quarantine_row_count,
        "quarantine_size_bytes": manifest.quarantine_size_bytes,
    }

    for field_name, value in count_fields.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise BatchStateManifestError(
                f"Batch state field must be a non-negative integer: {field_name}."
            )

    if (
        manifest.accepted_row_count + manifest.rejected_record_count
        != manifest.raw_row_count
    ):
        raise BatchStateManifestError(
            "Batch state accepted and rejected counts do not "
            "reconcile to the raw row count."
        )

    if manifest.curated_row_count != manifest.accepted_row_count:
        raise BatchStateManifestError(
            "Batch state curated row count does not match the accepted row count."
        )

    if manifest.quarantine_row_count != manifest.quarantine_issue_count:
        raise BatchStateManifestError(
            "Batch state quarantine row count does not match "
            "the quarantine issue count."
        )

    if manifest.error_count != manifest.quarantine_issue_count:
        raise BatchStateManifestError(
            "Batch state error count does not match the quarantine issue count."
        )


def _validate_dataset(dataset: str) -> None:
    """Validate the dataset identifier used in state filenames."""
    if not isinstance(dataset, str) or not _DATASET_PATTERN.fullmatch(dataset):
        raise BatchStateManifestError(
            "Dataset must contain only letters, digits, underscores, and hyphens."
        )


def _validate_batch_year(batch_year: int) -> None:
    """Validate the annual batch year."""
    if (
        isinstance(batch_year, bool)
        or not isinstance(batch_year, int)
        or not 1 <= batch_year <= 9999
    ):
        raise BatchStateManifestError(
            "Batch year must be an integer between 1 and 9999."
        )


def _validate_timestamp(value: str) -> None:
    """Require an ISO-8601 timezone-aware completion timestamp."""
    if not isinstance(value, str) or not value:
        raise BatchStateManifestError(
            "Batch completion timestamp must be a non-empty string."
        )

    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise BatchStateManifestError(
            "Batch completion timestamp must use ISO-8601 format."
        ) from exc

    if parsed.tzinfo is None:
        raise BatchStateManifestError(
            "Batch completion timestamp must include a timezone."
        )


def _validate_non_empty_string(
    value: str,
    *,
    field_name: str,
) -> None:
    """Validate one required manifest string field."""
    if not isinstance(value, str) or not value:
        raise BatchStateManifestError(
            f"Batch state field must be a non-empty string: {field_name}."
        )


def _output_matches(
    path_value: str,
    *,
    expected_size_bytes: int,
) -> bool:
    """Check that a recorded output file still exists unchanged in size."""
    path = Path(path_value)

    try:
        return path.is_file() and path.stat().st_size == expected_size_bytes
    except OSError:
        return False


def _current_utc_timestamp() -> str:
    """Return a compact ISO-8601 UTC timestamp."""
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _remove_temporary_file(path: Path) -> None:
    """Best-effort cleanup that preserves the original state failure."""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass
