"""Orchestrate validation and publication for one annual source batch."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from data_quality_pipeline.batch import (
    IncomingBatch,
    inspect_incoming_batch,
)
from data_quality_pipeline.config import load_config
from data_quality_pipeline.contract import load_data_contract
from data_quality_pipeline.curated_writer import (
    CuratedParquetWriteResult,
    write_curated_parquet,
)
from data_quality_pipeline.ingestion import read_raw_batch
from data_quality_pipeline.quarantine import build_quarantine_records
from data_quality_pipeline.quarantine_writer import (
    QuarantineParquetWriteResult,
    write_quarantine_parquet,
)
from data_quality_pipeline.state_manifest import (
    BatchStateManifest,
    batch_state_manifest_path,
    build_batch_state_manifest,
    is_batch_state_current,
    read_batch_state_manifest,
    write_batch_state_manifest,
)
from data_quality_pipeline.transformation import (
    transform_curated_records,
)
from data_quality_pipeline.validation_runner import (
    BatchValidationResult,
    validate_batch_records,
)


@dataclass(frozen=True, slots=True)
class BatchPipelineRunResult:
    """Describe one successfully completed annual pipeline run."""

    batch: IncomingBatch
    validation: BatchValidationResult
    raw_row_count: int
    accepted_row_count: int
    rejected_record_count: int
    quarantine_issue_count: int
    curated_write: CuratedParquetWriteResult
    quarantine_write: QuarantineParquetWriteResult

    @property
    def error_count(self) -> int:
        """Return the number of blocking validation issues."""
        return len(self.validation.errors)

    @property
    def warning_count(self) -> int:
        """Return the number of non-blocking validation issues."""
        return len(self.validation.warnings)


@dataclass(frozen=True, slots=True)
class BatchPipelineSkipResult:
    """Describe one unchanged annual batch skipped from processing."""

    batch: IncomingBatch
    manifest: BatchStateManifest
    state_manifest_path: Path


type BatchPipelineResult = BatchPipelineRunResult | BatchPipelineSkipResult


class BatchPipelineRunError(Exception):
    """Raised when pipeline orchestration detects inconsistent state."""


def run_batch_pipeline(
    file_path: str | Path,
    *,
    config_path: str | Path = "config/pipeline.yaml",
    contract_path: str | Path = "config/data_contract.yaml",
) -> BatchPipelineResult:
    """Run the incremental data-quality pipeline for one annual CSV batch.

    The execution order is:

    1. Load configuration and data contract.
    2. Inspect and checksum the incoming file.
    3. Return a skip result when persisted state is still current.
    4. Read the raw CSV with its exact source schema.
    5. Validate every source record.
    6. Build issue-level quarantine records.
    7. Remove source rows having at least one blocking error.
    8. Transform accepted records to the curated schema.
    9. Persist quarantine and curated Parquet outputs.
    10. Persist the successful batch state manifest.

    Args:
        file_path: Annual incoming CSV file.
        config_path: Pipeline configuration file.
        contract_path: Data-contract configuration file.

    Returns:
        Immutable metadata for either a processed or skipped annual batch.

    Raises:
        BatchPipelineRunError: If internal row counts or issue mappings
            become inconsistent.
        Exception: Domain-specific configuration, ingestion, validation,
            transformation, or persistence errors are propagated unchanged.
    """
    config = load_config(config_path)
    contract = load_data_contract(contract_path)

    source_config = config["source"]
    ingestion_config = config["ingestion"]
    output_config = config["output"]
    paths_config = config["paths"]
    dataset = source_config["dataset"]

    expected_columns = contract["source_schema"]["expected_columns"]
    primary_key = contract["contract"]["primary_key"]

    batch = inspect_incoming_batch(
        file_path,
        start_year=source_config["start_year"],
        end_year=source_config["end_year"],
        hash_algorithm=ingestion_config["hash_algorithm"],
    )

    existing_manifest = read_batch_state_manifest(
        paths_config["state"],
        dataset=dataset,
        batch_year=batch.year,
    )

    if existing_manifest is not None and is_batch_state_current(
        batch, existing_manifest
    ):
        return BatchPipelineSkipResult(
            batch=batch,
            manifest=existing_manifest,
            state_manifest_path=batch_state_manifest_path(
                paths_config["state"],
                dataset=dataset,
                batch_year=batch.year,
            ),
        )

    raw = read_raw_batch(
        batch,
        expected_columns=expected_columns,
        encoding=ingestion_config["encoding"],
        delimiter=ingestion_config["delimiter"],
    )

    validation = validate_batch_records(
        raw,
        batch_year=batch.year,
        contract=contract,
    )

    quarantine = build_quarantine_records(
        raw,
        validation_result=validation,
        batch_year=batch.year,
        primary_key=primary_key,
    )

    rejected_positions = _rejected_source_positions(
        validation,
        source_row_count=len(raw),
    )
    accepted = _select_accepted_records(
        raw,
        rejected_positions=rejected_positions,
    )

    _validate_run_counts(
        raw_row_count=len(raw),
        accepted_row_count=len(accepted),
        rejected_record_count=len(rejected_positions),
        quarantine_issue_count=len(quarantine),
        error_count=len(validation.errors),
    )

    curated = transform_curated_records(
        accepted,
        batch_year=batch.year,
        contract=contract,
    )

    quarantine_write = write_quarantine_parquet(
        quarantine,
        output_dir=paths_config["quarantine"],
        batch_year=batch.year,
        output_format=output_config["format"],
        compression=output_config["compression"],
    )

    curated_write = write_curated_parquet(
        curated,
        output_dir=paths_config["curated"],
        batch_year=batch.year,
        output_format=output_config["format"],
        compression=output_config["compression"],
        partition_by=output_config["partition_by"],
    )

    result = BatchPipelineRunResult(
        batch=batch,
        validation=validation,
        raw_row_count=len(raw),
        accepted_row_count=len(accepted),
        rejected_record_count=len(rejected_positions),
        quarantine_issue_count=len(quarantine),
        curated_write=curated_write,
        quarantine_write=quarantine_write,
    )

    manifest = build_batch_state_manifest(
        result,
        dataset=dataset,
    )
    write_batch_state_manifest(
        manifest,
        output_dir=paths_config["state"],
    )

    return result


def _rejected_source_positions(
    validation: BatchValidationResult,
    *,
    source_row_count: int,
) -> frozenset[int]:
    """Return unique DataFrame positions having blocking issues."""
    positions: set[int] = set()

    for issue in validation.errors:
        position = issue.source_row_number - 2

        if not 0 <= position < source_row_count:
            raise BatchPipelineRunError(
                "Blocking issue references an unavailable CSV line: "
                f"{issue.source_row_number}."
            )

        positions.add(position)

    return frozenset(positions)


def _select_accepted_records(
    data: pd.DataFrame,
    *,
    rejected_positions: frozenset[int],
) -> pd.DataFrame:
    """Return source rows having no blocking validation issue."""
    accepted_mask = [
        position not in rejected_positions for position in range(len(data))
    ]

    return data.iloc[accepted_mask].copy().reset_index(drop=True)


def _validate_run_counts(
    *,
    raw_row_count: int,
    accepted_row_count: int,
    rejected_record_count: int,
    quarantine_issue_count: int,
    error_count: int,
) -> None:
    """Validate row-count invariants before publishing outputs."""
    if accepted_row_count + rejected_record_count != raw_row_count:
        raise BatchPipelineRunError(
            "Accepted and rejected record counts do not match the raw batch row count."
        )

    if quarantine_issue_count != error_count:
        raise BatchPipelineRunError(
            "Quarantine issue count does not match blocking validation issue count."
        )
