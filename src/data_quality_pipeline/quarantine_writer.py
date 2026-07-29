"""Persist issue-level quarantine records as partitioned Parquet files."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from data_quality_pipeline.quarantine import (
    QUARANTINE_METADATA_COLUMNS,
)

QUARANTINE_PARTITION_COLUMN = "dq_batch_year"


@dataclass(frozen=True, slots=True)
class QuarantineParquetWriteResult:
    """Describe one successfully written quarantine Parquet file."""

    path: Path
    row_count: int
    size_bytes: int
    partition_column: str
    partition_value: int
    compression: str


class QuarantineParquetWriteError(Exception):
    """Raised when quarantine output cannot be written safely."""


def write_quarantine_parquet(
    data: pd.DataFrame,
    *,
    output_dir: str | Path,
    batch_year: int,
    output_format: str,
    compression: str,
) -> QuarantineParquetWriteResult:
    """Write one quarantine batch to a deterministic year partition.

    The final path follows Hive-style partitioning:

    ``<output_dir>/dq_batch_year=<year>/``
    ``food_inspections_<year>_quarantine.parquet``

    A temporary file is written in the target partition and atomically
    replaces any previous quarantine output for the same batch year.

    Args:
        data: Issue-level quarantine records ready for persistence.
        output_dir: Root directory for quarantine data.
        batch_year: Year derived from the incoming batch filename.
        output_format: Configured output format.
        compression: Configured Parquet compression codec.

    Returns:
        Immutable metadata describing the written file.

    Raises:
        QuarantineParquetWriteError: If configuration, schema, partition
            values, or filesystem operations prevent safe persistence.
    """
    _validate_batch_year(batch_year)
    _validate_output_format(output_format)
    normalized_compression = _validate_compression(compression)
    _validate_schema(data)
    _validate_partition_values(
        data[QUARANTINE_PARTITION_COLUMN],
        batch_year=batch_year,
    )
    _validate_severities(data["dq_severity"])

    root_path = Path(output_dir)
    partition_path = root_path / (f"{QUARANTINE_PARTITION_COLUMN}={batch_year}")
    target_path = partition_path / (f"food_inspections_{batch_year}_quarantine.parquet")
    temporary_path = partition_path / (f".{target_path.name}.tmp")

    try:
        if root_path.exists() and not root_path.is_dir():
            raise QuarantineParquetWriteError(
                f"Quarantine output path is not a directory: {root_path}"
            )

        partition_path.mkdir(parents=True, exist_ok=True)
        _remove_temporary_file(temporary_path)

        table = pa.Table.from_pandas(
            data,
            preserve_index=False,
        )

        pq.write_table(
            table,
            temporary_path,
            compression=normalized_compression,
            write_statistics=True,
        )

        metadata = pq.ParquetFile(temporary_path).metadata

        if metadata.num_rows != len(data):
            raise QuarantineParquetWriteError(
                "Written Parquet row count does not match quarantine data."
            )

        temporary_path.replace(target_path)
        size_bytes = target_path.stat().st_size
    except QuarantineParquetWriteError:
        _remove_temporary_file(temporary_path)
        raise
    except (OSError, TypeError, ValueError) as exc:
        _remove_temporary_file(temporary_path)
        raise QuarantineParquetWriteError(
            f"Unable to write quarantine Parquet file: {target_path}"
        ) from exc

    return QuarantineParquetWriteResult(
        path=target_path,
        row_count=len(data),
        size_bytes=size_bytes,
        partition_column=QUARANTINE_PARTITION_COLUMN,
        partition_value=batch_year,
        compression=normalized_compression,
    )


def _validate_batch_year(batch_year: int) -> None:
    """Validate the quarantine partition year."""
    if (
        isinstance(batch_year, bool)
        or not isinstance(batch_year, int)
        or not 1 <= batch_year <= 9999
    ):
        raise QuarantineParquetWriteError(
            "Batch year must be an integer between 1 and 9999."
        )


def _validate_output_format(output_format: str) -> None:
    """Require the Parquet output format implemented by this writer."""
    if not isinstance(output_format, str) or output_format.strip().lower() != "parquet":
        raise QuarantineParquetWriteError("Quarantine output format must be 'parquet'.")


def _validate_compression(compression: str) -> str:
    """Validate and normalize the configured Parquet codec."""
    if not isinstance(compression, str) or not compression.strip():
        raise QuarantineParquetWriteError(
            "Parquet compression must be a non-empty string."
        )

    normalized = compression.strip().lower()

    try:
        available = pa.Codec.is_available(normalized)
    except ValueError as exc:
        raise QuarantineParquetWriteError(
            f"Unsupported Parquet compression codec: {normalized}"
        ) from exc

    if not available:
        raise QuarantineParquetWriteError(
            f"Unavailable Parquet compression codec: {normalized}"
        )

    return normalized


def _validate_schema(data: pd.DataFrame) -> None:
    """Require the complete issue-level quarantine metadata schema."""
    if not data.columns.is_unique:
        raise QuarantineParquetWriteError(
            "Quarantine data contains duplicate column names."
        )

    missing_columns = [
        column for column in QUARANTINE_METADATA_COLUMNS if column not in data.columns
    ]

    if missing_columns:
        raise QuarantineParquetWriteError(
            f"Quarantine data is missing metadata columns: {missing_columns}"
        )


def _validate_partition_values(
    values: pd.Series,
    *,
    batch_year: int,
) -> None:
    """Require every quarantine row to belong to the target batch."""
    if values.isna().any():
        raise QuarantineParquetWriteError(
            "Quarantine partition column contains null values: "
            f"{QUARANTINE_PARTITION_COLUMN}"
        )

    try:
        numeric_values = pd.to_numeric(
            values,
            errors="raise",
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise QuarantineParquetWriteError(
            "Quarantine partition column must contain integer years: "
            f"{QUARANTINE_PARTITION_COLUMN}"
        ) from exc

    if numeric_values.mod(1).ne(0).any():
        raise QuarantineParquetWriteError(
            "Quarantine partition column must contain integer years: "
            f"{QUARANTINE_PARTITION_COLUMN}"
        )

    actual_years = {int(value) for value in numeric_values.tolist()}

    if actual_years and actual_years != {batch_year}:
        raise QuarantineParquetWriteError(
            "Quarantine partition values do not match batch year "
            f"{batch_year}: {sorted(actual_years)}"
        )


def _validate_severities(values: pd.Series) -> None:
    """Prevent non-blocking warnings from entering quarantine output."""
    if values.isna().any():
        raise QuarantineParquetWriteError("Quarantine severity contains null values.")

    actual_severities = {str(value) for value in values.tolist()}

    if actual_severities and actual_severities != {"error"}:
        raise QuarantineParquetWriteError(
            "Quarantine data must contain only error severity: "
            f"{sorted(actual_severities)}"
        )


def _remove_temporary_file(path: Path) -> None:
    """Best-effort cleanup that preserves the original write failure."""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass
