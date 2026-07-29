"""Persist curated records as deterministic partitioned Parquet files."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


@dataclass(frozen=True, slots=True)
class CuratedParquetWriteResult:
    """Describe one successfully written curated Parquet file."""

    path: Path
    row_count: int
    size_bytes: int
    partition_column: str
    partition_value: int
    compression: str


class CuratedParquetWriteError(Exception):
    """Raised when curated Parquet output cannot be written safely."""


def write_curated_parquet(
    data: pd.DataFrame,
    *,
    output_dir: str | Path,
    batch_year: int,
    output_format: str,
    compression: str,
    partition_by: Sequence[str],
) -> CuratedParquetWriteResult:
    """Write one curated batch to its deterministic year partition.

    The final path follows Hive partitioning:

    ``<output_dir>/inspection_year=<year>/food_inspections_<year>.parquet``

    A temporary file is written in the target partition and atomically
    replaces any previous output for the same batch year.

    Args:
        data: Curated records ready for publication.
        output_dir: Root directory for curated data.
        batch_year: Year encoded in the incoming batch filename.
        output_format: Configured output format.
        compression: Configured Parquet compression codec.
        partition_by: Configured partition columns.

    Returns:
        Immutable metadata describing the written file.

    Raises:
        CuratedParquetWriteError: If configuration, partition values, or
            filesystem operations prevent safe persistence.
    """
    _validate_batch_year(batch_year)
    _validate_output_format(output_format)

    partition_column = _validate_partition_columns(partition_by)
    normalized_compression = _validate_compression(compression)

    if not data.columns.is_unique:
        raise CuratedParquetWriteError("Curated data contains duplicate column names.")

    if partition_column not in data.columns:
        raise CuratedParquetWriteError(
            f"Curated data is missing partition column: {partition_column}"
        )

    _validate_partition_values(
        data[partition_column],
        partition_column=partition_column,
        batch_year=batch_year,
    )

    root_path = Path(output_dir)
    partition_path = root_path / (f"{partition_column}={batch_year}")
    target_path = partition_path / (f"food_inspections_{batch_year}.parquet")
    temporary_path = partition_path / (f".{target_path.name}.tmp")

    try:
        if root_path.exists() and not root_path.is_dir():
            raise CuratedParquetWriteError(
                f"Curated output path is not a directory: {root_path}"
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
            raise CuratedParquetWriteError(
                "Written Parquet row count does not match curated data."
            )

        temporary_path.replace(target_path)
        size_bytes = target_path.stat().st_size
    except CuratedParquetWriteError:
        _remove_temporary_file(temporary_path)
        raise
    except (OSError, TypeError, ValueError) as exc:
        _remove_temporary_file(temporary_path)
        raise CuratedParquetWriteError(
            f"Unable to write curated Parquet file: {target_path}"
        ) from exc

    return CuratedParquetWriteResult(
        path=target_path,
        row_count=len(data),
        size_bytes=size_bytes,
        partition_column=partition_column,
        partition_value=batch_year,
        compression=normalized_compression,
    )


def _remove_temporary_file(path: Path) -> None:
    """Best-effort cleanup that preserves the original write failure."""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _validate_batch_year(batch_year: int) -> None:
    """Validate the partition year."""
    if (
        isinstance(batch_year, bool)
        or not isinstance(batch_year, int)
        or not 1 <= batch_year <= 9999
    ):
        raise CuratedParquetWriteError(
            "Batch year must be an integer between 1 and 9999."
        )


def _validate_output_format(output_format: str) -> None:
    """Require the Parquet output format implemented by this writer."""
    if not isinstance(output_format, str) or output_format.strip().lower() != "parquet":
        raise CuratedParquetWriteError("Curated output format must be 'parquet'.")


def _validate_partition_columns(
    partition_by: Sequence[str],
) -> str:
    """Require the current inspection-year partition contract."""
    if isinstance(partition_by, (str, bytes)):
        raise CuratedParquetWriteError(
            "Partition columns must be a sequence of strings."
        )

    columns = list(partition_by)

    if columns != ["inspection_year"]:
        raise CuratedParquetWriteError(
            "Curated output must be partitioned by inspection_year."
        )

    return columns[0]


def _validate_compression(compression: str) -> str:
    """Validate and normalize the configured Parquet codec."""
    if not isinstance(compression, str) or not compression.strip():
        raise CuratedParquetWriteError(
            "Parquet compression must be a non-empty string."
        )

    normalized = compression.strip().lower()

    try:
        available = pa.Codec.is_available(normalized)
    except ValueError as exc:
        raise CuratedParquetWriteError(
            f"Unsupported Parquet compression codec: {normalized}"
        ) from exc

    if not available:
        raise CuratedParquetWriteError(
            f"Unavailable Parquet compression codec: {normalized}"
        )

    return normalized


def _validate_partition_values(
    values: pd.Series,
    *,
    partition_column: str,
    batch_year: int,
) -> None:
    """Require every curated row to belong to the target partition."""
    if values.isna().any():
        raise CuratedParquetWriteError(
            f"Partition column contains null values: {partition_column}"
        )

    try:
        numeric_values = pd.to_numeric(
            values,
            errors="raise",
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise CuratedParquetWriteError(
            f"Partition column must contain integer years: {partition_column}"
        ) from exc

    fractional_mask = numeric_values.mod(1).ne(0)

    if fractional_mask.any():
        raise CuratedParquetWriteError(
            f"Partition column must contain integer years: {partition_column}"
        )

    actual_years = {int(value) for value in numeric_values.tolist()}

    if actual_years and actual_years != {batch_year}:
        raise CuratedParquetWriteError(
            f"Partition values do not match batch year "
            f"{batch_year}: {sorted(actual_years)}"
        )
