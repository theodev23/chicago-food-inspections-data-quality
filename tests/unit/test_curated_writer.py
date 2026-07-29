"""Unit tests for curated Parquet persistence."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest
from pandas.testing import assert_frame_equal

import data_quality_pipeline.curated_writer as curated_writer_module
from data_quality_pipeline.curated_writer import (
    CuratedParquetWriteError,
    CuratedParquetWriteResult,
    write_curated_parquet,
)


def _curated_frame(
    inspection_ids: list[int] | None = None,
    *,
    inspection_year: int = 2019,
) -> pd.DataFrame:
    """Build a small curated frame suitable for Parquet tests."""
    identifiers = inspection_ids or [1, 2]

    return pd.DataFrame(
        {
            "inspection_id": pd.Series(
                identifiers,
                dtype="int64",
            ),
            "dba_name": pd.Series(
                [f"RESTAURANT {identifier}" for identifier in identifiers],
                dtype="string",
            ),
            "inspection_year": pd.Series(
                [inspection_year] * len(identifiers),
                dtype="int16",
            ),
        }
    )


def _write(
    data: pd.DataFrame,
    output_dir: Path,
    *,
    batch_year: int = 2019,
    output_format: object = "parquet",
) -> CuratedParquetWriteResult:
    """Write one test frame with the standard output configuration."""
    return write_curated_parquet(
        data,
        output_dir=output_dir,
        batch_year=batch_year,
        output_format=output_format,
        compression="snappy",
        partition_by=["inspection_year"],
    )


def test_curated_parquet_write_result_is_immutable() -> None:
    """Written-file metadata must not change after construction."""
    result = CuratedParquetWriteResult(
        path=Path("example.parquet"),
        row_count=1,
        size_bytes=100,
        partition_column="inspection_year",
        partition_value=2019,
        compression="snappy",
    )

    with pytest.raises(FrozenInstanceError):
        result.row_count = 2


def test_write_curated_parquet_writes_expected_partitioned_file(
    tmp_path: Path,
) -> None:
    """The writer should create a readable Snappy-compressed file."""
    data = _curated_frame()

    result = _write(
        data,
        tmp_path / "curated",
    )

    expected_path = (
        tmp_path / "curated" / "inspection_year=2019" / "food_inspections_2019.parquet"
    )

    assert result.path == expected_path
    assert result.path.is_file()
    assert result.row_count == 2
    assert result.size_bytes == result.path.stat().st_size
    assert result.size_bytes > 0
    assert result.partition_column == "inspection_year"
    assert result.partition_value == 2019
    assert result.compression == "snappy"

    parquet_file = pq.ParquetFile(result.path)
    metadata = parquet_file.metadata

    assert metadata.num_rows == 2
    assert metadata.num_columns == 3
    assert metadata.row_group(0).column(0).compression == "SNAPPY"

    reloaded = pd.read_parquet(result.path)

    assert reloaded["inspection_id"].tolist() == [1, 2]
    assert reloaded["inspection_year"].tolist() == [2019, 2019]


def test_write_curated_parquet_replaces_existing_batch_output(
    tmp_path: Path,
) -> None:
    """A rerun for one year should replace its previous output."""
    output_dir = tmp_path / "curated"

    first_result = _write(
        _curated_frame([1, 2]),
        output_dir,
    )
    second_result = _write(
        _curated_frame([3]),
        output_dir,
    )

    assert second_result.path == first_result.path
    assert second_result.row_count == 1

    reloaded = pd.read_parquet(second_result.path)

    assert reloaded["inspection_id"].tolist() == [3]
    assert not any(
        path.name.endswith(".tmp") for path in second_result.path.parent.iterdir()
    )


def test_write_curated_parquet_does_not_modify_input(
    tmp_path: Path,
) -> None:
    """Persistence must not mutate the curated DataFrame."""
    data = _curated_frame()
    original = data.copy(deep=True)

    _write(
        data,
        tmp_path / "curated",
    )

    assert_frame_equal(data, original)


@pytest.mark.parametrize(
    "batch_year",
    [
        True,
        0,
        10000,
        2019.0,
    ],
)
def test_write_curated_parquet_rejects_invalid_batch_year(
    tmp_path: Path,
    batch_year: object,
) -> None:
    """The partition year must be a real integer in the valid range."""
    with pytest.raises(
        CuratedParquetWriteError,
        match="Batch year must be an integer between 1 and 9999",
    ):
        _write(
            _curated_frame(),
            tmp_path / "curated",
            batch_year=batch_year,
        )


@pytest.mark.parametrize(
    "output_format",
    [
        "csv",
        "",
        None,
    ],
)
def test_write_curated_parquet_rejects_invalid_output_format(
    tmp_path: Path,
    output_format: object,
) -> None:
    """The writer should accept only the implemented Parquet format."""
    with pytest.raises(
        CuratedParquetWriteError,
        match="Curated output format must be 'parquet'",
    ):
        _write(
            _curated_frame(),
            tmp_path / "curated",
            output_format=output_format,
        )


def test_write_curated_parquet_rejects_string_partition_columns(
    tmp_path: Path,
) -> None:
    """A string must not be interpreted as a partition sequence."""
    with pytest.raises(
        CuratedParquetWriteError,
        match="Partition columns must be a sequence of strings",
    ):
        write_curated_parquet(
            _curated_frame(),
            output_dir=tmp_path / "curated",
            batch_year=2019,
            output_format="parquet",
            compression="snappy",
            partition_by="inspection_year",
        )


@pytest.mark.parametrize(
    "partition_by",
    [
        [],
        ["state"],
        ["inspection_year", "state"],
    ],
)
def test_write_curated_parquet_requires_inspection_year_partition(
    tmp_path: Path,
    partition_by: list[str],
) -> None:
    """The current output contract supports one year partition only."""
    with pytest.raises(
        CuratedParquetWriteError,
        match="Curated output must be partitioned by inspection_year",
    ):
        write_curated_parquet(
            _curated_frame(),
            output_dir=tmp_path / "curated",
            batch_year=2019,
            output_format="parquet",
            compression="snappy",
            partition_by=partition_by,
        )


def test_write_curated_parquet_rejects_duplicate_columns(
    tmp_path: Path,
) -> None:
    """Duplicate curated column names should be rejected."""
    data = _curated_frame()
    data.columns = [
        "inspection_id",
        "inspection_id",
        "inspection_year",
    ]

    with pytest.raises(
        CuratedParquetWriteError,
        match="Curated data contains duplicate column names",
    ):
        _write(
            data,
            tmp_path / "curated",
        )


def test_write_curated_parquet_requires_partition_column(
    tmp_path: Path,
) -> None:
    """The curated frame must contain the configured partition column."""
    data = _curated_frame().drop(columns=["inspection_year"])

    with pytest.raises(
        CuratedParquetWriteError,
        match="Curated data is missing partition column: inspection_year",
    ):
        _write(
            data,
            tmp_path / "curated",
        )


def test_write_curated_parquet_rejects_null_partition_value(
    tmp_path: Path,
) -> None:
    """Every curated record must have a partition value."""
    data = _curated_frame()
    data.loc[0, "inspection_year"] = pd.NA

    with pytest.raises(
        CuratedParquetWriteError,
        match="Partition column contains null values: inspection_year",
    ):
        _write(
            data,
            tmp_path / "curated",
        )


@pytest.mark.parametrize(
    "partition_value",
    [
        "year",
        2019.5,
    ],
)
def test_write_curated_parquet_rejects_non_integer_partition_value(
    tmp_path: Path,
    partition_value: object,
) -> None:
    """Partition values must be integer-compatible years."""
    data = _curated_frame()
    data["inspection_year"] = pd.Series(
        [partition_value, 2019],
        dtype="object",
    )

    with pytest.raises(
        CuratedParquetWriteError,
        match=("Partition column must contain integer years: inspection_year"),
    ):
        _write(
            data,
            tmp_path / "curated",
        )


def test_write_curated_parquet_rejects_partition_year_mismatch(
    tmp_path: Path,
) -> None:
    """Partition values must match the incoming batch year."""
    data = _curated_frame(inspection_year=2020)

    with pytest.raises(
        CuratedParquetWriteError,
        match=(
            "Partition values do not match batch year 2019: "
            r"\[2020\]"
        ),
    ):
        _write(
            data,
            tmp_path / "curated",
            batch_year=2019,
        )


@pytest.mark.parametrize(
    "compression",
    [
        "",
        None,
    ],
)
def test_write_curated_parquet_rejects_empty_compression(
    tmp_path: Path,
    compression: object,
) -> None:
    """A Parquet compression codec must be configured."""
    with pytest.raises(
        CuratedParquetWriteError,
        match="Parquet compression must be a non-empty string",
    ):
        write_curated_parquet(
            _curated_frame(),
            output_dir=tmp_path / "curated",
            batch_year=2019,
            output_format="parquet",
            compression=compression,
            partition_by=["inspection_year"],
        )


def test_write_curated_parquet_rejects_unsupported_compression(
    tmp_path: Path,
) -> None:
    """An unknown Arrow codec should fail before filesystem writes."""
    with pytest.raises(
        CuratedParquetWriteError,
        match="Unsupported Parquet compression codec",
    ):
        write_curated_parquet(
            _curated_frame(),
            output_dir=tmp_path / "curated",
            batch_year=2019,
            output_format="parquet",
            compression="definitely-not-a-codec",
            partition_by=["inspection_year"],
        )


def test_write_curated_parquet_rejects_file_output_root(
    tmp_path: Path,
) -> None:
    """The configured curated root must be a directory."""
    output_path = tmp_path / "curated"
    output_path.write_text("not a directory", encoding="utf-8")

    with pytest.raises(
        CuratedParquetWriteError,
        match="Curated output path is not a directory",
    ):
        _write(
            _curated_frame(),
            output_path,
        )


def test_write_curated_parquet_removes_partial_temporary_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed write should not leave a partial temporary file."""

    def fail_write_table(
        _table: object,
        where: str | Path,
        **_kwargs: object,
    ) -> None:
        Path(where).write_bytes(b"partial")
        raise OSError("Simulated disk failure.")

    monkeypatch.setattr(
        curated_writer_module.pq,
        "write_table",
        fail_write_table,
    )

    output_dir = tmp_path / "curated"
    temporary_path = (
        output_dir / "inspection_year=2019" / ".food_inspections_2019.parquet.tmp"
    )

    with pytest.raises(
        CuratedParquetWriteError,
        match="Unable to write curated Parquet file",
    ):
        _write(
            _curated_frame(),
            output_dir,
        )

    assert not temporary_path.exists()


def test_write_curated_parquet_supports_empty_batch(
    tmp_path: Path,
) -> None:
    """An empty accepted batch should still produce a valid Parquet file."""
    data = _curated_frame().iloc[0:0].copy()

    result = _write(
        data,
        tmp_path / "curated",
    )

    assert result.row_count == 0
    assert result.path.is_file()

    reloaded = pd.read_parquet(result.path)

    assert reloaded.shape == (0, 3)
    assert reloaded.columns.tolist() == [
        "inspection_id",
        "dba_name",
        "inspection_year",
    ]
