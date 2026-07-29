"""Unit tests for quarantine Parquet persistence."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest
from pandas.testing import assert_frame_equal

import data_quality_pipeline.quarantine_writer as quarantine_writer_module
from data_quality_pipeline.quarantine_writer import (
    QuarantineParquetWriteError,
    QuarantineParquetWriteResult,
    write_quarantine_parquet,
)


def _quarantine_frame(
    inspection_ids: list[str] | None = None,
    *,
    batch_year: int = 2019,
    severities: list[str] | None = None,
) -> pd.DataFrame:
    """Build a small issue-level quarantine frame."""
    identifiers = inspection_ids or ["10", "20"]
    issue_severities = severities or ["error"] * len(identifiers)

    return pd.DataFrame(
        {
            "inspection_id": pd.Series(
                identifiers,
                dtype="string",
            ),
            "dba_name": pd.Series(
                [f"RESTAURANT {identifier}" for identifier in identifiers],
                dtype="string",
            ),
            "dq_source_row_number": pd.Series(
                range(2, 2 + len(identifiers)),
                dtype="int64",
            ),
            "dq_rule_id": pd.Series(
                ["invalid_record"] * len(identifiers),
                dtype="string",
            ),
            "dq_column": pd.Series(
                ["results"] * len(identifiers),
                dtype="string",
            ),
            "dq_value": pd.Series(
                ["UNKNOWN"] * len(identifiers),
                dtype="string",
            ),
            "dq_message": pd.Series(
                ["Record is invalid."] * len(identifiers),
                dtype="string",
            ),
            "dq_severity": pd.Series(
                issue_severities,
                dtype="string",
            ),
            "dq_duplicate_of_inspection_id": pd.Series(
                [pd.NA] * len(identifiers),
                dtype="Int64",
            ),
            "dq_batch_year": pd.Series(
                [batch_year] * len(identifiers),
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
    compression: object = "snappy",
) -> QuarantineParquetWriteResult:
    """Write one test frame with the standard output configuration."""
    return write_quarantine_parquet(
        data,
        output_dir=output_dir,
        batch_year=batch_year,
        output_format=output_format,
        compression=compression,
    )


def test_quarantine_parquet_write_result_is_immutable() -> None:
    """Written-file metadata must not change after construction."""
    result = QuarantineParquetWriteResult(
        path=Path("example.parquet"),
        row_count=1,
        size_bytes=100,
        partition_column="dq_batch_year",
        partition_value=2019,
        compression="snappy",
    )

    with pytest.raises(FrozenInstanceError):
        result.row_count = 2


def test_write_quarantine_parquet_writes_expected_partitioned_file(
    tmp_path: Path,
) -> None:
    """The writer should create a readable Snappy-compressed file."""
    data = _quarantine_frame()

    result = _write(
        data,
        tmp_path / "quarantine",
    )

    expected_path = (
        tmp_path
        / "quarantine"
        / "dq_batch_year=2019"
        / "food_inspections_2019_quarantine.parquet"
    )

    assert result.path == expected_path
    assert result.path.is_file()
    assert result.row_count == 2
    assert result.size_bytes == result.path.stat().st_size
    assert result.size_bytes > 0
    assert result.partition_column == "dq_batch_year"
    assert result.partition_value == 2019
    assert result.compression == "snappy"

    parquet_file = pq.ParquetFile(result.path)
    metadata = parquet_file.metadata

    assert metadata.num_rows == 2
    assert metadata.num_columns == 10
    assert metadata.row_group(0).column(0).compression == "SNAPPY"

    reloaded = pd.read_parquet(result.path)

    assert reloaded["inspection_id"].tolist() == ["10", "20"]
    assert reloaded["dq_batch_year"].tolist() == [2019, 2019]
    assert reloaded["dq_severity"].tolist() == ["error", "error"]


def test_write_quarantine_parquet_replaces_existing_batch_output(
    tmp_path: Path,
) -> None:
    """A rerun for one year should replace its previous output."""
    output_dir = tmp_path / "quarantine"

    first_result = _write(
        _quarantine_frame(["10", "20"]),
        output_dir,
    )
    second_result = _write(
        _quarantine_frame(["30"]),
        output_dir,
    )

    assert second_result.path == first_result.path
    assert second_result.row_count == 1

    reloaded = pd.read_parquet(second_result.path)

    assert reloaded["inspection_id"].tolist() == ["30"]
    assert not any(
        path.name.endswith(".tmp") for path in second_result.path.parent.iterdir()
    )


def test_write_quarantine_parquet_does_not_modify_input(
    tmp_path: Path,
) -> None:
    """Persistence must not mutate the quarantine DataFrame."""
    data = _quarantine_frame()
    original = data.copy(deep=True)

    _write(
        data,
        tmp_path / "quarantine",
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
def test_write_quarantine_parquet_rejects_invalid_batch_year(
    tmp_path: Path,
    batch_year: object,
) -> None:
    """The partition year must be a real integer in the valid range."""
    with pytest.raises(
        QuarantineParquetWriteError,
        match="Batch year must be an integer between 1 and 9999",
    ):
        _write(
            _quarantine_frame(),
            tmp_path / "quarantine",
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
def test_write_quarantine_parquet_rejects_invalid_output_format(
    tmp_path: Path,
    output_format: object,
) -> None:
    """The writer should accept only the implemented Parquet format."""
    with pytest.raises(
        QuarantineParquetWriteError,
        match="Quarantine output format must be 'parquet'",
    ):
        _write(
            _quarantine_frame(),
            tmp_path / "quarantine",
            output_format=output_format,
        )


def test_write_quarantine_parquet_rejects_duplicate_columns(
    tmp_path: Path,
) -> None:
    """Duplicate quarantine column names should be rejected."""
    data = _quarantine_frame()
    columns = data.columns.tolist()
    columns[1] = "inspection_id"
    data.columns = columns

    with pytest.raises(
        QuarantineParquetWriteError,
        match="Quarantine data contains duplicate column names",
    ):
        _write(
            data,
            tmp_path / "quarantine",
        )


def test_write_quarantine_parquet_requires_metadata_columns(
    tmp_path: Path,
) -> None:
    """The issue-level diagnostic schema must be complete."""
    data = _quarantine_frame().drop(columns=["dq_message", "dq_severity"])

    with pytest.raises(
        QuarantineParquetWriteError,
        match="Quarantine data is missing metadata columns",
    ):
        _write(
            data,
            tmp_path / "quarantine",
        )


def test_write_quarantine_parquet_rejects_null_partition_value(
    tmp_path: Path,
) -> None:
    """Every quarantined issue must have a batch-year partition."""
    data = _quarantine_frame()
    data["dq_batch_year"] = pd.Series(
        [pd.NA, 2019],
        dtype="Int64",
    )

    with pytest.raises(
        QuarantineParquetWriteError,
        match=("Quarantine partition column contains null values: dq_batch_year"),
    ):
        _write(
            data,
            tmp_path / "quarantine",
        )


@pytest.mark.parametrize(
    "partition_value",
    [
        "year",
        2019.5,
    ],
)
def test_write_quarantine_parquet_rejects_non_integer_partition_value(
    tmp_path: Path,
    partition_value: object,
) -> None:
    """Partition values must be integer-compatible years."""
    data = _quarantine_frame()
    data["dq_batch_year"] = pd.Series(
        [partition_value, 2019],
        dtype="object",
    )

    with pytest.raises(
        QuarantineParquetWriteError,
        match=("Quarantine partition column must contain integer years: dq_batch_year"),
    ):
        _write(
            data,
            tmp_path / "quarantine",
        )


def test_write_quarantine_parquet_rejects_partition_year_mismatch(
    tmp_path: Path,
) -> None:
    """Partition values must match the incoming batch year."""
    data = _quarantine_frame(batch_year=2020)

    with pytest.raises(
        QuarantineParquetWriteError,
        match=(
            "Quarantine partition values do not match batch year "
            r"2019: \[2020\]"
        ),
    ):
        _write(
            data,
            tmp_path / "quarantine",
            batch_year=2019,
        )


def test_write_quarantine_parquet_rejects_null_severity(
    tmp_path: Path,
) -> None:
    """Every quarantined issue must have an explicit severity."""
    data = _quarantine_frame()
    data.loc[0, "dq_severity"] = pd.NA

    with pytest.raises(
        QuarantineParquetWriteError,
        match="Quarantine severity contains null values",
    ):
        _write(
            data,
            tmp_path / "quarantine",
        )


@pytest.mark.parametrize(
    "severities",
    [
        ["warning", "warning"],
        ["error", "warning"],
    ],
)
def test_write_quarantine_parquet_rejects_non_error_severity(
    tmp_path: Path,
    severities: list[str],
) -> None:
    """Warnings must never be persisted as quarantined records."""
    data = _quarantine_frame(severities=severities)

    with pytest.raises(
        QuarantineParquetWriteError,
        match="Quarantine data must contain only error severity",
    ):
        _write(
            data,
            tmp_path / "quarantine",
        )


@pytest.mark.parametrize(
    "compression",
    [
        "",
        None,
    ],
)
def test_write_quarantine_parquet_rejects_empty_compression(
    tmp_path: Path,
    compression: object,
) -> None:
    """A Parquet compression codec must be configured."""
    with pytest.raises(
        QuarantineParquetWriteError,
        match="Parquet compression must be a non-empty string",
    ):
        _write(
            _quarantine_frame(),
            tmp_path / "quarantine",
            compression=compression,
        )


def test_write_quarantine_parquet_rejects_unsupported_compression(
    tmp_path: Path,
) -> None:
    """An unknown Arrow codec should fail before filesystem writes."""
    with pytest.raises(
        QuarantineParquetWriteError,
        match="Unsupported Parquet compression codec",
    ):
        _write(
            _quarantine_frame(),
            tmp_path / "quarantine",
            compression="definitely-not-a-codec",
        )


def test_write_quarantine_parquet_rejects_file_output_root(
    tmp_path: Path,
) -> None:
    """The configured quarantine root must be a directory."""
    output_path = tmp_path / "quarantine"
    output_path.write_text("not a directory", encoding="utf-8")

    with pytest.raises(
        QuarantineParquetWriteError,
        match="Quarantine output path is not a directory",
    ):
        _write(
            _quarantine_frame(),
            output_path,
        )


def test_write_quarantine_parquet_removes_partial_temporary_file(
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
        quarantine_writer_module.pq,
        "write_table",
        fail_write_table,
    )

    output_dir = tmp_path / "quarantine"
    temporary_path = (
        output_dir
        / "dq_batch_year=2019"
        / ".food_inspections_2019_quarantine.parquet.tmp"
    )

    with pytest.raises(
        QuarantineParquetWriteError,
        match="Unable to write quarantine Parquet file",
    ):
        _write(
            _quarantine_frame(),
            output_dir,
        )

    assert not temporary_path.exists()


def test_write_quarantine_parquet_supports_empty_batch(
    tmp_path: Path,
) -> None:
    """A clean batch should still produce a valid empty Parquet file."""
    data = _quarantine_frame().iloc[0:0].copy()

    result = _write(
        data,
        tmp_path / "quarantine",
    )

    assert result.row_count == 0
    assert result.path.is_file()

    reloaded = pd.read_parquet(result.path)

    assert reloaded.shape == (0, 10)
    assert reloaded.columns.tolist() == data.columns.tolist()
