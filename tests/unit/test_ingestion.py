"""Unit tests for raw CSV batch reading."""

from pathlib import Path

import pytest

from data_quality_pipeline.batch import inspect_incoming_batch
from data_quality_pipeline.ingestion import (
    RawBatchReadError,
    read_raw_batch,
)

_EXPECTED_COLUMNS = [
    "inspection_id",
    "violations",
]


def _write_batch(
    tmp_path: Path,
    content: bytes,
) -> Path:
    """Write and inspect a temporary incoming CSV batch."""
    batch_path = tmp_path / "food_inspections_2019.csv"
    batch_path.write_bytes(content)
    return batch_path


def _inspect_batch(batch_path: Path):
    """Return metadata for a temporary batch."""
    return inspect_incoming_batch(
        batch_path,
        start_year=2019,
        end_year=2025,
    )


def test_read_raw_batch_preserves_source_strings(
    tmp_path: Path,
) -> None:
    """A valid CSV should retain strings and blank source values."""
    batch_path = _write_batch(
        tmp_path,
        b"inspection_id,violations\n1,\n2,Example violation\n",
    )

    data = read_raw_batch(
        _inspect_batch(batch_path),
        expected_columns=_EXPECTED_COLUMNS,
    )

    assert data.columns.tolist() == _EXPECTED_COLUMNS
    assert len(data) == 2
    assert all(str(dtype) == "string" for dtype in data.dtypes)
    assert data.loc[0, "inspection_id"] == "1"
    assert data.loc[0, "violations"] == ""
    assert data.loc[1, "violations"] == "Example violation"


def test_read_raw_batch_rejects_file_without_header(
    tmp_path: Path,
) -> None:
    """A zero-byte CSV should be rejected."""
    batch_path = _write_batch(tmp_path, b"")

    with pytest.raises(
        RawBatchReadError,
        match="does not contain a header",
    ):
        read_raw_batch(
            _inspect_batch(batch_path),
            expected_columns=_EXPECTED_COLUMNS,
        )


def test_read_raw_batch_rejects_malformed_csv(
    tmp_path: Path,
) -> None:
    """Invalid CSV quoting should raise a domain-specific error."""
    batch_path = _write_batch(
        tmp_path,
        b'inspection_id,violations\n1,"unclosed value\n',
    )

    with pytest.raises(
        RawBatchReadError,
        match="Incoming CSV is malformed",
    ):
        read_raw_batch(
            _inspect_batch(batch_path),
            expected_columns=_EXPECTED_COLUMNS,
        )


def test_read_raw_batch_rejects_incompatible_encoding(
    tmp_path: Path,
) -> None:
    """Bytes invalid under the configured encoding should be rejected."""
    batch_path = _write_batch(
        tmp_path,
        b"inspection_id,violations\n1,\xff\n",
    )

    with pytest.raises(
        RawBatchReadError,
        match="Unable to decode incoming CSV with utf-8",
    ):
        read_raw_batch(
            _inspect_batch(batch_path),
            expected_columns=_EXPECTED_COLUMNS,
            encoding="utf-8",
        )


def test_read_raw_batch_rejects_schema_mismatch(
    tmp_path: Path,
) -> None:
    """Missing and unexpected columns should be reported."""
    batch_path = _write_batch(
        tmp_path,
        b"inspection_id,results\n1,Pass\n",
    )

    with pytest.raises(RawBatchReadError) as error:
        read_raw_batch(
            _inspect_batch(batch_path),
            expected_columns=_EXPECTED_COLUMNS,
        )

    message = str(error.value)

    assert "Missing columns: ['violations']" in message
    assert "Unexpected columns: ['results']" in message


def test_read_raw_batch_rejects_incorrect_column_order(
    tmp_path: Path,
) -> None:
    """The source column order must match the contract."""
    batch_path = _write_batch(
        tmp_path,
        b"violations,inspection_id\nExample,1\n",
    )

    with pytest.raises(RawBatchReadError) as error:
        read_raw_batch(
            _inspect_batch(batch_path),
            expected_columns=_EXPECTED_COLUMNS,
        )

    message = str(error.value)

    assert "Missing columns: []" in message
    assert "Unexpected columns: []" in message
    assert "Expected order" in message
    assert "Actual order" in message


def test_read_raw_batch_rejects_header_only_file(
    tmp_path: Path,
) -> None:
    """A CSV with a header but no records should be rejected."""
    batch_path = _write_batch(
        tmp_path,
        b"inspection_id,violations\n",
    )

    with pytest.raises(
        RawBatchReadError,
        match="does not contain any data rows",
    ):
        read_raw_batch(
            _inspect_batch(batch_path),
            expected_columns=_EXPECTED_COLUMNS,
        )
