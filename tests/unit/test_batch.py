"""Unit tests for incoming batch metadata inspection."""

from dataclasses import FrozenInstanceError
from hashlib import sha256
from pathlib import Path

import pytest

from data_quality_pipeline.batch import (
    IncomingBatchError,
    inspect_incoming_batch,
)


def _write_batch(
    tmp_path: Path,
    *,
    filename: str = "food_inspections_2019.csv",
    content: bytes = b"inspection_id,dba_name\n1,Example\n",
) -> Path:
    """Write a temporary incoming batch."""
    batch_path = tmp_path / filename
    batch_path.write_bytes(content)
    return batch_path


def test_inspect_incoming_batch_returns_expected_metadata(
    tmp_path: Path,
) -> None:
    """A valid incoming file should produce deterministic metadata."""
    content = b"inspection_id,dba_name\n1,Example\n"
    batch_path = _write_batch(
        tmp_path,
        content=content,
    )

    batch = inspect_incoming_batch(
        batch_path,
        start_year=2019,
        end_year=2025,
        hash_algorithm=" SHA256 ",
    )

    assert batch.path == batch_path
    assert batch.year == 2019
    assert batch.size_bytes == len(content)
    assert batch.checksum == sha256(content).hexdigest()
    assert batch.checksum_algorithm == "sha256"


def test_incoming_batch_metadata_is_immutable(
    tmp_path: Path,
) -> None:
    """Batch identity fields must not change after inspection."""
    batch_path = _write_batch(tmp_path)

    batch = inspect_incoming_batch(
        batch_path,
        start_year=2019,
        end_year=2025,
    )

    with pytest.raises(FrozenInstanceError):
        batch.year = 2020


def test_inspect_incoming_batch_rejects_missing_file(
    tmp_path: Path,
) -> None:
    """A missing incoming file should raise a clear error."""
    missing_path = tmp_path / "food_inspections_2019.csv"

    with pytest.raises(
        IncomingBatchError,
        match="Incoming batch file not found",
    ):
        inspect_incoming_batch(
            missing_path,
            start_year=2019,
            end_year=2025,
        )


def test_inspect_incoming_batch_rejects_invalid_filename(
    tmp_path: Path,
) -> None:
    """The filename must follow the documented convention."""
    batch_path = _write_batch(
        tmp_path,
        filename="chicago_2019.csv",
    )

    with pytest.raises(
        IncomingBatchError,
        match="does not follow the expected convention",
    ):
        inspect_incoming_batch(
            batch_path,
            start_year=2019,
            end_year=2025,
        )


def test_inspect_incoming_batch_rejects_year_outside_range(
    tmp_path: Path,
) -> None:
    """The filename year must be inside the configured range."""
    batch_path = _write_batch(
        tmp_path,
        filename="food_inspections_2018.csv",
    )

    with pytest.raises(
        IncomingBatchError,
        match="outside the accepted range 2019-2025",
    ):
        inspect_incoming_batch(
            batch_path,
            start_year=2019,
            end_year=2025,
        )


def test_inspect_incoming_batch_rejects_empty_hash_algorithm(
    tmp_path: Path,
) -> None:
    """A blank hash algorithm should be rejected."""
    batch_path = _write_batch(tmp_path)

    with pytest.raises(
        IncomingBatchError,
        match="Hash algorithm must not be empty",
    ):
        inspect_incoming_batch(
            batch_path,
            start_year=2019,
            end_year=2025,
            hash_algorithm="   ",
        )


def test_inspect_incoming_batch_rejects_unknown_hash_algorithm(
    tmp_path: Path,
) -> None:
    """An algorithm unsupported by hashlib should be rejected."""
    batch_path = _write_batch(tmp_path)

    with pytest.raises(
        IncomingBatchError,
        match="Unsupported hash algorithm",
    ):
        inspect_incoming_batch(
            batch_path,
            start_year=2019,
            end_year=2025,
            hash_algorithm="unknown-hash",
        )
