"""Unit tests for incoming batch metadata inspection."""

from dataclasses import FrozenInstanceError
from hashlib import sha256
from pathlib import Path

import pytest

import data_quality_pipeline.batch as batch_module
from data_quality_pipeline.batch import (
    IncomingBatchError,
    discover_incoming_batch_paths,
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


def test_discover_incoming_batch_paths_returns_year_sorted_files(
    tmp_path: Path,
) -> None:
    """Matching annual files should be returned in deterministic year order."""
    expected_paths = [
        tmp_path / "food_inspections_2019.csv",
        tmp_path / "food_inspections_2020.csv",
        tmp_path / "food_inspections_2021.csv",
    ]

    for path in reversed(expected_paths):
        path.write_text("inspection_id\n", encoding="utf-8")

    (tmp_path / "notes.txt").write_text(
        "ignored",
        encoding="utf-8",
    )
    (tmp_path / ".gitkeep").touch()
    (tmp_path / "food_inspections_2022.csv").mkdir()

    discovered = discover_incoming_batch_paths(
        tmp_path,
        file_pattern="food_inspections_*.csv",
        start_year=2019,
        end_year=2025,
    )

    assert discovered == tuple(expected_paths)


def test_discover_incoming_batch_paths_returns_empty_tuple(
    tmp_path: Path,
) -> None:
    """A readable directory without matching batches should be valid."""
    (tmp_path / ".gitkeep").touch()
    (tmp_path / "notes.txt").write_text(
        "ignored",
        encoding="utf-8",
    )

    discovered = discover_incoming_batch_paths(
        tmp_path,
        file_pattern="food_inspections_*.csv",
        start_year=2019,
        end_year=2025,
    )

    assert discovered == ()


def test_discover_incoming_batch_paths_does_not_calculate_checksums(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Discovery should inspect filenames without reading file contents."""
    batch_path = _write_batch(tmp_path)

    def fail_checksum(*_args: object) -> str:
        raise AssertionError("Discovery must not calculate file checksums.")

    monkeypatch.setattr(
        batch_module,
        "_calculate_checksum",
        fail_checksum,
    )

    discovered = discover_incoming_batch_paths(
        tmp_path,
        file_pattern="food_inspections_*.csv",
        start_year=2019,
        end_year=2025,
    )

    assert discovered == (batch_path,)


def test_discover_incoming_batch_paths_rejects_missing_directory(
    tmp_path: Path,
) -> None:
    """A missing incoming directory should raise a clear error."""
    missing_path = tmp_path / "missing"

    with pytest.raises(
        IncomingBatchError,
        match="Incoming batch directory not found",
    ):
        discover_incoming_batch_paths(
            missing_path,
            file_pattern="food_inspections_*.csv",
            start_year=2019,
            end_year=2025,
        )


def test_discover_incoming_batch_paths_rejects_file_input_path(
    tmp_path: Path,
) -> None:
    """The configured incoming root must be a directory."""
    file_path = tmp_path / "incoming"
    file_path.write_text("not a directory", encoding="utf-8")

    with pytest.raises(
        IncomingBatchError,
        match="Incoming batch path is not a directory",
    ):
        discover_incoming_batch_paths(
            file_path,
            file_pattern="food_inspections_*.csv",
            start_year=2019,
            end_year=2025,
        )


@pytest.mark.parametrize(
    "file_pattern",
    [
        None,
        "",
        "   ",
        "nested/*.csv",
        r"nested\*.csv",
        ".",
        "..",
    ],
)
def test_discover_incoming_batch_paths_rejects_invalid_pattern(
    tmp_path: Path,
    file_pattern: object,
) -> None:
    """Discovery patterns must be non-empty filename-only strings."""
    with pytest.raises(IncomingBatchError):
        discover_incoming_batch_paths(
            tmp_path,
            file_pattern=file_pattern,
            start_year=2019,
            end_year=2025,
        )


def test_discover_incoming_batch_paths_rejects_matching_invalid_filename(
    tmp_path: Path,
) -> None:
    """Every matching file must follow the annual naming convention."""
    (tmp_path / "unexpected.csv").write_text(
        "inspection_id\n",
        encoding="utf-8",
    )

    with pytest.raises(
        IncomingBatchError,
        match="does not follow the expected convention",
    ):
        discover_incoming_batch_paths(
            tmp_path,
            file_pattern="*.csv",
            start_year=2019,
            end_year=2025,
        )


def test_discover_incoming_batch_paths_rejects_out_of_range_year(
    tmp_path: Path,
) -> None:
    """Matching annual files must fall inside the configured year range."""
    (tmp_path / "food_inspections_2018.csv").write_text(
        "inspection_id\n",
        encoding="utf-8",
    )

    with pytest.raises(
        IncomingBatchError,
        match="outside the accepted range 2019-2025",
    ):
        discover_incoming_batch_paths(
            tmp_path,
            file_pattern="food_inspections_*.csv",
            start_year=2019,
            end_year=2025,
        )
