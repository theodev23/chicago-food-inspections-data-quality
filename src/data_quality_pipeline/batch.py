"""Inspect incoming batch files and derive deterministic metadata."""

import re
from dataclasses import dataclass
from hashlib import new as new_hash
from pathlib import Path

_BATCH_FILENAME_PATTERN = re.compile(r"^food_inspections_(?P<year>[0-9]{4})\.csv$")
_READ_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True, slots=True)
class IncomingBatch:
    """Metadata identifying one incoming data batch."""

    path: Path
    year: int
    size_bytes: int
    checksum: str
    checksum_algorithm: str


class IncomingBatchError(Exception):
    """Raised when an incoming batch cannot be inspected safely."""


def inspect_incoming_batch(
    file_path: str | Path,
    *,
    start_year: int,
    end_year: int,
    hash_algorithm: str = "sha256",
) -> IncomingBatch:
    """Inspect an incoming CSV and return its deterministic metadata.

    Args:
        file_path: Path to the incoming CSV file.
        start_year: Earliest year accepted by the pipeline.
        end_year: Latest year accepted by the pipeline.
        hash_algorithm: Hash algorithm used to identify file contents.

    Returns:
        Immutable metadata describing the incoming batch.

    Raises:
        IncomingBatchError: If the file is missing, unreadable, incorrectly
            named, outside the accepted year range, or uses an unsupported
            hash algorithm.
    """
    path = Path(file_path)

    if not path.is_file():
        raise IncomingBatchError(f"Incoming batch file not found: {path}")

    filename_match = _BATCH_FILENAME_PATTERN.fullmatch(path.name)

    if filename_match is None:
        raise IncomingBatchError(
            "Incoming batch filename does not follow the expected convention: "
            f"{path.name}"
        )

    year = int(filename_match.group("year"))

    if not start_year <= year <= end_year:
        raise IncomingBatchError(
            f"Incoming batch year {year} is outside the accepted range "
            f"{start_year}-{end_year}."
        )

    algorithm = hash_algorithm.strip().lower()

    if not algorithm:
        raise IncomingBatchError("Hash algorithm must not be empty.")

    try:
        size_bytes = path.stat().st_size
        checksum = _calculate_checksum(path, algorithm)
    except ValueError as exc:
        raise IncomingBatchError(f"Unsupported hash algorithm: {algorithm}") from exc
    except OSError as exc:
        raise IncomingBatchError(f"Unable to inspect incoming batch: {path}") from exc

    return IncomingBatch(
        path=path,
        year=year,
        size_bytes=size_bytes,
        checksum=checksum,
        checksum_algorithm=algorithm,
    )


def _calculate_checksum(path: Path, algorithm: str) -> str:
    """Calculate a file checksum without loading the whole file into memory."""
    digest = new_hash(algorithm)

    with path.open("rb") as batch_file:
        while chunk := batch_file.read(_READ_CHUNK_SIZE):
            digest.update(chunk)

    return digest.hexdigest()
