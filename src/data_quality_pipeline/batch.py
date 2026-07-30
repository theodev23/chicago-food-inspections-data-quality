"""Discover and inspect incoming annual batch files."""

import re
from dataclasses import dataclass
from fnmatch import fnmatchcase
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
    """Raised when incoming batch discovery or inspection fails."""


def discover_incoming_batch_paths(
    input_dir: str | Path,
    *,
    file_pattern: str,
    start_year: int,
    end_year: int,
) -> tuple[Path, ...]:
    """Discover annual incoming CSV paths in deterministic year order.

    Discovery is intentionally lightweight: file contents and checksums
    are not read. Every matching filename must follow the annual naming
    convention and fall inside the configured year range.

    Args:
        input_dir: Directory containing incoming annual CSV files.
        file_pattern: Filename-only glob pattern used for discovery.
        start_year: Earliest year accepted by the pipeline.
        end_year: Latest year accepted by the pipeline.

    Returns:
        Immutable paths sorted by annual batch year and filename.

    Raises:
        IncomingBatchError: If the directory or discovery pattern is
            invalid, the directory cannot be read, or a matching file
            violates the annual batch naming and year rules.
    """
    root_path = Path(input_dir)

    if not root_path.exists():
        raise IncomingBatchError(f"Incoming batch directory not found: {root_path}")

    if not root_path.is_dir():
        raise IncomingBatchError(f"Incoming batch path is not a directory: {root_path}")

    pattern = _normalize_file_pattern(file_pattern)

    try:
        directory_entries = tuple(root_path.iterdir())
    except OSError as exc:
        raise IncomingBatchError(
            f"Unable to read incoming batch directory: {root_path}"
        ) from exc

    discovered: list[tuple[int, Path]] = []

    for path in directory_entries:
        try:
            is_file = path.is_file()
        except OSError as exc:
            raise IncomingBatchError(
                f"Unable to inspect incoming directory entry: {path}"
            ) from exc

        if not is_file or not fnmatchcase(path.name, pattern):
            continue

        year = _parse_batch_year(
            path.name,
            start_year=start_year,
            end_year=end_year,
        )
        discovered.append((year, path))

    discovered.sort(
        key=lambda item: (
            item[0],
            item[1].name,
        )
    )

    return tuple(path for _, path in discovered)


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
        IncomingBatchError: If the file is missing, unreadable,
            incorrectly named, outside the accepted year range, or uses
            an unsupported hash algorithm.
    """
    path = Path(file_path)

    if not path.is_file():
        raise IncomingBatchError(f"Incoming batch file not found: {path}")

    year = _parse_batch_year(
        path.name,
        start_year=start_year,
        end_year=end_year,
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


def _normalize_file_pattern(file_pattern: str) -> str:
    """Validate and normalize a filename-only discovery pattern."""
    if not isinstance(file_pattern, str):
        raise IncomingBatchError("Incoming file pattern must be a string.")

    pattern = file_pattern.strip()

    if not pattern:
        raise IncomingBatchError("Incoming file pattern must not be empty.")

    if "/" in pattern or "\\" in pattern or pattern in {".", ".."}:
        raise IncomingBatchError(
            "Incoming file pattern must contain a filename pattern "
            "without directory components."
        )

    return pattern


def _parse_batch_year(
    filename: str,
    *,
    start_year: int,
    end_year: int,
) -> int:
    """Extract and validate the year encoded in an annual filename."""
    filename_match = _BATCH_FILENAME_PATTERN.fullmatch(filename)

    if filename_match is None:
        raise IncomingBatchError(
            "Incoming batch filename does not follow the expected "
            f"convention: {filename}"
        )

    year = int(filename_match.group("year"))

    if not start_year <= year <= end_year:
        raise IncomingBatchError(
            f"Incoming batch year {year} is outside the accepted range "
            f"{start_year}-{end_year}."
        )

    return year


def _calculate_checksum(path: Path, algorithm: str) -> str:
    """Calculate a file checksum without loading it fully into memory."""
    digest = new_hash(algorithm)

    with path.open("rb") as batch_file:
        while chunk := batch_file.read(_READ_CHUNK_SIZE):
            digest.update(chunk)

    return digest.hexdigest()
