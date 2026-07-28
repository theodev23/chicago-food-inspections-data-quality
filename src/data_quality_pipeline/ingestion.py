"""Read incoming CSV batches without applying business transformations."""

from collections.abc import Sequence

import pandas as pd

from data_quality_pipeline.batch import IncomingBatch


class RawBatchReadError(Exception):
    """Raised when an incoming CSV cannot be read or validated."""


def read_raw_batch(
    batch: IncomingBatch,
    *,
    expected_columns: Sequence[str],
    encoding: str = "utf-8",
    delimiter: str = ",",
) -> pd.DataFrame:
    """Read and validate an incoming CSV batch.

    Args:
        batch: Previously inspected incoming batch metadata.
        expected_columns: Ordered source columns required by the contract.
        encoding: Character encoding used by the CSV.
        delimiter: Field delimiter used by the CSV.

    Returns:
        A DataFrame preserving source values as nullable string columns.

    Raises:
        RawBatchReadError: If the CSV is unreadable, malformed, empty,
            or does not match the expected source schema.
    """
    expected = list(expected_columns)

    try:
        data = pd.read_csv(
            batch.path,
            dtype="string",
            keep_default_na=False,
            encoding=encoding,
            sep=delimiter,
        )
    except pd.errors.EmptyDataError as exc:
        raise RawBatchReadError(
            f"Incoming CSV does not contain a header: {batch.path}"
        ) from exc
    except pd.errors.ParserError as exc:
        raise RawBatchReadError(f"Incoming CSV is malformed: {batch.path}") from exc
    except UnicodeError as exc:
        raise RawBatchReadError(
            f"Unable to decode incoming CSV with {encoding}: {batch.path}"
        ) from exc
    except OSError as exc:
        raise RawBatchReadError(f"Unable to read incoming CSV: {batch.path}") from exc

    actual = data.columns.tolist()

    if actual != expected:
        missing = [column for column in expected if column not in actual]
        unexpected = [column for column in actual if column not in expected]

        raise RawBatchReadError(
            "Incoming CSV columns do not match the data contract. "
            f"Missing columns: {missing}. "
            f"Unexpected columns: {unexpected}. "
            f"Expected order: {expected}. "
            f"Actual order: {actual}."
        )

    if data.empty:
        raise RawBatchReadError(
            f"Incoming CSV does not contain any data rows: {batch.path}"
        )

    return data
