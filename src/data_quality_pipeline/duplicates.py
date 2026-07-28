"""Detect exact duplicate source records deterministically."""

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class ExactDuplicateRecord:
    """Identify one duplicate and the source record retained for it."""

    inspection_id: int
    duplicate_of_inspection_id: int


class DuplicateDetectionError(Exception):
    """Raised when exact duplicate detection cannot be performed safely."""


def find_exact_duplicates(
    data: pd.DataFrame,
    *,
    primary_key: str,
    source_columns: Sequence[str],
) -> tuple[ExactDuplicateRecord, ...]:
    """Find records identical on every source column except the primary key.

    The lowest numeric primary key is retained in each exact duplicate group.

    Args:
        data: Raw source records.
        primary_key: Numeric source column identifying each record.
        source_columns: Ordered columns defined by the source contract.

    Returns:
        Immutable duplicate mappings sorted by duplicate primary key.

    Raises:
        DuplicateDetectionError: If the schema or primary-key values prevent
            deterministic duplicate detection.
    """
    columns = list(source_columns)

    if not columns:
        raise DuplicateDetectionError("Source columns must not be empty.")

    if len(columns) != len(set(columns)):
        raise DuplicateDetectionError("Source columns contain duplicate names.")

    if primary_key not in columns:
        raise DuplicateDetectionError(
            f"Primary key is not present in source columns: {primary_key}"
        )

    missing_columns = [column for column in columns if column not in data.columns]

    if missing_columns:
        raise DuplicateDetectionError(
            f"Data is missing required source columns: {missing_columns}"
        )

    key_values = data[primary_key].astype("string")
    valid_key_format = key_values.str.fullmatch(r"[0-9]+").fillna(False)

    if not valid_key_format.all():
        raise DuplicateDetectionError(
            f"Primary key contains non-numeric values: {primary_key}"
        )

    try:
        numeric_keys = pd.to_numeric(
            key_values,
            errors="raise",
        ).astype("int64")
    except (TypeError, ValueError, OverflowError) as exc:
        raise DuplicateDetectionError(
            f"Primary key cannot be represented as int64: {primary_key}"
        ) from exc

    if numeric_keys.duplicated().any():
        raise DuplicateDetectionError(
            f"Primary key contains duplicate values: {primary_key}"
        )

    comparison_columns = [column for column in columns if column != primary_key]

    if not comparison_columns:
        raise DuplicateDetectionError(
            "At least one non-primary-key source column is required."
        )

    duplicate_group_mask = data.duplicated(
        subset=comparison_columns,
        keep=False,
    )

    if not duplicate_group_mask.any():
        return ()

    candidates = data.loc[
        duplicate_group_mask,
        comparison_columns,
    ].copy()

    candidates["_primary_key"] = numeric_keys.loc[duplicate_group_mask]

    retained_keys = candidates.groupby(
        comparison_columns,
        dropna=False,
        sort=False,
    )["_primary_key"].transform("min")

    duplicate_candidates = candidates[candidates["_primary_key"] != retained_keys]

    duplicates = [
        ExactDuplicateRecord(
            inspection_id=int(inspection_id),
            duplicate_of_inspection_id=int(retained_id),
        )
        for inspection_id, retained_id in zip(
            duplicate_candidates["_primary_key"],
            retained_keys.loc[duplicate_candidates.index],
            strict=True,
        )
    ]

    return tuple(
        sorted(
            duplicates,
            key=lambda duplicate: duplicate.inspection_id,
        )
    )
