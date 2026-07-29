"""Transform validated source records into the curated schema."""

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
import pyarrow as pa


class CuratedTransformationError(Exception):
    """Raised when validated records cannot be transformed safely."""


def transform_curated_records(
    data: pd.DataFrame,
    *,
    batch_year: int,
    contract: Mapping[str, Any],
) -> pd.DataFrame:
    """Transform accepted source records according to the data contract.

    The input frame is not modified. Source columns marked with
    ``publish: false`` are omitted, target names and target types come from
    the contract, and ``inspection_year`` is added for Parquet partitioning.

    Args:
        data: Accepted source records after error rows have been removed.
        batch_year: Year encoded in the incoming batch filename.
        contract: Validated data contract.

    Returns:
        A new curated frame with deterministic column order and dtypes.

    Raises:
        CuratedTransformationError: If the source schema, contract settings,
            or source values prevent a safe transformation.
    """
    if (
        isinstance(batch_year, bool)
        or not isinstance(batch_year, int)
        or not 1 <= batch_year <= 9999
    ):
        raise CuratedTransformationError(
            "Batch year must be an integer between 1 and 9999."
        )

    expected_columns = list(contract["source_schema"]["expected_columns"])

    if list(data.columns) != expected_columns:
        raise CuratedTransformationError(
            "Source columns do not match the contract order."
        )

    column_contracts = contract["columns"]
    transformed_columns: dict[str, pd.Series] = {}
    target_names: set[str] = set()

    for source_name in expected_columns:
        column_contract = column_contracts[source_name]

        if column_contract.get("publish", True) is False:
            continue

        target_name = column_contract["target_name"]

        if target_name == "inspection_year":
            raise CuratedTransformationError(
                "Contract target names must not use inspection_year."
            )

        if target_name in target_names:
            raise CuratedTransformationError(
                f"Duplicate curated target column: {target_name}"
            )

        target_names.add(target_name)
        transformed_columns[target_name] = _transform_column(
            data[source_name],
            source_name=source_name,
            column_contract=column_contract,
        )

    curated = pd.DataFrame(transformed_columns).reset_index(drop=True)
    curated["inspection_year"] = pd.Series(
        batch_year,
        index=curated.index,
        dtype="int16",
    )

    return curated


def _transform_column(
    source: pd.Series,
    *,
    source_name: str,
    column_contract: Mapping[str, Any],
) -> pd.Series:
    """Transform one source column into its declared target type."""
    target_type = column_contract["target_type"]
    nullable = column_contract["nullable"]
    cleaning = column_contract.get("cleaning")

    if target_type == "string":
        return _transform_string(
            source,
            source_name=source_name,
            nullable=nullable,
            cleaning=cleaning,
        )

    if target_type == "int64":
        return _transform_integer(
            source,
            source_name=source_name,
            nullable=nullable,
            target_dtype="int64",
            zero_as_null=False,
        )

    if target_type == "Int64":
        zero_as_null = False

        if cleaning is not None:
            if not isinstance(cleaning, Mapping):
                raise CuratedTransformationError(
                    f"Numeric cleaning must be a mapping: {source_name}"
                )

            zero_as_null = cleaning.get("zero_as_null", False)

            if not isinstance(zero_as_null, bool):
                raise CuratedTransformationError(
                    f"zero_as_null must be boolean: {source_name}"
                )

        return _transform_integer(
            source,
            source_name=source_name,
            nullable=nullable,
            target_dtype="Int64",
            zero_as_null=zero_as_null,
        )

    if target_type == "float64":
        return _transform_float(
            source,
            source_name=source_name,
            nullable=nullable,
        )

    if target_type == "date":
        return _transform_date(
            source,
            source_name=source_name,
            nullable=nullable,
            accepted_formats=column_contract["accepted_formats"],
        )

    raise CuratedTransformationError(
        f"Unsupported target type for {source_name}: {target_type}"
    )


def _transform_string(
    source: pd.Series,
    *,
    source_name: str,
    nullable: bool,
    cleaning: object,
) -> pd.Series:
    """Apply contractual string cleaning and normalize blank values."""
    values = source.astype("string").copy()

    if cleaning is None:
        operations: Sequence[str] = ()
    elif isinstance(cleaning, Sequence) and not isinstance(
        cleaning,
        (str, bytes),
    ):
        operations = cleaning
    else:
        raise CuratedTransformationError(
            f"String cleaning must be a sequence: {source_name}"
        )

    for operation in operations:
        if operation == "trim_whitespace":
            values = values.str.strip()
        elif operation == "collapse_internal_whitespace":
            values = values.str.replace(r"\s+", " ", regex=True)
        elif operation == "uppercase":
            values = values.str.upper()
        else:
            raise CuratedTransformationError(
                f"Unsupported cleaning operation for {source_name}: {operation}"
            )

    blank_mask = values.isna() | values.str.strip().eq("")

    if not nullable and blank_mask.any():
        raise CuratedTransformationError(
            f"Non-nullable curated column contains blank values: {source_name}"
        )

    if nullable:
        values = values.mask(blank_mask, pd.NA)

    return values.astype("string")


def _transform_integer(
    source: pd.Series,
    *,
    source_name: str,
    nullable: bool,
    target_dtype: str,
    zero_as_null: bool,
) -> pd.Series:
    """Convert one source column into a pandas integer dtype."""
    values = _normalize_blank_values(source)

    if not nullable and values.isna().any():
        raise CuratedTransformationError(
            f"Non-nullable integer column contains blanks: {source_name}"
        )

    try:
        numeric = pd.to_numeric(values, errors="raise")
    except (TypeError, ValueError, OverflowError) as exc:
        raise CuratedTransformationError(
            f"Integer conversion failed: {source_name}"
        ) from exc

    if zero_as_null:
        numeric = numeric.mask(numeric.eq(0), pd.NA)

    try:
        return numeric.astype(target_dtype)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CuratedTransformationError(
            f"Integer conversion failed: {source_name}"
        ) from exc


def _transform_float(
    source: pd.Series,
    *,
    source_name: str,
    nullable: bool,
) -> pd.Series:
    """Convert one source column into non-extension float64."""
    values = _normalize_blank_values(source)

    if not nullable and values.isna().any():
        raise CuratedTransformationError(
            f"Non-nullable float column contains blanks: {source_name}"
        )

    try:
        numeric = pd.to_numeric(values, errors="raise")
        return numeric.astype("float64")
    except (TypeError, ValueError, OverflowError) as exc:
        raise CuratedTransformationError(
            f"Float conversion failed: {source_name}"
        ) from exc


def _transform_date(
    source: pd.Series,
    *,
    source_name: str,
    nullable: bool,
    accepted_formats: Sequence[str],
) -> pd.Series:
    """Convert accepted source date formats into Arrow date32 values."""
    values = _normalize_blank_values(source)
    formats = list(accepted_formats)

    if not formats:
        raise CuratedTransformationError(
            f"No accepted date format configured: {source_name}"
        )

    if not nullable and values.isna().any():
        raise CuratedTransformationError(
            f"Non-nullable date column contains blanks: {source_name}"
        )

    parsed = pd.Series(
        pd.NaT,
        index=values.index,
        dtype="datetime64[ns]",
    )

    for date_format in formats:
        unresolved = parsed.isna() & values.notna()

        if not unresolved.any():
            break

        parsed.loc[unresolved] = pd.to_datetime(
            values.loc[unresolved],
            format=date_format,
            errors="coerce",
        )

    invalid = parsed.isna() & values.notna()

    if invalid.any():
        raise CuratedTransformationError(f"Date conversion failed: {source_name}")

    dates = parsed.dt.date

    return dates.astype(pd.ArrowDtype(pa.date32()))


def _normalize_blank_values(source: pd.Series) -> pd.Series:
    """Convert missing and whitespace-only source values into pandas nulls."""
    values = source.astype("string")
    blank_mask = values.isna() | values.str.strip().eq("")

    return values.mask(blank_mask, pd.NA)
