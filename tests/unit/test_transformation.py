"""Unit tests for curated record transformation."""

from copy import deepcopy
from datetime import date
from typing import Any

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from data_quality_pipeline.contract import load_data_contract
from data_quality_pipeline.transformation import (
    CuratedTransformationError,
    transform_curated_records,
)


def _contract() -> dict[str, Any]:
    """Load an independent contract instance for one test."""
    return deepcopy(load_data_contract("config/data_contract.yaml"))


def _valid_row(
    inspection_id: str = "1",
    **overrides: str,
) -> dict[str, str]:
    """Build one valid source record."""
    row = {
        "inspection_id": inspection_id,
        "dba_name": "EXAMPLE RESTAURANT",
        "aka_name": "EXAMPLE",
        "license_": "1234567",
        "facility_type": "Restaurant",
        "risk": "Risk 1 (High)",
        "address": "1 MAIN ST",
        "city": "CHICAGO",
        "state": "IL",
        "zip": "60601",
        "inspection_date": "2019-01-01T00:00:00.000",
        "inspection_type": "Canvass",
        "results": "Pass",
        "violations": "1. EXAMPLE VIOLATION",
        "latitude": "41.881832",
        "longitude": "-87.623177",
        "location": "(41.881832, -87.623177)",
    }
    row.update(overrides)

    return row


def _frame(*rows: dict[str, str]) -> pd.DataFrame:
    """Create a source frame in contractual column order."""
    contract = load_data_contract("config/data_contract.yaml")

    return pd.DataFrame(
        rows,
        columns=contract["source_schema"]["expected_columns"],
        dtype="string",
    )


def test_transform_curated_records_builds_expected_schema_and_types() -> None:
    """Published columns should use contractual names and target types."""
    curated = transform_curated_records(
        _frame(_valid_row()),
        batch_year=2019,
        contract=_contract(),
    )

    assert curated.columns.tolist() == [
        "inspection_id",
        "dba_name",
        "aka_name",
        "license_number",
        "facility_type",
        "risk",
        "address",
        "city",
        "state",
        "zip_code",
        "inspection_date",
        "inspection_type",
        "inspection_result",
        "violations",
        "latitude",
        "longitude",
        "inspection_year",
    ]

    assert str(curated["inspection_id"].dtype) == "int64"
    assert str(curated["license_number"].dtype) == "Int64"
    assert str(curated["inspection_date"].dtype) == "date32[day][pyarrow]"
    assert str(curated["latitude"].dtype) == "float64"
    assert str(curated["longitude"].dtype) == "float64"
    assert str(curated["inspection_year"].dtype) == "int16"

    assert curated.at[0, "inspection_date"] == date(2019, 1, 1)
    assert curated.at[0, "inspection_result"] == "Pass"
    assert "location" not in curated.columns


def test_transform_curated_records_applies_string_cleaning() -> None:
    """Configured whitespace and uppercase operations should be applied."""
    data = _frame(
        _valid_row(
            dba_name="  EXAMPLE   RESTAURANT  ",
            aka_name="  EXAMPLE    CAFE ",
            facility_type="  Mobile   Food   Preparer ",
            address="  1   MAIN   ST ",
            city=" chicago ",
            state=" il ",
            zip=" 60601 ",
            inspection_type="  License   Re-Inspection ",
            violations="  1. EXAMPLE   VIOLATION  ",
        )
    )

    curated = transform_curated_records(
        data,
        batch_year=2019,
        contract=_contract(),
    )

    assert curated.at[0, "dba_name"] == "EXAMPLE RESTAURANT"
    assert curated.at[0, "aka_name"] == "EXAMPLE CAFE"
    assert curated.at[0, "facility_type"] == "Mobile Food Preparer"
    assert curated.at[0, "address"] == "1 MAIN ST"
    assert curated.at[0, "city"] == "CHICAGO"
    assert curated.at[0, "state"] == "IL"
    assert curated.at[0, "zip_code"] == "60601"
    assert curated.at[0, "inspection_type"] == "License Re-Inspection"

    # Violations are trimmed but their internal structure is preserved.
    assert curated.at[0, "violations"] == "1. EXAMPLE   VIOLATION"


def test_transform_curated_records_normalizes_nullable_values() -> None:
    """Blank nullable values and the zero license sentinel should become null."""
    data = _frame(
        _valid_row(
            aka_name="",
            license_="0",
            facility_type="   ",
            risk="",
            city="",
            state=" ",
            zip="",
            violations="",
            latitude="",
            longitude="",
        )
    )

    curated = transform_curated_records(
        data,
        batch_year=2019,
        contract=_contract(),
    )

    for column in (
        "aka_name",
        "license_number",
        "facility_type",
        "risk",
        "city",
        "state",
        "zip_code",
        "violations",
        "latitude",
        "longitude",
    ):
        assert pd.isna(curated.at[0, column])


def test_transform_curated_records_does_not_modify_source_frame() -> None:
    """Transformation must not mutate the raw source records."""
    data = _frame(
        _valid_row(
            dba_name="  EXAMPLE   RESTAURANT ",
            license_="0",
        )
    )
    original = data.copy(deep=True)

    transform_curated_records(
        data,
        batch_year=2019,
        contract=_contract(),
    )

    assert_frame_equal(data, original)


def test_transform_curated_records_resets_index_and_preserves_row_order() -> None:
    """Filtered source indices should not leak into curated output."""
    data = _frame(
        _valid_row(inspection_id="2"),
        _valid_row(inspection_id="1"),
    )
    data.index = [20, 10]

    curated = transform_curated_records(
        data,
        batch_year=2019,
        contract=_contract(),
    )

    assert curated.index.tolist() == [0, 1]
    assert curated["inspection_id"].tolist() == [2, 1]


@pytest.mark.parametrize(
    "batch_year",
    [
        True,
        0,
        10000,
        2019.0,
    ],
)
def test_transform_curated_records_rejects_invalid_batch_year(
    batch_year: object,
) -> None:
    """The partition year must be a real four-digit-compatible integer."""
    with pytest.raises(
        CuratedTransformationError,
        match="Batch year must be an integer between 1 and 9999",
    ):
        transform_curated_records(
            _frame(_valid_row()),
            batch_year=batch_year,
            contract=_contract(),
        )


@pytest.mark.parametrize(
    "schema_change",
    [
        "missing",
        "reordered",
    ],
)
def test_transform_curated_records_rejects_source_schema_mismatch(
    schema_change: str,
) -> None:
    """Source columns must match the exact contractual order."""
    data = _frame(_valid_row())

    if schema_change == "missing":
        data = data.drop(columns=["location"])
    else:
        columns = data.columns.tolist()
        data = data[[columns[-1], *columns[:-1]]]

    with pytest.raises(
        CuratedTransformationError,
        match="Source columns do not match the contract order",
    ):
        transform_curated_records(
            data,
            batch_year=2019,
            contract=_contract(),
        )


def test_transform_curated_records_rejects_duplicate_target_names() -> None:
    """Two source columns must not produce the same curated column."""
    contract = _contract()
    contract["columns"]["aka_name"]["target_name"] = "dba_name"

    with pytest.raises(
        CuratedTransformationError,
        match="Duplicate curated target column: dba_name",
    ):
        transform_curated_records(
            _frame(_valid_row()),
            batch_year=2019,
            contract=contract,
        )


def test_transform_curated_records_reserves_inspection_year() -> None:
    """The partition column name must remain owned by the transformer."""
    contract = _contract()
    contract["columns"]["dba_name"]["target_name"] = "inspection_year"

    with pytest.raises(
        CuratedTransformationError,
        match="Contract target names must not use inspection_year",
    ):
        transform_curated_records(
            _frame(_valid_row()),
            batch_year=2019,
            contract=contract,
        )


def test_transform_curated_records_rejects_unsupported_target_type() -> None:
    """Only implemented contractual target types should be accepted."""
    contract = _contract()
    contract["columns"]["risk"]["target_type"] = "category"

    with pytest.raises(
        CuratedTransformationError,
        match="Unsupported target type for risk: category",
    ):
        transform_curated_records(
            _frame(_valid_row()),
            batch_year=2019,
            contract=contract,
        )


def test_transform_curated_records_rejects_invalid_string_cleaning() -> None:
    """String cleaning must be declared as an operation sequence."""
    contract = _contract()
    contract["columns"]["dba_name"]["cleaning"] = "trim_whitespace"

    with pytest.raises(
        CuratedTransformationError,
        match="String cleaning must be a sequence: dba_name",
    ):
        transform_curated_records(
            _frame(_valid_row()),
            batch_year=2019,
            contract=contract,
        )


def test_transform_curated_records_rejects_unknown_cleaning_operation() -> None:
    """Unknown string-cleaning vocabulary should fail explicitly."""
    contract = _contract()
    contract["columns"]["dba_name"]["cleaning"] = [
        "trim_whitespace",
        "lowercase",
    ]

    with pytest.raises(
        CuratedTransformationError,
        match="Unsupported cleaning operation for dba_name: lowercase",
    ):
        transform_curated_records(
            _frame(_valid_row()),
            batch_year=2019,
            contract=contract,
        )


def test_transform_curated_records_rejects_invalid_numeric_cleaning() -> None:
    """Nullable integer cleaning must be represented by a mapping."""
    contract = _contract()
    contract["columns"]["license_"]["cleaning"] = ["zero_as_null"]

    with pytest.raises(
        CuratedTransformationError,
        match="Numeric cleaning must be a mapping: license_",
    ):
        transform_curated_records(
            _frame(_valid_row()),
            batch_year=2019,
            contract=contract,
        )


def test_transform_curated_records_rejects_invalid_zero_as_null() -> None:
    """The zero-sentinel configuration must be a real boolean."""
    contract = _contract()
    contract["columns"]["license_"]["cleaning"] = {
        "zero_as_null": "true",
    }

    with pytest.raises(
        CuratedTransformationError,
        match="zero_as_null must be boolean: license_",
    ):
        transform_curated_records(
            _frame(_valid_row()),
            batch_year=2019,
            contract=contract,
        )


def test_transform_curated_records_rejects_blank_required_string() -> None:
    """A blank non-nullable string must not enter curated output."""
    data = _frame(_valid_row(dba_name="   "))

    with pytest.raises(
        CuratedTransformationError,
        match=("Non-nullable curated column contains blank values: dba_name"),
    ):
        transform_curated_records(
            data,
            batch_year=2019,
            contract=_contract(),
        )


def test_transform_curated_records_rejects_blank_required_integer() -> None:
    """A blank non-nullable integer must not enter curated output."""
    data = _frame(_valid_row(inspection_id=""))

    with pytest.raises(
        CuratedTransformationError,
        match=("Non-nullable integer column contains blanks: inspection_id"),
    ):
        transform_curated_records(
            data,
            batch_year=2019,
            contract=_contract(),
        )


def test_transform_curated_records_rejects_invalid_integer() -> None:
    """A malformed integer should raise a transformation error."""
    data = _frame(_valid_row(inspection_id="invalid"))

    with pytest.raises(
        CuratedTransformationError,
        match="Integer conversion failed: inspection_id",
    ):
        transform_curated_records(
            data,
            batch_year=2019,
            contract=_contract(),
        )


def test_transform_curated_records_rejects_invalid_float() -> None:
    """A malformed coordinate should raise a transformation error."""
    data = _frame(_valid_row(latitude="north"))

    with pytest.raises(
        CuratedTransformationError,
        match="Float conversion failed: latitude",
    ):
        transform_curated_records(
            data,
            batch_year=2019,
            contract=_contract(),
        )


def test_transform_curated_records_requires_date_format() -> None:
    """At least one source date format must be configured."""
    contract = _contract()
    contract["columns"]["inspection_date"]["accepted_formats"] = []

    with pytest.raises(
        CuratedTransformationError,
        match="No accepted date format configured: inspection_date",
    ):
        transform_curated_records(
            _frame(_valid_row()),
            batch_year=2019,
            contract=contract,
        )


def test_transform_curated_records_rejects_blank_required_date() -> None:
    """A blank non-nullable date must not enter curated output."""
    data = _frame(_valid_row(inspection_date=""))

    with pytest.raises(
        CuratedTransformationError,
        match=("Non-nullable date column contains blanks: inspection_date"),
    ):
        transform_curated_records(
            data,
            batch_year=2019,
            contract=_contract(),
        )


def test_transform_curated_records_rejects_invalid_date() -> None:
    """A value matching no accepted date format must be rejected."""
    data = _frame(
        _valid_row(
            inspection_date="2019-02-30T00:00:00.000",
        )
    )

    with pytest.raises(
        CuratedTransformationError,
        match="Date conversion failed: inspection_date",
    ):
        transform_curated_records(
            data,
            batch_year=2019,
            contract=_contract(),
        )
