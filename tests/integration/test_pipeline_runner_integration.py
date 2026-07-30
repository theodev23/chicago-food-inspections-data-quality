"""Integration tests for the complete annual batch pipeline."""

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import yaml

from data_quality_pipeline.pipeline_runner import run_batch_pipeline

SOURCE_COLUMNS = [
    "inspection_id",
    "dba_name",
    "aka_name",
    "license_",
    "facility_type",
    "risk",
    "address",
    "city",
    "state",
    "zip",
    "inspection_date",
    "inspection_type",
    "results",
    "violations",
    "latitude",
    "longitude",
    "location",
]


def _source_record(
    inspection_id: str,
    *,
    dba_name: str,
) -> dict[str, str]:
    """Build one valid raw food-inspection record."""
    return {
        "inspection_id": inspection_id,
        "dba_name": dba_name,
        "aka_name": dba_name,
        "license_": "1234567",
        "facility_type": "Restaurant",
        "risk": "Risk 1 (High)",
        "address": "100 TEST STREET",
        "city": "CHICAGO",
        "state": "IL",
        "zip": "60601",
        "inspection_date": "2019-01-15T00:00:00.000",
        "inspection_type": "Canvass",
        "results": "Pass",
        "violations": "",
        "latitude": "41.881832",
        "longitude": "-87.623177",
        "location": "",
    }


def test_run_batch_pipeline_publishes_curated_and_quarantine_outputs(
    tmp_path: Path,
) -> None:
    """A real run should reconcile rows and publish both Parquet files."""
    incoming_dir = tmp_path / "incoming"
    incoming_dir.mkdir()

    source_path = incoming_dir / "food_inspections_2019.csv"

    retained = _source_record(
        "100",
        dba_name="DUPLICATE RESTAURANT",
    )
    duplicate = {
        **retained,
        "inspection_id": "101",
    }
    distinct = _source_record(
        "200",
        dba_name="DISTINCT RESTAURANT",
    )

    pd.DataFrame(
        [retained, duplicate, distinct],
        columns=SOURCE_COLUMNS,
    ).to_csv(
        source_path,
        index=False,
    )

    config = yaml.safe_load(
        Path("config/pipeline.yaml").read_text(
            encoding="utf-8",
        )
    )
    config["paths"]["curated"] = str(tmp_path / "curated")
    config["paths"]["quarantine"] = str(tmp_path / "quarantine")

    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text(
        yaml.safe_dump(
            config,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    result = run_batch_pipeline(
        source_path,
        config_path=config_path,
        contract_path="config/data_contract.yaml",
    )

    assert result.batch.year == 2019
    assert result.raw_row_count == 3
    assert result.accepted_row_count == 2
    assert result.rejected_record_count == 1
    assert result.quarantine_issue_count == 1
    assert result.error_count == 1
    assert result.warning_count == 0
    assert not result.validation.duplicate_detection_skipped

    assert result.curated_write.path == (
        tmp_path / "curated" / "inspection_year=2019" / "food_inspections_2019.parquet"
    )
    assert result.quarantine_write.path == (
        tmp_path
        / "quarantine"
        / "dq_batch_year=2019"
        / "food_inspections_2019_quarantine.parquet"
    )

    curated = pd.read_parquet(result.curated_write.path)
    quarantine = pd.read_parquet(result.quarantine_write.path)

    assert curated.shape == (2, 17)
    assert curated["inspection_id"].tolist() == [100, 200]
    assert curated["inspection_year"].tolist() == [2019, 2019]

    assert quarantine.shape == (1, 25)
    assert quarantine["inspection_id"].tolist() == ["101"]
    assert quarantine["dq_rule_id"].tolist() == ["exact_duplicate_record"]
    assert quarantine["dq_duplicate_of_inspection_id"].tolist() == [100]
    assert quarantine["dq_batch_year"].tolist() == [2019]

    assert (
        pq.ParquetFile(result.curated_write.path)
        .metadata.row_group(0)
        .column(0)
        .compression
        == "SNAPPY"
    )
    assert (
        pq.ParquetFile(result.quarantine_write.path)
        .metadata.row_group(0)
        .column(0)
        .compression
        == "SNAPPY"
    )

    rerun = run_batch_pipeline(
        source_path,
        config_path=config_path,
        contract_path="config/data_contract.yaml",
    )

    assert rerun.curated_write.path == result.curated_write.path
    assert rerun.quarantine_write.path == result.quarantine_write.path
    assert rerun.curated_write.row_count == 2
    assert rerun.quarantine_write.row_count == 1

    assert not list(tmp_path.rglob("*.tmp"))
