"""Run and validate the self-contained portfolio demonstration."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = PROJECT_ROOT / ".demo_runtime"

SOURCE_PATH = PROJECT_ROOT / "demo" / "input" / "food_inspections_2019.csv"
EXPECTED_PATH = PROJECT_ROOT / "demo" / "expected_summary.json"

CONFIG_PATH = PROJECT_ROOT / "config" / "demo_pipeline.yaml"
CONTRACT_PATH = PROJECT_ROOT / "config" / "data_contract.yaml"

INCOMING_PATH = RUNTIME_DIR / "incoming" / "food_inspections_2019.csv"
CURATED_PATH = (
    RUNTIME_DIR / "curated" / "inspection_year=2019" / "food_inspections_2019.parquet"
)
QUARANTINE_PATH = (
    RUNTIME_DIR
    / "quarantine"
    / "dq_batch_year=2019"
    / "food_inspections_2019_quarantine.parquet"
)
STATE_PATH = RUNTIME_DIR / "state" / "chicago_food_inspections_demo_2019.json"

FIRST_RUN_PATH = RUNTIME_DIR / "first_run.json"
SECOND_RUN_PATH = RUNTIME_DIR / "second_run.json"

type JsonObject = dict[str, Any]


class DemoError(RuntimeError):
    """Raised when the runnable demonstration is invalid."""


def _relative(path: Path) -> str:
    """Return a display path relative to the project root."""
    return str(path.relative_to(PROJECT_ROOT))


def _load_json(path: Path) -> JsonObject:
    """Load a JSON object from disk."""
    loaded = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(loaded, dict):
        raise DemoError(f"Expected a JSON object: {_relative(path)}")

    return loaded


def _find_cli() -> str:
    """Locate the installed project command."""
    cli_path = shutil.which("food-inspections-pipeline")

    if cli_path is None:
        raise DemoError(
            "The food-inspections-pipeline command is unavailable. "
            'Install the project with: python -m pip install -e ".[dev]"'
        )

    return cli_path


def _run_pipeline(output_path: Path) -> JsonObject:
    """Execute the real CLI and persist its JSON response."""
    command = [
        _find_cli(),
        "--all",
        "--config",
        _relative(CONFIG_PATH),
        "--contract",
        _relative(CONTRACT_PATH),
        "--json",
    ]

    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    if completed.returncode != 0:
        details = completed.stderr.strip() or completed.stdout.strip()
        raise DemoError(
            "Pipeline execution failed with exit code "
            f"{completed.returncode}: {details}"
        )

    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise DemoError("Pipeline output was not valid JSON.") from exc

    if not isinstance(payload, dict):
        raise DemoError("Pipeline output must be a JSON object.")

    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return payload


def _validate_first_run(
    payload: JsonObject,
    expected: JsonObject,
) -> JsonObject:
    """Validate the first processed execution summary."""
    expected_run = expected["first_run"]

    if payload.get("status") != "success":
        raise DemoError("The first pipeline execution did not succeed.")

    if payload.get("mode") != "all":
        raise DemoError("The first pipeline execution used the wrong mode.")

    if payload.get("summary") != expected_run["summary"]:
        raise DemoError(
            f"Unexpected first-run aggregate summary: {payload.get('summary')}"
        )

    batches = payload.get("batches")

    if not isinstance(batches, list) or len(batches) != 1:
        raise DemoError("Expected exactly one first-run batch.")

    batch = batches[0]

    if batch.get("status") != expected_run["status"]:
        raise DemoError(f"Unexpected first-run status: {batch.get('status')}")

    if batch.get("records") != expected_run["records"]:
        raise DemoError(f"Unexpected first-run record counts: {batch.get('records')}")

    if batch.get("validation") != expected_run["validation"]:
        raise DemoError(
            f"Unexpected first-run validation counts: {batch.get('validation')}"
        )

    if batch.get("batch", {}).get("year") != expected["batch_year"]:
        raise DemoError("Unexpected first-run batch year.")

    return batch


def _normalize_optional_identifier(value: object) -> str | None:
    """Normalize a nullable identifier read from Parquet."""
    if pd.isna(value):
        return None

    text = str(value)

    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]

    return text


def _validate_published_outputs(expected: JsonObject) -> None:
    """Validate curated, quarantine, and state artifacts."""
    required_paths = [
        CURATED_PATH,
        QUARANTINE_PATH,
        STATE_PATH,
    ]

    missing = [_relative(path) for path in required_paths if not path.is_file()]

    if missing:
        raise DemoError(f"Missing published outputs: {missing}")

    curated = pd.read_parquet(CURATED_PATH)
    quarantine = pd.read_parquet(QUARANTINE_PATH)
    state = _load_json(STATE_PATH)

    curated_ids = sorted(curated["inspection_id"].astype(int).tolist())

    if curated_ids != expected["curated_inspection_ids"]:
        raise DemoError(f"Unexpected curated inspection IDs: {curated_ids}")

    null_expectations = expected["null_transformations"]

    license_value = curated.loc[
        curated["inspection_id"] == null_expectations["license_number_inspection_id"],
        "license_number",
    ].iloc[0]

    risk_value = curated.loc[
        curated["inspection_id"] == null_expectations["risk_inspection_id"],
        "risk",
    ].iloc[0]

    if not pd.isna(license_value):
        raise DemoError("The zero-valued demo license was not converted to null.")

    if not pd.isna(risk_value):
        raise DemoError("The missing demo risk was not preserved as null.")

    actual_quarantine = []

    for _, row in quarantine.sort_values("inspection_id").iterrows():
        actual_quarantine.append(
            {
                "inspection_id": str(row["inspection_id"]),
                "rule_id": str(row["dq_rule_id"]),
                "duplicate_of_inspection_id": (
                    _normalize_optional_identifier(row["dq_duplicate_of_inspection_id"])
                ),
            }
        )

    if actual_quarantine != expected["quarantine"]:
        raise DemoError(f"Unexpected quarantine issues: {actual_quarantine}")

    first_run = expected["first_run"]
    records = first_run["records"]
    validation = first_run["validation"]

    expected_state = {
        "raw_row_count": records["raw"],
        "accepted_row_count": records["accepted"],
        "rejected_record_count": records["rejected"],
        "quarantine_row_count": records["quarantine_issues"],
        "error_count": validation["errors"],
        "warning_count": validation["warnings"],
    }

    actual_state = {key: state.get(key) for key in expected_state}

    if actual_state != expected_state:
        raise DemoError(f"Unexpected state-manifest counts: {actual_state}")


def _capture_output_timestamps() -> dict[Path, int]:
    """Capture output modification times after publication."""
    return {
        path: path.stat().st_mtime_ns
        for path in [
            CURATED_PATH,
            QUARANTINE_PATH,
            STATE_PATH,
        ]
    }


def _validate_second_run(
    payload: JsonObject,
    expected: JsonObject,
    timestamps: dict[Path, int],
) -> JsonObject:
    """Validate incremental skip behavior."""
    expected_run = expected["second_run"]

    if payload.get("status") != "success":
        raise DemoError("The second pipeline execution did not succeed.")

    if payload.get("summary") != expected_run["summary"]:
        raise DemoError(
            f"Unexpected second-run aggregate summary: {payload.get('summary')}"
        )

    batches = payload.get("batches")

    if not isinstance(batches, list) or len(batches) != 1:
        raise DemoError("Expected exactly one second-run batch.")

    batch = batches[0]

    if batch.get("status") != expected_run["status"]:
        raise DemoError(f"Unexpected second-run status: {batch.get('status')}")

    if batch.get("reason") != expected_run["reason"]:
        raise DemoError(f"Unexpected skip reason: {batch.get('reason')}")

    current_timestamps = {path: path.stat().st_mtime_ns for path in timestamps}

    if current_timestamps != timestamps:
        raise DemoError(
            "One or more published outputs were rewritten during the skipped execution."
        )

    return batch


def _print_report(expected: JsonObject) -> None:
    """Print a concise recruiter-facing demonstration report."""
    first_run = expected["first_run"]
    records = first_run["records"]
    validation = first_run["validation"]

    print()
    print("Chicago Food Inspections Data Quality Demo")
    print("=" * 42)
    print()
    print("First execution")
    print("----------------")
    print(f"Status:              {first_run['status']}")
    print(f"Raw records:         {records['raw']}")
    print(f"Accepted records:    {records['accepted']}")
    print(f"Rejected records:    {records['rejected']}")
    print(f"Blocking issues:     {validation['errors']}")
    print(f"Non-blocking warnings: {validation['warnings']}")
    print()
    print("Curated inspection IDs")
    print("------------------------")
    print(", ".join(str(value) for value in expected["curated_inspection_ids"]))
    print()
    print("Quarantine issues")
    print("-----------------")

    for issue in expected["quarantine"]:
        detail = issue["rule_id"]

        if issue["duplicate_of_inspection_id"] is not None:
            detail += f" -> duplicate of {issue['duplicate_of_inspection_id']}"

        print(f"{issue['inspection_id']}: {detail}")

    print()
    print("Validated warning transformations")
    print("---------------------------------")
    print("1003: zero license converted to null")
    print("1004: missing risk retained as null")
    print()
    print("Second execution")
    print("----------------")
    print(f"Status:  {expected['second_run']['status']}")
    print(f"Reason:  {expected['second_run']['reason']}")
    print("Outputs rewritten: no")
    print()
    print("Published artifacts")
    print("-------------------")
    print(_relative(CURATED_PATH))
    print(_relative(QUARANTINE_PATH))
    print(_relative(STATE_PATH))
    print(_relative(FIRST_RUN_PATH))
    print(_relative(SECOND_RUN_PATH))
    print()
    print("Demo completed successfully.")


def main() -> int:
    """Run the self-contained demonstration."""
    try:
        expected = _load_json(EXPECTED_PATH)

        if not SOURCE_PATH.is_file():
            raise DemoError(f"Demo input not found: {_relative(SOURCE_PATH)}")

        shutil.rmtree(RUNTIME_DIR, ignore_errors=True)
        INCOMING_PATH.parent.mkdir(parents=True)
        shutil.copy2(SOURCE_PATH, INCOMING_PATH)

        print("Running first pipeline execution...")
        first_payload = _run_pipeline(FIRST_RUN_PATH)
        _validate_first_run(first_payload, expected)
        _validate_published_outputs(expected)

        timestamps = _capture_output_timestamps()

        print("Running second pipeline execution...")
        second_payload = _run_pipeline(SECOND_RUN_PATH)
        _validate_second_run(
            second_payload,
            expected,
            timestamps,
        )

        _print_report(expected)
    except (DemoError, OSError, KeyError, IndexError) as exc:
        print(f"Demo failed: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
