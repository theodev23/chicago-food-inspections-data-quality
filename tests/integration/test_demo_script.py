"""Integration test for the self-contained runnable demonstration."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_demo.sh"
RUNTIME_DIR = PROJECT_ROOT / ".demo_runtime"

EXPECTED_ARTIFACTS = (
    RUNTIME_DIR / "curated" / "inspection_year=2019" / "food_inspections_2019.parquet",
    RUNTIME_DIR
    / "quarantine"
    / "dq_batch_year=2019"
    / "food_inspections_2019_quarantine.parquet",
    RUNTIME_DIR / "state" / "chicago_food_inspections_demo_2019.json",
    RUNTIME_DIR / "first_run.json",
    RUNTIME_DIR / "second_run.json",
)


def _load_json(path: Path) -> dict[str, object]:
    """Load a JSON object created by the demonstration."""
    loaded = json.loads(path.read_text(encoding="utf-8"))

    assert isinstance(loaded, dict)

    return loaded


def test_runnable_demo_from_external_working_directory(
    tmp_path: Path,
) -> None:
    """Run the complete demo and verify its published summaries."""
    shutil.rmtree(RUNTIME_DIR, ignore_errors=True)

    try:
        assert SCRIPT_PATH.is_file()
        assert os.access(SCRIPT_PATH, os.X_OK)

        completed = subprocess.run(
            [str(SCRIPT_PATH)],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
        )

        assert completed.returncode == 0, completed.stderr

        expected_output = (
            "Running first pipeline execution...",
            "Running second pipeline execution...",
            "Status:              processed",
            "Raw records:         7",
            "Accepted records:    5",
            "Rejected records:    2",
            "Blocking issues:     2",
            "Non-blocking warnings: 2",
            "1005: coordinate_pair_consistency",
            "1007: exact_duplicate_record -> duplicate of 1006",
            "Status:  skipped",
            "Reason:  batch_state_current",
            "Outputs rewritten: no",
            "Demo completed successfully.",
        )

        for value in expected_output:
            assert value in completed.stdout

        for artifact_path in EXPECTED_ARTIFACTS:
            assert artifact_path.is_file()

        first_run = _load_json(RUNTIME_DIR / "first_run.json")
        second_run = _load_json(RUNTIME_DIR / "second_run.json")

        assert first_run["summary"] == {
            "discovered": 1,
            "processed": 1,
            "skipped": 0,
        }
        assert first_run["batches"][0]["status"] == "processed"

        assert second_run["summary"] == {
            "discovered": 1,
            "processed": 0,
            "skipped": 1,
        }
        assert second_run["batches"][0]["status"] == "skipped"
        assert second_run["batches"][0]["reason"] == "batch_state_current"
    finally:
        shutil.rmtree(RUNTIME_DIR, ignore_errors=True)
