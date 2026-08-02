# Runnable Demo

This directory contains a small synthetic Chicago food-inspections batch that
demonstrates the pipeline without requiring an external dataset download.

The demonstration uses the real project CLI, data contract, validation rules,
transformation code, Parquet writers, and incremental state logic. It does not
use a simplified or separate implementation.

## Quick start

From the repository root:

    python3.12 -m venv .venv
    source .venv/bin/activate
    python -m pip install -e ".[dev]"
    ./scripts/run_demo.sh

The command runs the pipeline twice:

1. the first execution validates, transforms, quarantines, and publishes the
   synthetic batch;
2. the second execution verifies that the unchanged batch is skipped and that
   the published files are not rewritten.

## Synthetic scenarios

The input file is:

    demo/input/food_inspections_2019.csv

It contains seven inspections designed to exercise key pipeline behavior:

| Inspection | Scenario | Expected behavior |
|---|---|---|
| `1001` | Valid restaurant inspection | Published |
| `1002` | Valid conditional pass | Published |
| `1003` | License number equal to zero | Published with license converted to null |
| `1004` | Missing risk value | Published with nullable risk |
| `1005` | Latitude present without longitude | Quarantined |
| `1006` | Canonical inspection in duplicate pair | Published |
| `1007` | Exact duplicate of inspection `1006` | Quarantined as duplicate |

The expected results are versioned in:

    demo/expected_summary.json

## Expected result

The first execution produces the following data-quality summary:

| Metric | Count |
|---|---:|
| Raw records | 7 |
| Accepted records | 5 |
| Rejected records | 2 |
| Blocking errors | 2 |
| Non-blocking warnings | 2 |
| Quarantine issues | 2 |

The second execution detects that the batch state is current and skips the
unchanged input without rewriting the published artifacts.

## Generated artifacts

The demonstration writes its runtime files under `.demo_runtime/`:

    .demo_runtime/curated/inspection_year=2019/food_inspections_2019.parquet
    .demo_runtime/quarantine/dq_batch_year=2019/food_inspections_2019_quarantine.parquet
    .demo_runtime/state/chicago_food_inspections_demo_2019.json
    .demo_runtime/first_run.json
    .demo_runtime/second_run.json

This directory is excluded from version control and is recreated on every demo
execution.

## Validation performed by the runner

The runner exits with a non-zero status when any expected behavior changes. It
verifies:

- aggregate record and validation counts;
- the exact curated inspection identifiers;
- null handling for the zero license and missing risk scenarios;
- quarantine rule identifiers and duplicate lineage;
- state-manifest counts;
- incremental skip behavior;
- unchanged output modification timestamps after the skipped execution.

## Automated test

Run the dedicated integration test with:

    python -m pytest tests/integration/test_demo_script.py -v

The test launches the shell entry point from an external working directory,
checks the generated summaries and artifacts, and removes `.demo_runtime/`
after completion.

## Manual cleanup

The runner replaces the runtime directory automatically. To remove its outputs
manually:

    rm -rf .demo_runtime
