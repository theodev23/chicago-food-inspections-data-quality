"""Command-line interface for the annual data-quality pipeline."""

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any

from data_quality_pipeline.pipeline_runner import (
    BatchPipelineRunResult,
    run_batch_pipeline,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="food-inspections-pipeline",
        description=(
            "Validate, transform, and publish one annual Chicago "
            "food-inspections CSV batch."
        ),
    )
    parser.add_argument(
        "file_path",
        help="Path to the annual incoming CSV file.",
    )
    parser.add_argument(
        "--config",
        default="config/pipeline.yaml",
        help=("Path to the pipeline configuration file (default: %(default)s)."),
    )
    parser.add_argument(
        "--contract",
        default="config/data_contract.yaml",
        help=("Path to the data-contract file (default: %(default)s)."),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Print the successful run summary as JSON.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the CLI and return a process exit code."""
    parser = build_parser()
    arguments = parser.parse_args(argv)

    try:
        result = run_batch_pipeline(
            arguments.file_path,
            config_path=arguments.config,
            contract_path=arguments.contract,
        )
    except Exception as exc:
        print(
            f"Pipeline failed ({type(exc).__name__}): {exc}",
            file=sys.stderr,
        )
        return 1

    if arguments.json_output:
        print(
            json.dumps(
                build_json_summary(result),
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(format_text_summary(result))

    return 0


def build_json_summary(
    result: BatchPipelineRunResult,
) -> dict[str, Any]:
    """Build a machine-readable successful-run summary."""
    return {
        "status": "success",
        "batch": {
            "path": str(result.batch.path),
            "year": result.batch.year,
            "size_bytes": result.batch.size_bytes,
            "checksum_algorithm": result.batch.checksum_algorithm,
            "checksum": result.batch.checksum,
        },
        "validation": {
            "errors": result.error_count,
            "warnings": result.warning_count,
            "duplicate_detection_skipped": (
                result.validation.duplicate_detection_skipped
            ),
        },
        "records": {
            "raw": result.raw_row_count,
            "accepted": result.accepted_row_count,
            "rejected": result.rejected_record_count,
            "quarantine_issues": result.quarantine_issue_count,
        },
        "outputs": {
            "curated": {
                "path": str(result.curated_write.path),
                "rows": result.curated_write.row_count,
                "size_bytes": result.curated_write.size_bytes,
                "compression": result.curated_write.compression,
            },
            "quarantine": {
                "path": str(result.quarantine_write.path),
                "rows": result.quarantine_write.row_count,
                "size_bytes": result.quarantine_write.size_bytes,
                "compression": result.quarantine_write.compression,
            },
        },
    }


def format_text_summary(
    result: BatchPipelineRunResult,
) -> str:
    """Format a readable successful-run summary."""
    duplicate_status = (
        "skipped" if result.validation.duplicate_detection_skipped else "completed"
    )

    lines = [
        "Pipeline completed successfully",
        "",
        "[Batch]",
        f"Path: {result.batch.path}",
        f"Year: {result.batch.year}",
        f"Size bytes: {result.batch.size_bytes}",
        (f"Checksum ({result.batch.checksum_algorithm}): {result.batch.checksum}"),
        "",
        "[Records]",
        f"Raw: {result.raw_row_count}",
        f"Accepted: {result.accepted_row_count}",
        f"Rejected: {result.rejected_record_count}",
        f"Quarantine issues: {result.quarantine_issue_count}",
        "",
        "[Validation]",
        f"Errors: {result.error_count}",
        f"Warnings: {result.warning_count}",
        f"Duplicate detection: {duplicate_status}",
        "",
        "[Curated output]",
        f"Path: {result.curated_write.path}",
        f"Rows: {result.curated_write.row_count}",
        f"Size bytes: {result.curated_write.size_bytes}",
        f"Compression: {result.curated_write.compression}",
        "",
        "[Quarantine output]",
        f"Path: {result.quarantine_write.path}",
        f"Rows: {result.quarantine_write.row_count}",
        f"Size bytes: {result.quarantine_write.size_bytes}",
        f"Compression: {result.quarantine_write.compression}",
    ]

    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
