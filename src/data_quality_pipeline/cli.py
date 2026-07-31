"""Command-line interface for the annual data-quality pipeline."""

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any

from data_quality_pipeline.multi_batch_runner import (
    MultiBatchPipelineResult,
    run_discovered_batches,
)
from data_quality_pipeline.pipeline_runner import (
    BatchPipelineResult,
    BatchPipelineRunResult,
    BatchPipelineSkipResult,
    run_batch_pipeline,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="food-inspections-pipeline",
        description=(
            "Validate, transform, and publish one annual Chicago "
            "food-inspections CSV batch or all configured incoming "
            "batches."
        ),
    )
    parser.add_argument(
        "file_path",
        nargs="?",
        help=("Path to one annual incoming CSV file. Omit when using --all."),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="all_batches",
        help=("Discover and execute all configured incoming annual batches."),
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

    if arguments.all_batches and arguments.file_path is not None:
        parser.error("file_path cannot be used together with --all.")

    if not arguments.all_batches and arguments.file_path is None:
        parser.error("file_path is required unless --all is used.")

    try:
        if arguments.all_batches:
            multi_result = run_discovered_batches(
                config_path=arguments.config,
                contract_path=arguments.contract,
            )
        else:
            batch_result = run_batch_pipeline(
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

    if arguments.all_batches:
        if arguments.json_output:
            print(
                json.dumps(
                    build_multi_batch_json_summary(multi_result),
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            print(format_multi_batch_text_summary(multi_result))
    elif arguments.json_output:
        print(
            json.dumps(
                build_json_summary(batch_result),
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(format_text_summary(batch_result))

    return 0


def build_multi_batch_json_summary(
    result: MultiBatchPipelineResult,
) -> dict[str, Any]:
    """Build a machine-readable multi-batch execution summary."""
    return {
        "status": "success",
        "mode": "all",
        "summary": {
            "discovered": result.discovered_count,
            "processed": result.processed_count,
            "skipped": result.skipped_count,
        },
        "batches": [
            build_json_summary(batch_result) for batch_result in result.results
        ],
    }


def format_multi_batch_text_summary(
    result: MultiBatchPipelineResult,
) -> str:
    """Format a readable multi-batch execution summary."""
    lines = [
        "Multi-batch pipeline completed successfully",
        "",
        "[Summary]",
        f"Discovered: {result.discovered_count}",
        f"Processed: {result.processed_count}",
        f"Skipped: {result.skipped_count}",
        "",
        "[Batches]",
    ]

    if not result.results:
        lines.append("No incoming batches discovered.")
    else:
        for batch_result in result.results:
            status = (
                "processed"
                if isinstance(
                    batch_result,
                    BatchPipelineRunResult,
                )
                else "skipped"
            )
            lines.append(
                f"{batch_result.batch.year}: {status} - {batch_result.batch.path}"
            )

    return "\n".join(lines)


def build_json_summary(
    result: BatchPipelineResult,
) -> dict[str, Any]:
    """Build a machine-readable pipeline-result summary."""
    if isinstance(result, BatchPipelineSkipResult):
        manifest = result.manifest

        return {
            "status": "skipped",
            "reason": "batch_state_current",
            "batch": {
                "path": str(result.batch.path),
                "year": result.batch.year,
                "size_bytes": result.batch.size_bytes,
                "checksum_algorithm": (result.batch.checksum_algorithm),
                "checksum": result.batch.checksum,
            },
            "state": {
                "manifest_path": str(result.state_manifest_path),
                "completed_at_utc": (manifest.completed_at_utc),
            },
            "validation": {
                "errors": manifest.error_count,
                "warnings": manifest.warning_count,
            },
            "records": {
                "raw": manifest.raw_row_count,
                "accepted": manifest.accepted_row_count,
                "rejected": manifest.rejected_record_count,
                "quarantine_issues": (manifest.quarantine_issue_count),
            },
            "outputs": {
                "curated": {
                    "path": manifest.curated_path,
                    "rows": manifest.curated_row_count,
                    "size_bytes": (manifest.curated_size_bytes),
                    "compression": (manifest.curated_compression),
                },
                "quarantine": {
                    "path": manifest.quarantine_path,
                    "rows": manifest.quarantine_row_count,
                    "size_bytes": (manifest.quarantine_size_bytes),
                    "compression": (manifest.quarantine_compression),
                },
            },
        }

    return {
        "status": "processed",
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
    result: BatchPipelineResult,
) -> str:
    """Format a readable pipeline-result summary."""
    if isinstance(result, BatchPipelineSkipResult):
        manifest = result.manifest

        lines = [
            "Pipeline skipped: batch state is current",
            "",
            "[Batch]",
            f"Path: {result.batch.path}",
            f"Year: {result.batch.year}",
            f"Size bytes: {result.batch.size_bytes}",
            (f"Checksum ({result.batch.checksum_algorithm}): {result.batch.checksum}"),
            "",
            "[State]",
            f"Manifest path: {result.state_manifest_path}",
            f"Completed at UTC: {manifest.completed_at_utc}",
            "",
            "[Records]",
            f"Raw: {manifest.raw_row_count}",
            f"Accepted: {manifest.accepted_row_count}",
            f"Rejected: {manifest.rejected_record_count}",
            (f"Quarantine issues: {manifest.quarantine_issue_count}"),
            "",
            "[Validation]",
            f"Errors: {manifest.error_count}",
            f"Warnings: {manifest.warning_count}",
            "",
            "[Curated output]",
            f"Path: {manifest.curated_path}",
            f"Rows: {manifest.curated_row_count}",
            f"Size bytes: {manifest.curated_size_bytes}",
            f"Compression: {manifest.curated_compression}",
            "",
            "[Quarantine output]",
            f"Path: {manifest.quarantine_path}",
            f"Rows: {manifest.quarantine_row_count}",
            f"Size bytes: {manifest.quarantine_size_bytes}",
            f"Compression: {manifest.quarantine_compression}",
        ]

        return "\n".join(lines)

    return _format_processed_text_summary(result)


def _format_processed_text_summary(
    result: BatchPipelineRunResult,
) -> str:
    """Format a readable summary for a processed annual batch."""
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
