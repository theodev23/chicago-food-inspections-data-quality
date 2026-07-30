"""Discover and execute all configured incoming annual batches."""

from dataclasses import dataclass
from pathlib import Path

from data_quality_pipeline.batch import (
    discover_incoming_batch_paths,
)
from data_quality_pipeline.config import load_config
from data_quality_pipeline.pipeline_runner import (
    BatchPipelineResult,
    BatchPipelineRunResult,
    BatchPipelineSkipResult,
    run_batch_pipeline,
)


@dataclass(frozen=True, slots=True)
class MultiBatchPipelineResult:
    """Describe one deterministic execution across discovered batches."""

    batch_paths: tuple[Path, ...]
    results: tuple[BatchPipelineResult, ...]

    @property
    def discovered_count(self) -> int:
        """Return the number of discovered annual batch files."""
        return len(self.batch_paths)

    @property
    def processed_count(self) -> int:
        """Return the number of annual batches fully processed."""
        return sum(
            isinstance(result, BatchPipelineRunResult) for result in self.results
        )

    @property
    def skipped_count(self) -> int:
        """Return the number of unchanged annual batches skipped."""
        return sum(
            isinstance(result, BatchPipelineSkipResult) for result in self.results
        )


class MultiBatchPipelineRunError(Exception):
    """Raised when multi-batch orchestration detects inconsistent results."""


def run_discovered_batches(
    *,
    config_path: str | Path = "config/pipeline.yaml",
    contract_path: str | Path = "config/data_contract.yaml",
) -> MultiBatchPipelineResult:
    """Discover and sequentially execute every configured annual batch.

    Batches are processed in the deterministic order returned by incoming
    discovery. Execution is fail-fast: any batch exception is propagated
    immediately and later batches are not started.

    Args:
        config_path: Pipeline configuration file.
        contract_path: Data-contract configuration file.

    Returns:
        Immutable aggregate metadata for all discovered annual batches.

    Raises:
        MultiBatchPipelineRunError: If a batch result does not correspond
            to the discovered path that was submitted.
        Exception: Discovery and per-batch pipeline errors are propagated.
    """
    config = load_config(config_path)

    source_config = config["source"]
    ingestion_config = config["ingestion"]
    paths_config = config["paths"]

    batch_paths = discover_incoming_batch_paths(
        paths_config["incoming"],
        file_pattern=ingestion_config["file_pattern"],
        start_year=source_config["start_year"],
        end_year=source_config["end_year"],
    )

    results: list[BatchPipelineResult] = []

    for batch_path in batch_paths:
        result = run_batch_pipeline(
            batch_path,
            config_path=config_path,
            contract_path=contract_path,
        )

        if result.batch.path != batch_path:
            raise MultiBatchPipelineRunError(
                "Pipeline result path does not match the discovered "
                f"batch path: {result.batch.path} != {batch_path}."
            )

        results.append(result)

    return MultiBatchPipelineResult(
        batch_paths=batch_paths,
        results=tuple(results),
    )
