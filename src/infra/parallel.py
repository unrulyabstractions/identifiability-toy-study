"""Parallelization utilities for trial execution.

Provides clean interfaces for running independent computations in parallel
using ThreadPoolExecutor (appropriate for PyTorch operations that release GIL).
"""

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, TypeVar

import torch

from src.schemas.schema_class import SchemaClass


@dataclass
class ParallelConfig(SchemaClass):
    """Configuration for parallelization and compute optimization.

    Optimized defaults based on M4 Max benchmarks:
    - MPS precomputed is 2.7x faster than CPU for batched eval
    - Sequential structure analysis is FASTER than parallel (thread overhead dominates)
    - Larger batch sizes improve throughput

    PyTorch GPU ops are NOT thread-safe, so we avoid threading for GPU work.
    """

    # Device selection for batched circuit evaluation
    eval_device: str = "mps"  # "cpu" or "mps" - MPS is 2.7x faster with precompute
    use_mps_if_available: bool = True

    # Structure analysis parallelization
    # BENCHMARK RESULT: Sequential is FASTER (77ms vs 130ms) because
    # thread overhead exceeds computation time per circuit
    max_workers_structure: int = 1  # 1 = sequential (fastest based on benchmark)
    enable_parallel_structure: bool = False  # Disabled - sequential is faster

    # Batched evaluation settings (GPU)
    precompute_masks: bool = True  # Pre-stack masks: 5.8ms vs 9.6ms on MPS

    # Robustness/Faithfulness - these involve GPU, so threading is risky
    # KEEP FALSE to avoid GPU thread safety issues
    enable_parallel_robustness: bool = False
    enable_parallel_faithfulness: bool = False

    # Memory optimization - use more memory for speed
    cache_subcircuit_models: bool = True  # Cache models for best subcircuits


def get_eval_device(parallel_config: ParallelConfig, default_device: str) -> str:
    """Determine the device to use for batched circuit evaluation.

    Prefers MPS if available and configured, otherwise falls back to the
    configured eval_device or default_device.
    """
    if parallel_config.use_mps_if_available and parallel_config.eval_device == "mps":
        if torch.backends.mps.is_available():
            return "mps"
    return (
        parallel_config.eval_device
        if parallel_config.eval_device != "mps"
        else default_device
    )


T = TypeVar("T")


@dataclass
class ParallelTasks:
    """Context manager for running tasks in parallel.

    Usage:
        with ParallelTasks() as tasks:
            robustness_future = tasks.submit(calculate_robustness, args...)
            counterfactual_future = tasks.submit(create_counterfactual, args...)

        # Results available after exiting context
        robustness = robustness_future.result()
        counterfactual = counterfactual_future.result()
    """

    max_workers: int = 4
    _executor: ThreadPoolExecutor | None = None

    def __enter__(self) -> "ParallelTasks":
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
        return self

    def __exit__(self, *args) -> None:
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None

    def submit(self, fn: Callable[..., T], *args, **kwargs) -> Future[T]:
        """Submit a task for parallel execution."""
        if self._executor is None:
            raise RuntimeError("ParallelTasks must be used as context manager")
        return self._executor.submit(fn, *args, **kwargs)


def run_parallel(*tasks: Callable[[], T]) -> list[T]:
    """Run multiple zero-argument callables in parallel and return results.

    Usage:
        results = run_parallel(
            lambda: calculate_robustness(...),
            lambda: create_counterfactual(...),
        )
        robustness, counterfactual = results
    """
    with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
        futures = [executor.submit(task) for task in tasks]
        return [f.result() for f in futures]
