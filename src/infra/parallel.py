"""Parallelization utilities for trial execution.

Provides clean interfaces for running independent computations in parallel
using ThreadPoolExecutor (appropriate for PyTorch operations that release GIL).
"""

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, TypeVar

import torch

from src.schema_class import SchemaClass


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


R = TypeVar("R")


def parallel_map(
    items: list[T],
    fn: Callable[[T], R],
    max_workers: int = 4,
    desc: str = "items",
    single_item_fn: Callable[[T], R] | None = None,
) -> list[tuple[int, R | None, Exception | None]]:
    """Apply fn to items in parallel, yielding (index, result, error) tuples.

    Optimizes single-item case by running directly without thread overhead.
    For single items, uses single_item_fn if provided (e.g., to enable logging).

    Args:
        items: List of items to process
        fn: Function to apply to each item (used for parallel execution)
        max_workers: Maximum parallel workers
        desc: Description for progress messages
        single_item_fn: Optional function for single-item case (e.g., with logging)

    Returns:
        List of (index, result, error) tuples. result is None if error occurred,
        error is None if successful.

    Usage:
        results = parallel_map(
            configs,
            lambda c: process(c, logger=None),  # Parallel: no logging
            single_item_fn=lambda c: process(c, logger=logger),  # Single: with logging
            desc="trials",
        )
        for idx, result, error in results:
            if error:
                print(f"Item {idx} failed: {error}")
            else:
                handle_result(result)
    """
    from concurrent.futures import as_completed

    if len(items) == 0:
        return []

    # Use single_item_fn for sequential execution (single item or max_workers=1)
    use_sequential = len(items) == 1 or max_workers == 1
    item_fn = single_item_fn if (single_item_fn and use_sequential) else fn

    if use_sequential:
        # Run sequentially without threading overhead
        results: list[tuple[int, R | None, Exception | None]] = []
        for idx, item in enumerate(items):
            try:
                result = item_fn(item)
                results.append((idx, result, None))
            except Exception as e:
                results.append((idx, None, e))
                print(f"  {desc.capitalize()} {idx + 1}/{len(items)} failed: {e}")
        return results

    # Multiple items - run in parallel
    n_workers = min(len(items), max_workers)
    print(f"Running {len(items)} {desc} with {n_workers} parallel workers")

    results = []

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(fn, item): i for i, item in enumerate(items)}

        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                results.append((idx, result, None))
                print(f"  {desc.capitalize()} {idx + 1}/{len(items)} completed")
            except Exception as e:
                results.append((idx, None, e))
                print(f"  {desc.capitalize()} {idx + 1}/{len(items)} failed: {e}")

    return results
