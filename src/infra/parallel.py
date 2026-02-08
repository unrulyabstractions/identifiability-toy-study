"""Parallelization utilities for trial execution.

Provides clean interfaces for running independent computations in parallel
using ThreadPoolExecutor (appropriate for PyTorch operations that release GIL).
"""

from concurrent.futures import ThreadPoolExecutor, Future
from typing import Callable, TypeVar
from dataclasses import dataclass

import torch

from src.schemas.config import ParallelConfig


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
