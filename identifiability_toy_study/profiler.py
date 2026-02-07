"""Simple global profiler for timing code sections.

Usage:
    from identifiability_toy_study.profiler import profile, print_profile, logged

    with profile("section_name"):
        # code to time
        pass

    # Or as decorator with logging:
    @logged("function_name")
    def my_function(logger=None):
        pass

    # Or with memory logging:
    @profile_fn("My Function")
    def my_function():
        pass

    # Print results at end:
    print_profile()
"""

import functools
import os
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Callable, TypeVar

# Global state
_times: dict[str, float] = defaultdict(float)
_counts: dict[str, int] = defaultdict(int)
_memory_snapshots: dict[str, float] = {}  # MB at each checkpoint
_enabled = True

F = TypeVar("F", bound=Callable[..., Any])


def reset():
    """Reset all timing data."""
    global _times, _counts, _memory_snapshots
    _times.clear()
    _counts.clear()
    _memory_snapshots.clear()


def enable():
    """Enable profiling."""
    global _enabled
    _enabled = True


def disable():
    """Disable profiling."""
    global _enabled
    _enabled = False


@contextmanager
def profile(name: str):
    """Context manager to time a code section."""
    if not _enabled:
        yield
        return

    t0 = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - t0
        _times[name] += elapsed
        _counts[name] += 1


def print_profile(min_ms: float = 1.0, include_memory: bool = True):
    """Print profiling results sorted by total time."""
    if not _times:
        print("[PROFILE] No timing data")
        return

    total = sum(_times.values())
    print(f"\n[PROFILE] Total: {total*1000:.0f}ms")
    print("-" * 60)

    for name, t in sorted(_times.items(), key=lambda x: -x[1]):
        ms = t * 1000
        if ms < min_ms:
            continue
        count = _counts[name]
        avg = ms / count if count > 0 else 0
        pct = (t / total * 100) if total > 0 else 0
        print(f"  {name}: {ms:.0f}ms ({count} calls, avg {avg:.0f}ms) [{pct:.1f}%]")

    if include_memory:
        print_memory_summary()


def logged(
    phase_name: str,
    log_start: bool = True,
    log_end: bool = True,
) -> Callable:
    """Decorator to log and profile a function.

    Usage:
        @logged("training")
        def train_model(data, logger=None):
            ...

    The decorated function should accept an optional 'logger' kwarg.
    If logger is provided, logs start/end messages with timing.
    Always profiles the function duration.

    Args:
        phase_name: Name for logging and profiling
        log_start: Whether to log "Starting..." message
        log_end: Whether to log "Complete" message with duration
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            logger = kwargs.get("logger")

            if logger and log_start:
                logger.info(f"\n{'='*60}\n  [{phase_name}] Starting...\n{'='*60}")

            t0 = time.time()
            with profile(phase_name):
                result = func(*args, **kwargs)
            elapsed_ms = (time.time() - t0) * 1000

            if logger and log_end:
                logger.info(f"  [{phase_name}] Complete ({elapsed_ms:.0f}ms)\n")

            return result

        return wrapper

    return decorator


def get_memory_mb() -> float:
    """Get current process memory usage in MB."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        # Fallback for systems without psutil
        return 0.0


def log_memory(checkpoint: str) -> float:
    """Log memory at a checkpoint and return current usage in MB."""
    mem_mb = get_memory_mb()
    _memory_snapshots[checkpoint] = mem_mb
    return mem_mb


def profile_fn(identifier: str) -> Callable[[F], F]:
    """Decorator to profile functions with timing and memory tracking.

    Usage:
        @profile_fn("Train Model")
        def train_model(data):
            ...

    Prints a header, profiles the function, and logs memory after completion.
    """
    profile_name = identifier.lower().replace(" ", "_")

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get memory before
            mem_before = get_memory_mb()

            # Print step header
            print(f"\n{'=' * 60}")
            print(f"  {identifier}")
            print(f"  Memory before: {mem_before:.1f} MB")
            print("=" * 60)

            # Run with profiling
            t0 = time.time()
            with profile(profile_name):
                result = func(*args, **kwargs)
            elapsed_ms = (time.time() - t0) * 1000

            # Log memory after
            mem_after = log_memory(f"after_{profile_name}")
            mem_delta = mem_after - mem_before

            # Print summary
            print(f"  -> Completed in {elapsed_ms:.0f}ms")
            print(f"  -> Memory after: {mem_after:.1f} MB (delta: {mem_delta:+.1f} MB)")

            return result

        return wrapper  # type: ignore

    return decorator


def print_memory_summary():
    """Print memory usage at all checkpoints."""
    if not _memory_snapshots:
        print("[MEMORY] No memory snapshots")
        return

    print(f"\n[MEMORY] Snapshots:")
    print("-" * 60)

    # Sort by checkpoint name for readability
    for name, mem_mb in sorted(_memory_snapshots.items()):
        print(f"  {name}: {mem_mb:.1f} MB")
