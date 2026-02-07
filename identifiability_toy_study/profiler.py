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

    # Print results at end:
    print_profile()
"""

import functools
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Callable

# Global state
_times: dict[str, float] = defaultdict(float)
_counts: dict[str, int] = defaultdict(int)
_enabled = True


def reset():
    """Reset all timing data."""
    global _times, _counts
    _times.clear()
    _counts.clear()


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


def print_profile(min_ms: float = 1.0):
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
