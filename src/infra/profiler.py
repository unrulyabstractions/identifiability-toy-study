"""Simple global profiler for timing code sections with optional debug tracing.

Usage:
    from src.infra.profiler import profile, print_profile, logged

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

Debug Tracing:
    from src.infra.profiler import Trace, trace, traced, debug_break, debug_break_if

    # Enable detailed tracing (typically via main.py --debug)
    Trace.enable()

    # One-liner trace messages
    trace("Processing batch", batch_id=5, size=128)

    # Block entry/exit tracing
    with traced("compute_gradients"):
        do_work()

    # Conditional breakpoints (only active when --debug)
    debug_break("Checkpoint reached", step=100)
    debug_break_if(loss > 10.0, "Loss exploded", loss=loss)
"""

import functools
import os
import pdb
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Callable, TypeVar

# Global profiler state
_times: dict[str, float] = defaultdict(float)
_counts: dict[str, int] = defaultdict(int)
_memory_snapshots: dict[str, float] = {}  # MB at each checkpoint
_enabled = True


# =============================================================================
# Debug Tracing System
# =============================================================================


class Trace:
    """Global debug tracing configuration.

    When enabled, provides detailed execution traces without cluttering
    production logs. Enable via main.py --debug flag.

    The tracing system prints timestamped messages showing:
    - Function/block entry and exit
    - Key variable values
    - Progress through loops and iterations

    Usage:
        # Enable globally (typically in main.py based on --debug flag)
        Trace.enable()

        # One-liner trace
        trace("Processing batch", batch_id=5, size=128)
        # Output: [TRACE +12.3s] Processing batch (batch_id=5, size=128)

        # Block tracing
        with traced("compute_gradients"):
            do_work()
        # Output:
        # [TRACE +12.3s] → compute_gradients
        # [TRACE +15.7s] ← compute_gradients (3.4s)
    """

    _enabled: bool = False
    _start_time: float | None = None
    _indent: int = 0

    @classmethod
    def enable(cls) -> None:
        """Enable debug tracing globally."""
        cls._enabled = True
        cls._start_time = time.time()
        cls._indent = 0

    @classmethod
    def disable(cls) -> None:
        """Disable debug tracing globally."""
        cls._enabled = False
        cls._start_time = None
        cls._indent = 0

    @classmethod
    def is_enabled(cls) -> bool:
        """Check if tracing is enabled."""
        return cls._enabled

    @classmethod
    def elapsed(cls) -> str:
        """Get formatted elapsed time since tracing was enabled."""
        if cls._start_time is None:
            return "0.0s"
        return f"{time.time() - cls._start_time:.1f}s"

    @classmethod
    def indent_str(cls) -> str:
        """Get current indentation string."""
        return "  " * cls._indent


def trace(msg: str, **kwargs: Any) -> None:
    """Print a trace message if debug tracing is enabled.

    Args:
        msg: The message to print
        **kwargs: Key-value pairs to include in the trace

    Usage:
        trace("Processing item", idx=5, total=100)
        # Output: [TRACE +12.3s] Processing item (idx=5, total=100)
    """
    if not Trace.is_enabled():
        return

    elapsed = Trace.elapsed()
    indent = Trace.indent_str()
    if kwargs:
        extra = " (" + ", ".join(f"{k}={v}" for k, v in kwargs.items()) + ")"
    else:
        extra = ""
    print(f"[TRACE +{elapsed}] {indent}{msg}{extra}", flush=True)


@contextmanager
def traced(name: str, **kwargs: Any):
    """Context manager for tracing block entry/exit.

    Shows entry with →, exit with ←, and duration.
    Supports nesting with indentation.

    Args:
        name: Name of the block being traced
        **kwargs: Key-value pairs to include in entry trace

    Usage:
        with traced("compute_metrics", gate="XOR"):
            do_heavy_computation()
        # Output:
        # [TRACE +12.3s] → compute_metrics (gate=XOR)
        # [TRACE +15.7s] ← compute_metrics (3.4s)
    """
    if not Trace.is_enabled():
        yield
        return

    elapsed_start = Trace.elapsed()
    indent = Trace.indent_str()
    extra = ""
    if kwargs:
        extra = " (" + ", ".join(f"{k}={v}" for k, v in kwargs.items()) + ")"
    print(f"[TRACE +{elapsed_start}] {indent}→ {name}{extra}", flush=True)

    Trace._indent += 1
    t0 = time.time()
    try:
        yield
    finally:
        Trace._indent -= 1
        duration = time.time() - t0
        elapsed_end = Trace.elapsed()
        indent = Trace.indent_str()
        print(f"[TRACE +{elapsed_end}] {indent}← {name} ({duration:.1f}s)", flush=True)


def trace_progress(current: int, total: int, desc: str, every: int = 10) -> None:
    """Print progress trace at regular intervals.

    Only prints when current % every == 0 or current == total.

    Args:
        current: Current item number (1-indexed)
        total: Total number of items
        desc: Description of what's being processed
        every: Print every N items

    Usage:
        for i, item in enumerate(items):
            trace_progress(i + 1, len(items), "subcircuits", every=10)
            process(item)
    """
    if not Trace.is_enabled():
        return

    if current % every == 0 or current == total or current == 1:
        pct = current / total * 100 if total > 0 else 0
        trace(f"{desc}: {current}/{total} ({pct:.0f}%)")


def debug_break(msg: str = "", **kwargs: Any) -> None:
    """Drop into pdb debugger if debug tracing is enabled.

    This is a no-op when --debug is not passed to main.py.
    When debug is enabled, prints a trace message and drops into pdb.

    Args:
        msg: Optional message to print before breaking
        **kwargs: Key-value pairs to include in trace output

    Usage:
        debug_break("About to process batch", batch_id=5)
        # When --debug: prints trace and drops into pdb
        # Without --debug: does nothing
    """
    if not Trace.is_enabled():
        return

    if msg:
        trace(f"BREAK: {msg}", **kwargs)
    else:
        trace("BREAK (entering debugger)")

    # Use set_trace() to drop into debugger
    # The caller's frame is 1 level up
    pdb.set_trace()


def debug_break_if(condition: bool, msg: str = "", **kwargs: Any) -> None:
    """Conditionally drop into pdb debugger if condition is True and debug enabled.

    This is a no-op when:
    - --debug is not passed to main.py, OR
    - condition is False

    Args:
        condition: Only break if this is True
        msg: Optional message to print before breaking
        **kwargs: Key-value pairs to include in trace output

    Usage:
        debug_break_if(loss > 10.0, "Loss exploded", loss=loss)
        debug_break_if(batch_idx == 100, "Check batch 100")
    """
    if not condition:
        return
    debug_break(msg, **kwargs)


# =============================================================================
# Profiler Core
# =============================================================================

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
def profile(name: str, trace_it: bool = True):
    """Context manager to time a code section.

    Args:
        name: Name for this profiled section
        trace_it: If True and Trace.is_enabled(), also emit trace messages
    """
    should_trace = trace_it and Trace.is_enabled()

    if should_trace:
        elapsed_start = Trace.elapsed()
        indent = Trace.indent_str()
        print(f"[TRACE +{elapsed_start}] {indent}▶ {name}", flush=True)
        Trace._indent += 1

    if not _enabled:
        try:
            yield
        finally:
            if should_trace:
                Trace._indent -= 1
        return

    t0 = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - t0
        _times[name] += elapsed
        _counts[name] += 1

        if should_trace:
            Trace._indent -= 1
            elapsed_end = Trace.elapsed()
            indent = Trace.indent_str()
            print(
                f"[TRACE +{elapsed_end}] {indent}◀ {name} ({elapsed*1000:.0f}ms)",
                flush=True,
            )


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


def profile_fn(identifier: str, trace_args: bool = False) -> Callable[[F], F]:
    """Decorator to profile functions with timing and memory tracking.

    Usage:
        @profile_fn("Train Model")
        def train_model(data):
            ...

    Prints a header, profiles the function, and logs memory after completion.
    When Trace is enabled, also emits trace messages.

    Args:
        identifier: Human-readable name for this function
        trace_args: If True, include function arguments in trace output
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

            # Trace entry if enabled
            if Trace.is_enabled() and trace_args:
                arg_summary = _summarize_args(args, kwargs)
                trace(f"Called {func.__name__}", **arg_summary)

            # Run with profiling (trace_it=False since we handle it here)
            t0 = time.time()
            with profile(profile_name, trace_it=False):
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


def _summarize_args(args: tuple, kwargs: dict, max_len: int = 50) -> dict[str, Any]:
    """Create a summary of function arguments for tracing."""
    summary = {}
    for i, arg in enumerate(args[:3]):  # First 3 positional args
        val = _summarize_value(arg, max_len)
        summary[f"arg{i}"] = val
    for k, v in list(kwargs.items())[:3]:  # First 3 kwargs
        summary[k] = _summarize_value(v, max_len)
    return summary


def _summarize_value(val: Any, max_len: int = 50) -> str:
    """Summarize a value for tracing output."""
    if hasattr(val, "shape"):  # Tensor/array
        return f"<{type(val).__name__} shape={val.shape}>"
    if isinstance(val, (list, tuple)):
        return f"<{type(val).__name__} len={len(val)}>"
    if isinstance(val, dict):
        return f"<dict len={len(val)}>"
    s = str(val)
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


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


@contextmanager
def timed_phase(phase_name: str, show_header: bool = True):
    """Context manager for timing a phase with memory tracking.

    Usage:
        with timed_phase("Activation Visualizations"):
            do_work()

    Prints a header with memory before, profiles the block,
    and prints completion with elapsed time and memory delta.
    """
    profile_name = phase_name.lower().replace(" ", "_")
    mem_before = get_memory_mb()

    if show_header:
        print(f"\n{'=' * 60}")
        print(f"  [VIZ] {phase_name}")
        print(f"  Memory before: {mem_before:.1f} MB")
        print("=" * 60)

    t0 = time.time()
    try:
        with profile(profile_name):
            yield
    finally:
        elapsed_ms = (time.time() - t0) * 1000
        mem_after = log_memory(f"after_viz_{profile_name}")
        mem_delta = mem_after - mem_before

        print(f"  -> Completed in {elapsed_ms:.0f}ms")
        print(f"  -> Memory after: {mem_after:.1f} MB (delta: {mem_delta:+.1f} MB)")
