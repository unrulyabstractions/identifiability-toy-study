"""Status tracking utilities for trial execution."""

import time
from typing import Optional

from src.schemas import ProfilingEvent, TrialResult


def update_status_fx(trial_result: TrialResult, logger=None, device: str = "cpu"):
    """Create a status updater that tracks timing in trial_result.profiling."""
    start_time_ms = time.time() * 1000
    last_time_ms = start_time_ms

    # Initialize profiling
    trial_result.profiling.device = device
    trial_result.profiling.start_time_ms = start_time_ms

    def update_status(status: str, mssg: Optional[str] = None):
        nonlocal last_time_ms

        trial_result.status = status

        # Record timing
        current_time_ms = time.time() * 1000
        timestamp_ms = current_time_ms - start_time_ms
        elapsed_ms = current_time_ms - last_time_ms
        last_time_ms = current_time_ms

        # Add event
        event = ProfilingEvent(
            status=status,
            timestamp_ms=round(timestamp_ms, 2),
            elapsed_ms=round(elapsed_ms, 2),
        )
        trial_result.profiling.events.append(event)

        # Update total duration
        trial_result.profiling.total_duration_ms = round(timestamp_ms, 2)

        # Aggregate phase durations for STARTED_*/ENDED_* or STARTED_*/FINISHED_* pairs
        if status.startswith("ENDED_") or status.startswith("FINISHED_"):
            # Find matching start event
            phase_prefix = status.split("_", 1)[1]  # e.g., "MLP_TRAINING" or "GATE:0"
            for prev_event in reversed(trial_result.profiling.events[:-1]):
                if (
                    prev_event.status.startswith("STARTED_")
                    and phase_prefix in prev_event.status
                ):
                    phase_name = phase_prefix
                    phase_duration = timestamp_ms - prev_event.timestamp_ms
                    trial_result.profiling.phase_durations_ms[phase_name] = round(
                        phase_duration, 2
                    )
                    break

        if logger:
            if mssg:
                logger.info(
                    f" trial_id:{trial_result.trial_id} trial_status:{status} "
                    f"[{elapsed_ms:.0f}ms] mssg:{mssg}"
                )
            else:
                logger.info(
                    f" trial_id:{trial_result.trial_id} trial_status:{status} "
                    f"[{elapsed_ms:.0f}ms]"
                )

    return update_status
