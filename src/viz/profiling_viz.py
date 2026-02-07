"""Profiling visualization.

Contains functions for visualizing profiling data:
- visualize_profiling_timeline: Visualize profiling timeline
- visualize_profiling_phases: Visualize profiling phase durations
- visualize_profiling_summary: Visualize profiling summary
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from ..common.schemas import ProfilingData


def visualize_profiling_timeline(
    profiling: ProfilingData,
    output_dir: str,
    filename: str = "timeline.png",
) -> str | None:
    """Visualize profiling timeline."""
    if not profiling.events:
        return None

    fig, ax = plt.subplots(figsize=(14, 6))

    events = profiling.events
    start_time = events[0].timestamp_ms if events else 0

    # Group by phase
    phases = {}
    for event in events:
        phase = event.status.split(":")[0] if ":" in event.status else event.status
        if phase not in phases:
            phases[phase] = []
        phases[phase].append((event.timestamp_ms - start_time) / 1000.0)

    colors = plt.cm.tab10(np.linspace(0, 1, len(phases)))
    y_pos = 0
    for (phase, times), color in zip(phases.items(), colors):
        for t in times:
            ax.barh(y_pos, 0.1, left=t, color=color, alpha=0.8)
        ax.text(-0.5, y_pos, phase[:20], ha="right", va="center", fontsize=8)
        y_pos += 1

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Phase")
    ax.set_title("Trial Timeline", fontweight="bold")
    ax.set_yticks([])

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def visualize_profiling_phases(
    profiling: ProfilingData,
    output_dir: str,
    filename: str = "phases.png",
) -> str | None:
    """Visualize profiling phase durations."""
    if not profiling.events:
        return None

    # Compute phase durations
    phase_durations = {}
    events = profiling.events
    for i, event in enumerate(events):
        if event.status.startswith("STARTED_"):
            phase = event.status.replace("STARTED_", "")
            end_status = (
                f"ENDED_{phase}"
                if f"ENDED_{phase}" in [e.status for e in events]
                else f"FINISHED_{phase}"
            )
            for j in range(i + 1, len(events)):
                if events[j].status == end_status:
                    duration = (events[j].timestamp_ms - event.timestamp_ms) / 1000.0
                    phase_durations[phase] = phase_durations.get(phase, 0) + duration
                    break

    if not phase_durations:
        return None

    fig, ax = plt.subplots(figsize=(12, 6))
    phases = list(phase_durations.keys())
    durations = [phase_durations[p] for p in phases]

    bars = ax.barh(phases, durations, color="steelblue", alpha=0.8)
    ax.set_xlabel("Duration (s)")
    ax.set_title("Phase Durations", fontweight="bold")

    for bar, d in zip(bars, durations):
        ax.text(
            bar.get_width() + 0.1,
            bar.get_y() + bar.get_height() / 2,
            f"{d:.2f}s",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def visualize_profiling_summary(
    profiling: ProfilingData,
    output_dir: str,
    filename: str = "summary.png",
) -> str | None:
    """Visualize profiling summary."""
    if not profiling.events:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Event count by phase
    ax = axes[0]
    phase_counts = {}
    for event in profiling.events:
        phase = event.status.split(":")[0] if ":" in event.status else event.status
        phase_counts[phase] = phase_counts.get(phase, 0) + 1

    phases = list(phase_counts.keys())[:15]
    counts = [phase_counts[p] for p in phases]
    ax.barh(phases, counts, color="coral", alpha=0.8)
    ax.set_xlabel("Count")
    ax.set_title("Events by Phase", fontweight="bold")

    # Right: Text summary
    ax = axes[1]
    ax.axis("off")
    total_time = (
        (profiling.events[-1].timestamp_ms - profiling.events[0].timestamp_ms) / 1000.0
        if len(profiling.events) > 1
        else 0
    )
    summary_text = f"""
    Total Events: {len(profiling.events)}
    Total Time: {total_time:.2f}s
    Unique Phases: {len(phase_counts)}
    """
    ax.text(
        0.1,
        0.5,
        summary_text.strip(),
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="center",
        fontfamily="monospace",
    )

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path
