"""Metrics schema classes.

Contains metric-related dataclasses:
- SubcircuitMetrics: Metrics for individual subcircuits
- GateMetrics: Metrics for logic gates
- Metrics: Overall trial metrics (aliased as TrialMetrics for backwards compat)
- ProfilingEvent: Single profiling event with timing
- ProfilingData: Complete profiling data for a trial
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .base import SchemaClass

# Forward references for circular imports
if TYPE_CHECKING:
    from .faithfulness import FaithfulnessMetrics, RobustnessMetrics


@dataclass
class SubcircuitMetrics(SchemaClass):
    idx: int
    # Simple
    accuracy: float  # to gt, not full circuit (target)

    # Observational
    logit_similarity: float
    bit_similarity: float
    best_similarity: float = 0.0  # After clamping to binary [0,1]


@dataclass
class GateMetrics(SchemaClass):
    test_acc: float
    subcircuit_metrics: list[SubcircuitMetrics] = field(default_factory=list)


@dataclass
class ProfilingEvent(SchemaClass):
    """A single profiling event with timing info."""

    status: str
    timestamp_ms: float  # Milliseconds since trial start
    elapsed_ms: float  # Milliseconds since last event


@dataclass
class ProfilingData(SchemaClass):
    """Profiling data for a trial."""

    device: str = "cpu"
    start_time_ms: float = 0.0  # Unix timestamp in ms when trial started
    total_duration_ms: float = 0.0  # Total trial duration in ms
    events: list[ProfilingEvent] = field(default_factory=list)

    # Aggregated phase durations (computed from events)
    phase_durations_ms: dict[str, float] = field(default_factory=dict)


@dataclass
class Metrics(SchemaClass):
    # Train info
    avg_loss: float | None = None
    val_acc: float | None = None
    test_acc: float | None = None

    # Circuit Info
    per_gate_metrics: dict[str, GateMetrics] = field(default_factory=dict)
    # Index of subcircuits that produces best result
    per_gate_bests: dict[str, list[int]] = field(
        default_factory=lambda: defaultdict(list)
    )
    per_gate_bests_robust: dict[str, list["RobustnessMetrics"]] = field(
        default_factory=lambda: defaultdict(list)
    )
    per_gate_bests_faith: dict[str, list["FaithfulnessMetrics"]] = field(
        default_factory=lambda: defaultdict(list)
    )


# Alias for backwards compatibility - Metrics is sometimes referred to as TrialMetrics
TrialMetrics = Metrics
