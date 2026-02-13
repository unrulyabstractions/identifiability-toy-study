"""Evaluation schema classes.

Contains metric-related dataclasses:
- SubcircuitMetrics: Metrics for individual subcircuits
- GateMetrics: Metrics for logic gates
- Metrics: Overall trial metrics
- ProfilingEvent: Single profiling event with timing
- ProfilingData: Complete profiling data for a trial
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.schema_class import SchemaClass

# Forward references for circular imports
if TYPE_CHECKING:
    from .faithfulness import FaithfulnessMetrics
    from src.training_analysis.types import TrainingAnalysis


@dataclass
class SubcircuitMetrics(SchemaClass):
    idx: int
    # Simple
    accuracy: float  # to gt, not full circuit (target)

    # Observational
    logit_similarity: float
    bit_similarity: float
    best_similarity: float = 0.0


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


# Type alias for subcircuit keys: flat index from make_subcircuit_idx(node_mask_idx, edge_variant_rank)
# Use parse_subcircuit_idx() to decompose back to (node_mask_idx, edge_variant_rank)
SubcircuitKey = int


@dataclass
class NodeMaskRanking(SchemaClass):
    """Ranking entry for a node pattern."""

    node_mask_idx: int
    rank: int = 0
    # Metrics (optional, populated during ranking)
    avg_accuracy: float | None = None
    avg_bit_similarity: float | None = None
    avg_faithfulness: float | None = None
    n_edge_variants: int = 1


@dataclass
class EdgeMaskRanking(SchemaClass):
    """Ranking entry for an edge pattern."""

    edge_mask_idx: int
    rank: int = 0
    # Metrics (optional, populated during ranking)
    avg_accuracy: float | None = None
    avg_bit_similarity: float | None = None
    avg_faithfulness: float | None = None
    n_compatible_nodes: int = 1


@dataclass
class CompatibilityPair(SchemaClass):
    """A compatible (node_mask_idx, edge_mask_idx) pair."""

    node_mask_idx: int
    edge_mask_idx: int
    subcircuit_idx: int | None = None  # The resulting subcircuit index


@dataclass
class CircuitDiagramsMetadata(SchemaClass):
    """Metadata for circuit diagrams folder."""

    node_rankings: list[NodeMaskRanking] = field(default_factory=list)
    edge_rankings: list[EdgeMaskRanking] = field(default_factory=list)
    compatibility_pairs: list[CompatibilityPair] = field(default_factory=list)
    ranking_metrics: list[str] = field(default_factory=list)
    description: str = ""


@dataclass
class Metrics(SchemaClass):
    # Train info
    avg_loss: float | None = None
    val_acc: float | None = None
    test_acc: float | None = None
    training_analysis: "TrainingAnalysis | None" = None

    # Circuit Info
    per_gate_metrics: dict[str, GateMetrics] = field(default_factory=dict)
    # Keys of subcircuits that produce best results (flat indices from make_subcircuit_idx)
    per_gate_bests: dict[str, list[SubcircuitKey]] = field(
        default_factory=lambda: defaultdict(list)
    )
    # Faithfulness metrics for each best subcircuit (includes observational robustness)
    per_gate_bests_faith: dict[str, list["FaithfulnessMetrics"]] = field(
        default_factory=lambda: defaultdict(list)
    )
    # Edge-masked circuits for each best subcircuit (for decision boundary visualization)
    # Maps gate_name -> {subcircuit_idx -> circuit_dict}
    per_gate_circuits: dict[str, dict[int, dict]] = field(
        default_factory=lambda: defaultdict(dict)
    )
