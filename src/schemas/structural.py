"""Structural faithfulness schema classes.

Inspired by Zennaro's "Abstraction between Structural Causal Models" (2022):
- Functionality, Surjectivity, Injectivity, Bijectivity (node mappings)
- Functoriality, Fullness, Faithfulness (edge mappings)
- Path preservation and causal structure metrics

For subcircuit analysis, we treat:
- Full circuit = base/micro model (M)
- Subcircuit = abstract/macro model (M')
- Mapping tau: nodes_full -> nodes_subcircuit (active nodes map to themselves, inactive dropped)
"""

from dataclasses import dataclass, field
from enum import Enum

from src.schema_class import SchemaClass


class AbstractionType(str, Enum):
    """Types of abstraction relationship between circuits.

    Based on Zennaro's taxonomy of SCM abstractions.
    """
    IDENTITY = "identity"                  # Exact match
    NODE_COARSENING = "node_coarsening"    # Multiple nodes -> one (not applicable for masking)
    NODE_DROPPING = "node_dropping"        # Nodes removed
    EDGE_DROPPING = "edge_dropping"        # Edges removed (beyond node-induced)
    EDGE_COARSENING = "edge_coarsening"    # Multiple edges -> one
    MIXED = "mixed"                        # Combination of above


@dataclass
class TopologyMetrics(SchemaClass):
    """Graph topology metrics for a circuit."""

    # Node counts
    n_active_nodes: int = 0           # Total active hidden nodes
    n_total_nodes: int = 0            # Total hidden nodes in architecture
    n_active_edges: int = 0           # Total active edges
    n_total_edges: int = 0            # Total possible edges

    # Path metrics
    n_input_output_paths: int = 0     # Distinct paths from input to output
    avg_path_length: float = 0.0      # Average path length (input->output)
    max_path_length: int = 0          # Longest path (effective depth)
    min_path_length: int = 0          # Shortest path

    # Bottleneck analysis
    bottleneck_width: int = 0         # Minimum active nodes in any hidden layer
    bottleneck_layer: int = 0         # Which layer is the bottleneck
    bottleneck_ratio: float = 0.0     # bottleneck_width / max_width

    # Layer distribution
    layer_widths: list[int] = field(default_factory=list)      # Active nodes per layer
    layer_densities: list[float] = field(default_factory=list) # Active/total per layer


@dataclass
class CentralityMetrics(SchemaClass):
    """Graph centrality metrics for active nodes."""

    # Betweenness centrality (identifies bottleneck nodes)
    avg_betweenness: float = 0.0      # Mean betweenness of active hidden nodes
    max_betweenness: float = 0.0      # Max betweenness (critical node)
    max_betweenness_node: str = ""    # Node with max betweenness (layer, idx)

    # Degree centrality
    avg_in_degree: float = 0.0        # Mean incoming edges per active hidden node
    avg_out_degree: float = 0.0       # Mean outgoing edges per active hidden node
    max_in_degree: int = 0            # Max incoming edges
    max_out_degree: int = 0           # Max outgoing edges

    # PageRank-like importance (propagated from outputs)
    avg_importance: float = 0.0       # Mean importance score
    max_importance: float = 0.0       # Max importance
    max_importance_node: str = ""     # Most important node


@dataclass
class StructuralFaithfulnessSample(SchemaClass):
    """Single comparison between subcircuit and full circuit structure.

    This is a "sample" in the sense of one subcircuit being analyzed
    against the full circuit reference.
    """

    # Subcircuit identification
    node_pattern_idx: int = 0
    edge_variation_idx: int = 0

    # Node mapping properties (Zennaro Section 3.1)
    node_functionality: float = 1.0     # Active nodes preserve function (always 1.0 for masking)
    node_surjectivity: float = 1.0      # All subcircuit nodes have full-model counterparts (1.0)
    node_injectivity: float = 1.0       # Each full node maps to at most one subcircuit node (1.0)
    node_retention_ratio: float = 0.0   # Fraction of hidden nodes retained

    # Edge mapping properties (Zennaro Section 3.2)
    edge_faithfulness: float = 0.0      # Edges between active nodes preserved
    edge_fullness: float = 1.0          # All subcircuit edges from full model (1.0 for masking)
    edge_retention_ratio: float = 0.0   # Fraction of total edges retained
    edge_induced_ratio: float = 0.0     # Edges induced by node retention vs explicit masking

    # Path preservation (causal structure)
    path_coverage: float = 0.0          # Fraction of I/O paths preserved
    causal_order_preserved: bool = True # Feed-forward order maintained (always True)

    # Layer-wise analysis
    per_layer_node_retention: list[float] = field(default_factory=list)
    per_layer_edge_retention: list[float] = field(default_factory=list)

    # Abstraction classification
    abstraction_type: str = AbstractionType.MIXED


@dataclass
class StructuralFaithfulnessMetrics(SchemaClass):
    """Aggregated structural faithfulness metrics for a subcircuit.

    Combines topology, centrality, and structural abstraction metrics.
    """

    # Subcircuit identification
    node_pattern_idx: int = 0
    edge_variation_idx: int = 0

    # Topology (computed from subcircuit alone)
    topology: TopologyMetrics = field(default_factory=TopologyMetrics)

    # Centrality (computed from subcircuit graph)
    centrality: CentralityMetrics = field(default_factory=CentralityMetrics)

    # Structural faithfulness (comparison to full circuit)
    faithfulness: StructuralFaithfulnessSample = field(default_factory=StructuralFaithfulnessSample)

    # Overall structural score (weighted combination)
    overall_structural: float = 0.0

    # Interpretation
    interpretation: str = ""  # Human-readable assessment


@dataclass
class StructuralRanking(SchemaClass):
    """Ranking entry for structural metrics."""

    node_pattern_idx: int = 0
    edge_variation_idx: int = 0

    # Key metrics for ranking
    overall_structural: float = 0.0
    path_coverage: float = 0.0
    node_retention_ratio: float = 0.0
    edge_faithfulness: float = 0.0
    bottleneck_ratio: float = 0.0
    avg_betweenness: float = 0.0

    # Derived scores
    efficiency_score: float = 0.0       # Low retention, high faithfulness
    robustness_score: float = 0.0       # High path coverage, low bottleneck risk


@dataclass
class StructuralAnalysisSummary(SchemaClass):
    """Summary of structural analysis across all subcircuits for a gate."""

    gate_name: str = ""
    n_subcircuits_analyzed: int = 0

    # Full circuit reference stats
    full_circuit_n_paths: int = 0
    full_circuit_n_nodes: int = 0
    full_circuit_n_edges: int = 0

    # Distribution statistics
    avg_node_retention: float = 0.0
    avg_edge_retention: float = 0.0
    avg_path_coverage: float = 0.0
    avg_bottleneck_ratio: float = 0.0

    # Best subcircuits by structural metrics
    best_by_path_coverage: int = -1           # node_pattern_idx
    best_by_efficiency: int = -1
    best_by_robustness: int = -1

    # Rankings
    rankings: list[StructuralRanking] = field(default_factory=list)
