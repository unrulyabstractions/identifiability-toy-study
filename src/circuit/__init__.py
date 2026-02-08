"""Circuit representation and evaluation.

This module provides circuit-related functionality:
- Circuit: Core circuit class
- NodePattern, enumerate_subcircuits: Subcircuit enumeration utilities
- batched_eval: Batched circuit evaluation
- grounding: Circuit grounding operations
"""

from .circuit import (
    Circuit,
    CircuitStructure,
    compute_jaccard_index,
    enumerate_all_valid_circuit,
    enumerate_edge_variants,
    analyze_circuits,
    find_circuits,
    visualize_circuit_heatmap,
)
from .enumeration import (
    NodePattern,
    enumerate_subcircuits,
    enumerate_subcircuits_with_constraint,
    enumerate_node_patterns,
    enumerate_edge_configs,
    enumerate_circuits,
    count_node_patterns,
    full_edge_config,
    pattern_to_circuit,
)
from .batched_eval import (
    run_batched_node_interventions,
    batch_evaluate_subcircuits,
    batch_compute_metrics,
    batch_evaluate_edge_variants,
    precompute_circuit_masks,
    precompute_circuit_masks_base,
    adapt_masks_for_gate,
)
from .grounding import (
    ground_subcircuit,
    Grounding,
    enumerate_tts,
    compute_local_tts,
    load_circuits,
)

__all__ = [
    # Core
    "Circuit",
    "CircuitStructure",
    "compute_jaccard_index",
    "enumerate_all_valid_circuit",
    "enumerate_edge_variants",
    "analyze_circuits",
    "find_circuits",
    "visualize_circuit_heatmap",
    # Enumeration
    "NodePattern",
    "enumerate_subcircuits",
    "enumerate_subcircuits_with_constraint",
    "enumerate_node_patterns",
    "enumerate_edge_configs",
    "enumerate_circuits",
    "count_node_patterns",
    "full_edge_config",
    "pattern_to_circuit",
    # Batched eval
    "run_batched_node_interventions",
    "batch_evaluate_subcircuits",
    "batch_compute_metrics",
    "batch_evaluate_edge_variants",
    "precompute_circuit_masks",
    "precompute_circuit_masks_base",
    "adapt_masks_for_gate",
    # Grounding
    "ground_subcircuit",
    "Grounding",
    "enumerate_tts",
    "compute_local_tts",
    "load_circuits",
]
