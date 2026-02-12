"""Structural faithfulness analysis module.

Computes structural metrics comparing subcircuits to full circuits,
inspired by Zennaro's "Abstraction between Structural Causal Models".

Key concepts:
- Topology: paths, bottlenecks, layer distribution
- Centrality: betweenness, degree, importance
- Structural faithfulness: node/edge retention, path coverage
"""

from collections import defaultdict
from typing import Optional

import networkx as nx
import numpy as np

from src.circuit import Circuit
from src.schemas import (
    AbstractionType,
    CentralityMetrics,
    StructuralAnalysisSummary,
    StructuralFaithfulnessMetrics,
    StructuralFaithfulnessSample,
    StructuralRanking,
    TopologyMetrics,
)


def _build_circuit_graph(circuit: Circuit) -> nx.DiGraph:
    """Build a NetworkX DiGraph from a Circuit's active nodes/edges.

    Nodes are named as "(layer, idx)" strings.
    Only active nodes and edges are included.
    """
    G = nx.DiGraph()

    # Add active nodes
    for layer_idx, node_mask in enumerate(circuit.node_masks):
        for node_idx, active in enumerate(node_mask):
            if active == 1:
                node_name = f"({layer_idx},{node_idx})"
                G.add_node(node_name, layer=layer_idx, idx=node_idx)

    # Add active edges
    for layer_idx, edge_mask in enumerate(circuit.edge_masks):
        for out_idx, row in enumerate(edge_mask):
            for in_idx, active in enumerate(row):
                if active == 1:
                    src = f"({layer_idx},{in_idx})"
                    dst = f"({layer_idx + 1},{out_idx})"
                    # Only add if both nodes exist (are active)
                    if src in G and dst in G:
                        G.add_edge(src, dst)

    return G


def _count_io_paths(G: nx.DiGraph, input_layer: int, output_layer: int) -> tuple[int, list[int]]:
    """Count distinct paths from input layer to output layer.

    Returns (n_paths, path_lengths).
    """
    input_nodes = [n for n, d in G.nodes(data=True) if d.get("layer") == input_layer]
    output_nodes = [n for n, d in G.nodes(data=True) if d.get("layer") == output_layer]

    all_path_lengths = []
    n_paths = 0

    for src in input_nodes:
        for dst in output_nodes:
            try:
                paths = list(nx.all_simple_paths(G, src, dst))
                n_paths += len(paths)
                all_path_lengths.extend(len(p) - 1 for p in paths)  # -1 for edges not nodes
            except nx.NetworkXNoPath:
                continue

    return n_paths, all_path_lengths


def compute_topology_metrics(circuit: Circuit) -> TopologyMetrics:
    """Compute topology metrics for a circuit."""
    G = _build_circuit_graph(circuit)
    n_layers = len(circuit.node_masks)

    # Node counts (hidden layers only: exclude input and output)
    hidden_masks = circuit.node_masks[1:-1]
    n_active_nodes = sum(int(np.sum(m)) for m in hidden_masks)
    n_total_nodes = sum(len(m) for m in hidden_masks)

    # Edge counts
    n_active_edges = sum(int(np.sum(m)) for m in circuit.edge_masks)
    n_total_edges = sum(m.size for m in circuit.edge_masks)

    # Path metrics
    n_paths, path_lengths = _count_io_paths(G, 0, n_layers - 1)

    avg_path_length = float(np.mean(path_lengths)) if path_lengths else 0.0
    max_path_length = max(path_lengths) if path_lengths else 0
    min_path_length = min(path_lengths) if path_lengths else 0

    # Layer-wise analysis
    layer_widths = [int(np.sum(m)) for m in circuit.node_masks]
    layer_totals = [len(m) for m in circuit.node_masks]
    layer_densities = [
        w / t if t > 0 else 0.0 for w, t in zip(layer_widths, layer_totals)
    ]

    # Bottleneck analysis (hidden layers only)
    hidden_widths = layer_widths[1:-1]
    if hidden_widths:
        bottleneck_width = min(hidden_widths)
        bottleneck_layer = hidden_widths.index(bottleneck_width) + 1  # +1 for layer index
        max_width = max(hidden_widths)
        bottleneck_ratio = bottleneck_width / max_width if max_width > 0 else 0.0
    else:
        bottleneck_width = 0
        bottleneck_layer = 0
        bottleneck_ratio = 0.0

    return TopologyMetrics(
        n_active_nodes=n_active_nodes,
        n_total_nodes=n_total_nodes,
        n_active_edges=n_active_edges,
        n_total_edges=n_total_edges,
        n_input_output_paths=n_paths,
        avg_path_length=avg_path_length,
        max_path_length=max_path_length,
        min_path_length=min_path_length,
        bottleneck_width=bottleneck_width,
        bottleneck_layer=bottleneck_layer,
        bottleneck_ratio=bottleneck_ratio,
        layer_widths=layer_widths,
        layer_densities=layer_densities,
    )


def compute_centrality_metrics(circuit: Circuit) -> CentralityMetrics:
    """Compute graph centrality metrics for active nodes."""
    G = _build_circuit_graph(circuit)

    if len(G) == 0:
        return CentralityMetrics()

    # Get hidden nodes only (exclude input layer 0 and output layer)
    n_layers = len(circuit.node_masks)
    hidden_nodes = [
        n for n, d in G.nodes(data=True)
        if 0 < d.get("layer", 0) < n_layers - 1
    ]

    if not hidden_nodes:
        return CentralityMetrics()

    # Betweenness centrality
    try:
        betweenness = nx.betweenness_centrality(G)
        hidden_betweenness = [betweenness.get(n, 0.0) for n in hidden_nodes]
        avg_betweenness = float(np.mean(hidden_betweenness)) if hidden_betweenness else 0.0
        max_betweenness = max(hidden_betweenness) if hidden_betweenness else 0.0
        max_betweenness_node = (
            hidden_nodes[hidden_betweenness.index(max_betweenness)]
            if hidden_betweenness else ""
        )
    except Exception:
        avg_betweenness = 0.0
        max_betweenness = 0.0
        max_betweenness_node = ""

    # Degree centrality
    in_degrees = [G.in_degree(n) for n in hidden_nodes]
    out_degrees = [G.out_degree(n) for n in hidden_nodes]

    avg_in_degree = float(np.mean(in_degrees)) if in_degrees else 0.0
    avg_out_degree = float(np.mean(out_degrees)) if out_degrees else 0.0
    max_in_degree = max(in_degrees) if in_degrees else 0
    max_out_degree = max(out_degrees) if out_degrees else 0

    # PageRank-like importance (reverse graph from outputs)
    try:
        # Compute PageRank on reverse graph to propagate importance from outputs
        importance = nx.pagerank(G.reverse())
        hidden_importance = [importance.get(n, 0.0) for n in hidden_nodes]
        avg_importance = float(np.mean(hidden_importance)) if hidden_importance else 0.0
        max_importance = max(hidden_importance) if hidden_importance else 0.0
        max_importance_node = (
            hidden_nodes[hidden_importance.index(max_importance)]
            if hidden_importance else ""
        )
    except Exception:
        avg_importance = 0.0
        max_importance = 0.0
        max_importance_node = ""

    return CentralityMetrics(
        avg_betweenness=avg_betweenness,
        max_betweenness=max_betweenness,
        max_betweenness_node=max_betweenness_node,
        avg_in_degree=avg_in_degree,
        avg_out_degree=avg_out_degree,
        max_in_degree=max_in_degree,
        max_out_degree=max_out_degree,
        avg_importance=avg_importance,
        max_importance=max_importance,
        max_importance_node=max_importance_node,
    )


def compute_structural_faithfulness(
    subcircuit: Circuit,
    full_circuit: Circuit,
    node_pattern_idx: int = 0,
    edge_variation_idx: int = 0,
) -> StructuralFaithfulnessSample:
    """Compare subcircuit structure to full circuit.

    Implements structural properties from Zennaro's SCM abstraction framework:
    - Node properties: functionality, surjectivity, injectivity
    - Edge properties: faithfulness, fullness
    - Path coverage: fraction of I/O paths preserved
    """
    n_layers = len(subcircuit.node_masks)

    # Node retention (hidden layers only)
    hidden_sub = subcircuit.node_masks[1:-1]
    hidden_full = full_circuit.node_masks[1:-1]

    per_layer_node_retention = []
    total_sub_nodes = 0
    total_full_nodes = 0

    for sub_mask, full_mask in zip(hidden_sub, hidden_full):
        sub_active = int(np.sum(sub_mask))
        full_active = int(np.sum(full_mask))
        retention = sub_active / full_active if full_active > 0 else 0.0
        per_layer_node_retention.append(retention)
        total_sub_nodes += sub_active
        total_full_nodes += full_active

    node_retention_ratio = total_sub_nodes / total_full_nodes if total_full_nodes > 0 else 0.0

    # Edge retention and faithfulness
    per_layer_edge_retention = []
    total_sub_edges = 0
    total_full_edges = 0
    total_edges_between_active = 0  # Max possible given active nodes

    for layer_idx, (sub_edge, full_edge) in enumerate(
        zip(subcircuit.edge_masks, full_circuit.edge_masks)
    ):
        sub_active = int(np.sum(sub_edge))
        full_active = int(np.sum(full_edge))
        retention = sub_active / full_active if full_active > 0 else 0.0
        per_layer_edge_retention.append(retention)
        total_sub_edges += sub_active
        total_full_edges += full_active

        # Compute max edges between active nodes in subcircuit
        active_in = int(np.sum(subcircuit.node_masks[layer_idx]))
        active_out = int(np.sum(subcircuit.node_masks[layer_idx + 1]))
        total_edges_between_active += active_in * active_out

    edge_retention_ratio = total_sub_edges / total_full_edges if total_full_edges > 0 else 0.0

    # Edge faithfulness: active_edges / max_possible_between_active_nodes
    # This is the same as connectivity_density
    edge_faithfulness = (
        total_sub_edges / total_edges_between_active
        if total_edges_between_active > 0 else 0.0
    )

    # Edge induced ratio: what fraction of edge reduction is from node dropping vs explicit
    # If all edges between active nodes are kept, induced_ratio = 1.0
    edge_induced_ratio = edge_faithfulness

    # Path coverage
    G_sub = _build_circuit_graph(subcircuit)
    G_full = _build_circuit_graph(full_circuit)

    n_paths_sub, _ = _count_io_paths(G_sub, 0, n_layers - 1)
    n_paths_full, _ = _count_io_paths(G_full, 0, n_layers - 1)

    path_coverage = n_paths_sub / n_paths_full if n_paths_full > 0 else 0.0

    # Determine abstraction type
    if node_retention_ratio == 1.0 and edge_faithfulness == 1.0:
        abstraction_type = AbstractionType.IDENTITY
    elif node_retention_ratio < 1.0 and edge_faithfulness == 1.0:
        abstraction_type = AbstractionType.NODE_DROPPING
    elif node_retention_ratio == 1.0 and edge_faithfulness < 1.0:
        abstraction_type = AbstractionType.EDGE_DROPPING
    else:
        abstraction_type = AbstractionType.MIXED

    return StructuralFaithfulnessSample(
        node_pattern_idx=node_pattern_idx,
        edge_variation_idx=edge_variation_idx,
        node_functionality=1.0,  # Always 1.0 for masking
        node_surjectivity=1.0,   # Always 1.0 for masking
        node_injectivity=1.0,    # Always 1.0 for masking
        node_retention_ratio=node_retention_ratio,
        edge_faithfulness=edge_faithfulness,
        edge_fullness=1.0,       # Always 1.0 for masking
        edge_retention_ratio=edge_retention_ratio,
        edge_induced_ratio=edge_induced_ratio,
        path_coverage=path_coverage,
        causal_order_preserved=True,  # Always True for feed-forward
        per_layer_node_retention=per_layer_node_retention,
        per_layer_edge_retention=per_layer_edge_retention,
        abstraction_type=abstraction_type,
    )


def _compute_overall_structural_score(
    topology: TopologyMetrics,
    centrality: CentralityMetrics,
    faithfulness: StructuralFaithfulnessSample,
) -> float:
    """Compute overall structural score as weighted combination.

    Weights balance:
    - Path coverage (most important for functional equivalence)
    - Edge faithfulness (structural preservation)
    - Bottleneck ratio (robustness to single-point failures)
    - Low node retention bonus (efficiency)
    """
    # Weighted components
    path_score = faithfulness.path_coverage * 0.35
    edge_score = faithfulness.edge_faithfulness * 0.25
    bottleneck_score = topology.bottleneck_ratio * 0.20

    # Efficiency bonus: reward low retention with high path coverage
    efficiency = (
        faithfulness.path_coverage * (1 - faithfulness.node_retention_ratio)
        if faithfulness.path_coverage > 0 else 0.0
    )
    efficiency_score = efficiency * 0.20

    return path_score + edge_score + bottleneck_score + efficiency_score


def _interpret_structural_metrics(
    topology: TopologyMetrics,
    faithfulness: StructuralFaithfulnessSample,
) -> str:
    """Generate human-readable interpretation of structural metrics."""
    parts = []

    # Path coverage interpretation
    if faithfulness.path_coverage >= 0.9:
        parts.append("Excellent path preservation (>90%)")
    elif faithfulness.path_coverage >= 0.5:
        parts.append(f"Moderate path preservation ({faithfulness.path_coverage:.0%})")
    else:
        parts.append(f"Low path preservation ({faithfulness.path_coverage:.0%})")

    # Node efficiency
    reduction = 1 - faithfulness.node_retention_ratio
    if reduction >= 0.5:
        parts.append(f"High compression ({reduction:.0%} nodes removed)")
    elif reduction >= 0.2:
        parts.append(f"Moderate compression ({reduction:.0%} nodes removed)")
    else:
        parts.append("Low compression")

    # Bottleneck warning
    if topology.bottleneck_ratio < 0.5 and topology.bottleneck_width == 1:
        parts.append("WARNING: Single-node bottleneck")
    elif topology.bottleneck_ratio < 0.5:
        parts.append("Narrow bottleneck detected")

    # Abstraction type
    parts.append(f"Type: {faithfulness.abstraction_type}")

    return "; ".join(parts)


def analyze_subcircuit_structure(
    subcircuit: Circuit,
    full_circuit: Circuit,
    node_pattern_idx: int = 0,
    edge_variation_idx: int = 0,
) -> StructuralFaithfulnessMetrics:
    """Full structural analysis of a subcircuit vs full circuit.

    Returns combined topology, centrality, and faithfulness metrics.
    """
    topology = compute_topology_metrics(subcircuit)
    centrality = compute_centrality_metrics(subcircuit)
    faithfulness = compute_structural_faithfulness(
        subcircuit, full_circuit, node_pattern_idx, edge_variation_idx
    )

    overall = _compute_overall_structural_score(topology, centrality, faithfulness)
    interpretation = _interpret_structural_metrics(topology, faithfulness)

    return StructuralFaithfulnessMetrics(
        node_pattern_idx=node_pattern_idx,
        edge_variation_idx=edge_variation_idx,
        topology=topology,
        centrality=centrality,
        faithfulness=faithfulness,
        overall_structural=overall,
        interpretation=interpretation,
    )


def compute_structural_rankings(
    subcircuits: list[Circuit],
    full_circuit: Circuit,
    subcircuit_indices: Optional[list[tuple[int, int]]] = None,
) -> list[StructuralRanking]:
    """Compute structural rankings for all subcircuits.

    Args:
        subcircuits: List of subcircuits to analyze
        full_circuit: Reference full circuit
        subcircuit_indices: Optional list of (node_pattern_idx, edge_variation_idx) tuples

    Returns:
        Sorted list of StructuralRanking (highest overall_structural first)
    """
    if subcircuit_indices is None:
        subcircuit_indices = [(i, 0) for i in range(len(subcircuits))]

    rankings = []
    for i, subcircuit in enumerate(subcircuits):
        node_idx, edge_idx = subcircuit_indices[i] if i < len(subcircuit_indices) else (i, 0)

        metrics = analyze_subcircuit_structure(
            subcircuit, full_circuit, node_idx, edge_idx
        )

        # Efficiency: high path coverage with low node retention
        efficiency = (
            metrics.faithfulness.path_coverage * (1 - metrics.faithfulness.node_retention_ratio)
        )

        # Robustness: high path coverage, high bottleneck ratio
        robustness = (
            metrics.faithfulness.path_coverage * metrics.topology.bottleneck_ratio
        )

        rankings.append(StructuralRanking(
            node_pattern_idx=node_idx,
            edge_variation_idx=edge_idx,
            overall_structural=metrics.overall_structural,
            path_coverage=metrics.faithfulness.path_coverage,
            node_retention_ratio=metrics.faithfulness.node_retention_ratio,
            edge_faithfulness=metrics.faithfulness.edge_faithfulness,
            bottleneck_ratio=metrics.topology.bottleneck_ratio,
            avg_betweenness=metrics.centrality.avg_betweenness,
            efficiency_score=efficiency,
            robustness_score=robustness,
        ))

    # Sort by overall_structural (descending)
    rankings.sort(key=lambda r: r.overall_structural, reverse=True)
    return rankings


def compute_structural_summary(
    subcircuits: list[Circuit],
    full_circuit: Circuit,
    gate_name: str,
    subcircuit_indices: Optional[list[tuple[int, int]]] = None,
) -> StructuralAnalysisSummary:
    """Compute summary statistics for structural analysis.

    Args:
        subcircuits: List of subcircuits to analyze
        full_circuit: Reference full circuit
        gate_name: Name of the gate being analyzed
        subcircuit_indices: Optional list of (node_pattern_idx, edge_variation_idx) tuples

    Returns:
        StructuralAnalysisSummary with aggregate stats and rankings
    """
    # Full circuit reference stats
    full_topology = compute_topology_metrics(full_circuit)

    # Compute all rankings
    rankings = compute_structural_rankings(subcircuits, full_circuit, subcircuit_indices)

    if not rankings:
        return StructuralAnalysisSummary(
            gate_name=gate_name,
            n_subcircuits_analyzed=0,
            full_circuit_n_paths=full_topology.n_input_output_paths,
            full_circuit_n_nodes=full_topology.n_active_nodes,
            full_circuit_n_edges=full_topology.n_active_edges,
        )

    # Compute averages
    avg_node_retention = float(np.mean([r.node_retention_ratio for r in rankings]))
    avg_edge_retention = float(np.mean([r.edge_faithfulness for r in rankings]))
    avg_path_coverage = float(np.mean([r.path_coverage for r in rankings]))
    avg_bottleneck_ratio = float(np.mean([r.bottleneck_ratio for r in rankings]))

    # Find best by different criteria
    best_by_path = max(rankings, key=lambda r: r.path_coverage)
    best_by_eff = max(rankings, key=lambda r: r.efficiency_score)
    best_by_robust = max(rankings, key=lambda r: r.robustness_score)

    return StructuralAnalysisSummary(
        gate_name=gate_name,
        n_subcircuits_analyzed=len(subcircuits),
        full_circuit_n_paths=full_topology.n_input_output_paths,
        full_circuit_n_nodes=full_topology.n_active_nodes,
        full_circuit_n_edges=full_topology.n_active_edges,
        avg_node_retention=avg_node_retention,
        avg_edge_retention=avg_edge_retention,
        avg_path_coverage=avg_path_coverage,
        avg_bottleneck_ratio=avg_bottleneck_ratio,
        best_by_path_coverage=best_by_path.node_pattern_idx,
        best_by_efficiency=best_by_eff.node_pattern_idx,
        best_by_robustness=best_by_robust.node_pattern_idx,
        rankings=rankings,
    )
