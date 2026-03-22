from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from src.model import PatchShape

from .enumeration import (
    NodePattern,
    count_node_patterns,
    enumerate_edge_configs,
    enumerate_node_patterns,
    full_edge_config,
)


def _parse_circuit_patches(data) -> list[PatchShape]:
    """Parse in_circuit/out_circuit from dict, handling both old and new formats.

    Old format: single dict or None
    New format: list of dicts
    """
    if data is None:
        return []
    if isinstance(data, list):
        return [PatchShape.from_dict(p) for p in data]
    # Old format: single PatchShape dict
    return [PatchShape.from_dict(data)]


@dataclass
class CircuitStructure:
    """Structure analysis of a circuit.

    Basic metrics:
    - node_sparsity, edge_sparsity: Fraction of inactive components
    - connectivity_density: Active edges / max possible between active nodes

    Topology metrics:
    - n_active_nodes, n_active_edges: Count of active components
    - n_input_output_paths: Distinct paths from input to output
    - avg_path_length, max_path_length: Path statistics
    - bottleneck_width, bottleneck_ratio: Narrowest layer analysis
    - layer_widths: Active nodes per layer [input, h1, ..., output]
    """

    # Basic metrics
    node_sparsity: float
    edge_sparsity: float
    connectivity_density: float  # active_edges / max_possible_edges_between_active_nodes

    # Intervention patches
    in_patches: list[PatchShape]
    out_patches: list[PatchShape]
    in_circuit: list[PatchShape]  # All in-circuit nodes (one PatchShape per layer)
    out_circuit: list[PatchShape]  # All out-circuit nodes (one PatchShape per layer)

    # Architecture info
    input_size: int
    output_size: int
    width: int
    depth: int

    # Extended topology metrics (optional for backwards compat)
    n_active_nodes: int = 0           # Total active hidden nodes
    n_active_edges: int = 0           # Total active edges
    n_input_output_paths: int = 0     # Distinct paths from input to output
    avg_path_length: float = 0.0      # Average path length
    max_path_length: int = 0          # Longest path (effective depth)
    bottleneck_width: int = 0         # Min active nodes in any hidden layer
    bottleneck_layer: int = 0         # Which layer is the bottleneck
    bottleneck_ratio: float = 0.0     # bottleneck_width / max_width
    layer_widths: list[int] = None    # Active nodes per layer
    layer_densities: list[float] = None  # Active/total per layer

    def __post_init__(self):
        """Initialize mutable defaults."""
        if self.layer_widths is None:
            self.layer_widths = []
        if self.layer_densities is None:
            self.layer_densities = []

    def to_dict(self) -> dict:
        """Convert CircuitStructure to a serializable dictionary."""
        return {
            "node_sparsity": self.node_sparsity,
            "edge_sparsity": self.edge_sparsity,
            "connectivity_density": self.connectivity_density,
            "in_patches": [p.to_dict() for p in self.in_patches],
            "out_patches": [p.to_dict() for p in self.out_patches],
            "in_circuit": [p.to_dict() for p in self.in_circuit] if self.in_circuit else [],
            "out_circuit": [p.to_dict() for p in self.out_circuit] if self.out_circuit else [],
            "input_size": self.input_size,
            "output_size": self.output_size,
            "width": self.width,
            "depth": self.depth,
            # Extended topology metrics
            "n_active_nodes": self.n_active_nodes,
            "n_active_edges": self.n_active_edges,
            "n_input_output_paths": self.n_input_output_paths,
            "avg_path_length": self.avg_path_length,
            "max_path_length": self.max_path_length,
            "bottleneck_width": self.bottleneck_width,
            "bottleneck_layer": self.bottleneck_layer,
            "bottleneck_ratio": self.bottleneck_ratio,
            "layer_widths": self.layer_widths,
            "layer_densities": self.layer_densities,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CircuitStructure":
        """Create from dict."""
        return cls(
            node_sparsity=data["node_sparsity"],
            edge_sparsity=data["edge_sparsity"],
            connectivity_density=data.get("connectivity_density", 0.0),
            in_patches=[PatchShape.from_dict(p) for p in data["in_patches"]],
            out_patches=[PatchShape.from_dict(p) for p in data["out_patches"]],
            in_circuit=_parse_circuit_patches(data.get("in_circuit")),
            out_circuit=_parse_circuit_patches(data.get("out_circuit")),
            input_size=data["input_size"],
            output_size=data["output_size"],
            width=data["width"],
            depth=data["depth"],
            # Extended topology metrics (with defaults for old data)
            n_active_nodes=data.get("n_active_nodes", 0),
            n_active_edges=data.get("n_active_edges", 0),
            n_input_output_paths=data.get("n_input_output_paths", 0),
            avg_path_length=data.get("avg_path_length", 0.0),
            max_path_length=data.get("max_path_length", 0),
            bottleneck_width=data.get("bottleneck_width", 0),
            bottleneck_layer=data.get("bottleneck_layer", 0),
            bottleneck_ratio=data.get("bottleneck_ratio", 0.0),
            layer_widths=data.get("layer_widths", []),
            layer_densities=data.get("layer_densities", []),
        )


class Circuit:
    """
    A class representing a circuit of a neural network.

    Attributes:
        node_masks (list of np.ndarray): A list of node masks for each layer.
        edge_masks (list of np.ndarray): A list of edge masks for each layer.
    """

    def __init__(self, node_masks, edge_masks):
        """
        Initializes a Circuit object with node_masks and edge_masks.

        Args:
            node_masks: The node masks for each layer
            edge_masks: The edge masks for each layer
        """
        self.node_masks = node_masks
        self.edge_masks = edge_masks

    def to_dict(self):
        """Convert Circuit to a serializable dictionary."""
        return {
            "node_masks": [mask.tolist() for mask in self.node_masks],
            "edge_masks": [mask.tolist() for mask in self.edge_masks],
        }

    @classmethod
    def from_dict(cls, data):
        """Create a Circuit from a dictionary."""
        node_masks = [np.array(mask) for mask in data["node_masks"]]
        edge_masks = [np.array(mask) for mask in data["edge_masks"]]
        return cls(node_masks, edge_masks)

    def __repr__(self):
        return f"Circuit(node_masks={self.node_masks}, edge_masks={self.edge_masks})"

    @classmethod
    def full(cls, layer_sizes: list[int]) -> "Circuit":
        """Create a full circuit with all nodes and edges active.

        Args:
            layer_sizes: List of layer sizes [input, hidden1, ..., output]

        Returns:
            Circuit with all nodes=1 and all edges=1
        """
        node_masks = [np.ones(size, dtype=np.int8) for size in layer_sizes]
        edge_masks = [
            np.ones((layer_sizes[i + 1], layer_sizes[i]), dtype=np.int8)
            for i in range(len(layer_sizes) - 1)
        ]
        return cls(node_masks=node_masks, edge_masks=edge_masks)

    def visualize(
        self,
        file_path: str = None,
        node_size: str = "medium",
        title: str = None,
        show: bool = False,
    ) -> str | None:
        """Visualize the circuit structure.

        Args:
            file_path: Path to save the figure (if None, returns without saving)
            node_size: Size of nodes ("small", "medium", "large")
            title: Optional title for the plot
            show: Whether to display the plot

        Returns:
            Path to saved figure, or None if not saved
        """
        import matplotlib.pyplot as plt
        import networkx as nx

        size_map = {"small": 200, "medium": 400, "large": 600}
        ns = size_map.get(node_size, 400)

        G = nx.DiGraph()
        pos = {}
        node_colors = []

        # Build graph with positions
        layer_sizes = [len(m) for m in self.node_masks]
        max_width = max(layer_sizes)

        for layer_idx, node_mask in enumerate(self.node_masks):
            n = len(node_mask)
            y_offset = (max_width - n) / 2
            for node_idx in range(n):
                name = f"L{layer_idx}_{node_idx}"
                G.add_node(name)
                pos[name] = (layer_idx, y_offset + node_idx)
                # Active nodes are blue, inactive are gray
                if node_mask[node_idx] == 1:
                    node_colors.append("#4a90d9")
                else:
                    node_colors.append("#d3d3d3")

        # Add edges
        for layer_idx, edge_mask in enumerate(self.edge_masks):
            for out_idx, row in enumerate(edge_mask):
                for in_idx, active in enumerate(row):
                    src = f"L{layer_idx}_{in_idx}"
                    dst = f"L{layer_idx + 1}_{out_idx}"
                    G.add_edge(src, dst, active=active)

        # Draw
        fig, ax = plt.subplots(figsize=(3 + len(layer_sizes), max_width * 0.8))

        # Draw inactive edges first (thin, gray)
        inactive_edges = [(u, v) for u, v, d in G.edges(data=True) if d["active"] == 0]
        nx.draw_networkx_edges(
            G, pos, edgelist=inactive_edges, ax=ax,
            edge_color="#e0e0e0", width=0.5, alpha=0.3, arrows=False
        )

        # Draw active edges (thicker, dark)
        active_edges = [(u, v) for u, v, d in G.edges(data=True) if d["active"] == 1]
        nx.draw_networkx_edges(
            G, pos, edgelist=active_edges, ax=ax,
            edge_color="#333333", width=1.5, alpha=0.8, arrows=False
        )

        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos, ax=ax, node_color=node_colors,
            node_size=ns, edgecolors="black", linewidths=1
        )

        if title:
            ax.set_title(title)
        ax.axis("off")
        plt.tight_layout()

        if file_path:
            import os
            os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else ".", exist_ok=True)
            plt.savefig(file_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return file_path

    def overlap_jaccard(self, other: "Circuit") -> float:
        """Compute Jaccard similarity between this circuit and another.

        Jaccard = |intersection| / |union| for active components.

        Args:
            other: Another Circuit to compare with

        Returns:
            Jaccard similarity score between 0 and 1
        """
        # Combine node masks (excluding input and output layers)
        self_nodes = np.concatenate(self.node_masks[1:-1])
        other_nodes = np.concatenate(other.node_masks[1:-1])

        # Combine edge masks
        self_edges = np.concatenate([m.flatten() for m in self.edge_masks])
        other_edges = np.concatenate([m.flatten() for m in other.edge_masks])

        # Combine all masks
        self_all = np.concatenate([self_nodes, self_edges])
        other_all = np.concatenate([other_nodes, other_edges])

        # Compute Jaccard: intersection / union
        intersection = np.sum((self_all == 1) & (other_all == 1))
        union = np.sum((self_all == 1) | (other_all == 1))

        if union == 0:
            return 1.0  # Both empty = identical
        return float(intersection / union)

    def is_subset_of(self, other: "Circuit") -> bool:
        """Check if this circuit is a subset of another.

        A circuit A is a subset of B if every active component in A
        is also active in B (for both nodes and edges).

        Args:
            other: Another Circuit to compare with

        Returns:
            True if self is a subset of other
        """
        # Check nodes (hidden layers only)
        for self_mask, other_mask in zip(self.node_masks[1:-1], other.node_masks[1:-1]):
            # For each active node in self, it must be active in other
            if not np.all((self_mask == 1) <= (other_mask == 1)):
                return False

        # Check edges
        for self_mask, other_mask in zip(self.edge_masks, other.edge_masks):
            if not np.all((self_mask == 1) <= (other_mask == 1)):
                return False

        return True

    def node_overlap_metrics(self, other: "Circuit") -> dict:
        """Compute overlap metrics based on NODE MASKS ONLY (ignoring edges).

        Use this for comparing node patterns to determine if one is a subset of another.
        This is independent of edge configurations.

        Args:
            other: Another Circuit to compare with

        Returns:
            Dict with jaccard, intersection_size, union_size, self_size, other_size,
            is_subset, is_superset (all based on node masks only)
        """
        # Combine node masks (excluding input and output layers)
        self_nodes = np.concatenate(self.node_masks[1:-1])
        other_nodes = np.concatenate(other.node_masks[1:-1])

        self_active = self_nodes == 1
        other_active = other_nodes == 1

        intersection = int(np.sum(self_active & other_active))
        union = int(np.sum(self_active | other_active))
        self_size = int(np.sum(self_active))
        other_size = int(np.sum(other_active))

        jaccard = intersection / union if union > 0 else 1.0

        return {
            "jaccard": round(jaccard, 4),
            "intersection_size": intersection,
            "union_size": union,
            "self_size": self_size,
            "other_size": other_size,
            "is_subset": intersection == self_size,  # All of self's nodes are in other
            "is_superset": intersection == other_size,  # All of other's nodes are in self
        }

    def overlap_metrics(self, other: "Circuit") -> dict:
        """Compute detailed overlap metrics between two circuits (nodes AND edges).

        Args:
            other: Another Circuit to compare with

        Returns:
            Dict with jaccard, intersection_size, union_size, self_size, other_size,
            is_subset, is_superset
        """
        # Combine node masks (excluding input and output layers)
        self_nodes = np.concatenate(self.node_masks[1:-1])
        other_nodes = np.concatenate(other.node_masks[1:-1])

        # Combine edge masks
        self_edges = np.concatenate([m.flatten() for m in self.edge_masks])
        other_edges = np.concatenate([m.flatten() for m in other.edge_masks])

        # Combine all masks
        self_all = np.concatenate([self_nodes, self_edges])
        other_all = np.concatenate([other_nodes, other_edges])

        self_active = self_all == 1
        other_active = other_all == 1

        intersection = int(np.sum(self_active & other_active))
        union = int(np.sum(self_active | other_active))
        self_size = int(np.sum(self_active))
        other_size = int(np.sum(other_active))

        jaccard = intersection / union if union > 0 else 1.0

        return {
            "jaccard": round(jaccard, 4),
            "intersection_size": intersection,
            "union_size": union,
            "self_size": self_size,
            "other_size": other_size,
            "is_subset": intersection == self_size,  # All of self is in other
            "is_superset": intersection == other_size,  # All of other is in self
        }

    def get_all_possible_intervention_patches(
        self, max_patch_size: int = -1
    ) -> tuple[list[PatchShape], list[PatchShape], PatchShape, PatchShape]:
        """
        Get all possible positions(PatchShape) for interventions.
        Patches returned should modify at most max_patch_size neurons. (-1 return all possible combinations of patches)
        Returns:
            in_patches: Patches that are interventions within circuit (active neurons)
            out_patches: Patches that are interventions outside circuit (inactive neurons)
            in_circuit: Single PatchShape covering all in-circuit hidden neurons
            out_circuit: Single PatchShape covering all out-circuit hidden neurons
        """
        in_patches = []
        out_patches = []

        # Collect all in-circuit and out-circuit neuron indices per layer
        # Skip layer 0 (input) and last layer (output)
        all_in_indices = []
        all_out_indices = []
        hidden_layers = []

        for layer_idx in range(1, len(self.node_masks) - 1):
            node_mask = self.node_masks[layer_idx]
            in_indices = tuple(i for i, active in enumerate(node_mask) if active == 1)
            out_indices = tuple(i for i, active in enumerate(node_mask) if active == 0)

            hidden_layers.append(layer_idx)
            all_in_indices.append(in_indices)
            all_out_indices.append(out_indices)

            # Create per-neuron patches for in-circuit
            for idx in in_indices:
                in_patches.append(
                    PatchShape(layers=(layer_idx,), indices=(idx,), axis="neuron")
                )

            # Create per-neuron patches for out-circuit
            for idx in out_indices:
                out_patches.append(
                    PatchShape(layers=(layer_idx,), indices=(idx,), axis="neuron")
                )

        # If max_patch_size is specified, filter or generate combinations
        if max_patch_size > 0:
            # Filter to patches of at most max_patch_size
            in_patches = [p for p in in_patches if len(p.indices) <= max_patch_size]
            out_patches = [p for p in out_patches if len(p.indices) <= max_patch_size]

        # Create single PatchShape covering all in-circuit neurons per layer
        # We create one per layer since indices differ per layer
        in_circuit_patches = []
        out_circuit_patches = []
        for layer_idx, in_idx, out_idx in zip(
            hidden_layers, all_in_indices, all_out_indices
        ):
            if in_idx:
                in_circuit_patches.append(
                    PatchShape(layers=(layer_idx,), indices=in_idx, axis="neuron")
                )
            if out_idx:
                out_circuit_patches.append(
                    PatchShape(layers=(layer_idx,), indices=out_idx, axis="neuron")
                )

        # in_circuit and out_circuit as lists of PatchShape (one per layer with in-circuit nodes)
        return in_patches, out_patches, in_circuit_patches, out_circuit_patches

    def analyze_structure(self) -> CircuitStructure:
        node_sparsity, edge_sparsity, _ = self.sparsity()

        in_patches, out_patches, in_circuit, out_circuit = (
            self.get_all_possible_intervention_patches()
        )

        # Compute connectivity_density: active_edges / max_possible_edges_between_active_nodes
        # For each layer transition, max edges = (active_nodes_in_layer_i) * (active_nodes_in_layer_i+1)
        total_active_edges = 0
        total_max_possible = 0
        for layer_idx, edge_mask in enumerate(self.edge_masks):
            active_in = int(np.sum(self.node_masks[layer_idx]))
            active_out = int(np.sum(self.node_masks[layer_idx + 1]))
            max_possible = active_in * active_out
            active_edges = int(np.sum(edge_mask))
            total_active_edges += active_edges
            total_max_possible += max_possible

        connectivity_density = (
            total_active_edges / total_max_possible if total_max_possible > 0 else 0.0
        )

        # Extended topology metrics
        # Layer widths and densities
        layer_widths = [int(np.sum(m)) for m in self.node_masks]
        layer_totals = [len(m) for m in self.node_masks]
        layer_densities = [
            w / t if t > 0 else 0.0 for w, t in zip(layer_widths, layer_totals)
        ]

        # Active node count (hidden layers only)
        hidden_widths = layer_widths[1:-1]
        n_active_nodes = sum(hidden_widths)

        # Bottleneck analysis (hidden layers only)
        if hidden_widths:
            bottleneck_width = min(hidden_widths)
            bottleneck_layer = hidden_widths.index(bottleneck_width) + 1  # +1 for layer index
            max_width = max(hidden_widths)
            bottleneck_ratio = bottleneck_width / max_width if max_width > 0 else 0.0
        else:
            bottleneck_width = 0
            bottleneck_layer = 0
            bottleneck_ratio = 0.0

        # Path counting using graph traversal
        n_input_output_paths, path_lengths = self._count_paths()
        avg_path_length = float(np.mean(path_lengths)) if path_lengths else 0.0
        max_path_length = max(path_lengths) if path_lengths else 0

        return CircuitStructure(
            node_sparsity=node_sparsity,
            edge_sparsity=edge_sparsity,
            connectivity_density=connectivity_density,
            in_patches=in_patches,
            out_patches=out_patches,
            in_circuit=in_circuit,
            out_circuit=out_circuit,
            input_size=len(self.node_masks[0]),
            output_size=len(self.node_masks[-1]),
            width=len(self.node_masks[1]),
            depth=len(self.node_masks[1:-1]),
            # Extended topology metrics
            n_active_nodes=n_active_nodes,
            n_active_edges=total_active_edges,
            n_input_output_paths=n_input_output_paths,
            avg_path_length=avg_path_length,
            max_path_length=max_path_length,
            bottleneck_width=bottleneck_width,
            bottleneck_layer=bottleneck_layer,
            bottleneck_ratio=bottleneck_ratio,
            layer_widths=layer_widths,
            layer_densities=layer_densities,
        )

    def _count_paths(self) -> tuple[int, list[int]]:
        """Count distinct paths from input to output using dynamic programming.

        Returns (n_paths, path_lengths).
        """
        n_layers = len(self.node_masks)

        # Build adjacency: for each layer, which outputs can each input reach?
        # path_counts[layer][node] = number of paths from input layer to this node
        # path_lengths[layer][node] = list of path lengths to this node

        from collections import defaultdict

        path_counts = [defaultdict(int) for _ in range(n_layers)]
        path_lengths_per_node = [defaultdict(list) for _ in range(n_layers)]

        # Initialize input layer: each active input has 1 path of length 0
        for i, active in enumerate(self.node_masks[0]):
            if active == 1:
                path_counts[0][i] = 1
                path_lengths_per_node[0][i] = [0]

        # Propagate through layers
        for layer_idx, edge_mask in enumerate(self.edge_masks):
            for out_idx, row in enumerate(edge_mask):
                if self.node_masks[layer_idx + 1][out_idx] == 0:
                    continue  # Skip inactive output nodes
                for in_idx, edge_active in enumerate(row):
                    if edge_active == 1 and path_counts[layer_idx][in_idx] > 0:
                        path_counts[layer_idx + 1][out_idx] += path_counts[layer_idx][in_idx]
                        # Extend path lengths
                        for pl in path_lengths_per_node[layer_idx][in_idx]:
                            path_lengths_per_node[layer_idx + 1][out_idx].append(pl + 1)

        # Sum paths to output layer
        output_layer = n_layers - 1
        n_paths = sum(path_counts[output_layer].values())
        all_lengths = []
        for lengths in path_lengths_per_node[output_layer].values():
            all_lengths.extend(lengths)

        return n_paths, all_lengths

    def sparsity(self):
        """
        Computes the overall sparsity of nodes and edges across all layers.

        Returns:
            tuple: (overall_node_sparsity, overall_edge_sparsity)
                   where each is the fraction of excluded nodes/edges across all layers.
        """
        # Combine all node masks into a single array
        combined_node_mask = np.concatenate(self.node_masks[1:-1], axis=None)

        # Combine all edge masks into a single array
        combined_edge_mask = np.concatenate(self.edge_masks, axis=None)

        # Calculate overall sparsity as the fraction of zero values
        overall_node_sparsity = np.mean(combined_node_mask == 0)
        overall_edge_sparsity = np.mean(combined_edge_mask == 0)

        # Combine node and edge masks into a single array for overall combined sparsity
        combined_mask = np.concatenate([combined_node_mask, combined_edge_mask], axis=0)
        overall_combined_sparsity = np.mean(combined_mask == 0)

        return overall_node_sparsity, overall_edge_sparsity, overall_combined_sparsity


def enumerate_circuits_for_architecture(
    layer_widths: list[int], min_sparsity: float = 0.0, use_tqdm: bool = True
) -> list[Circuit]:
    """
    Enumerate valid node patterns as circuits with full edge connectivity.

    Formula: (2^w - 1)^d valid patterns for d hidden layers of width w.
    Each pattern has at least 1 active node per hidden layer.

    Args:
        layer_widths: List of layer sizes [input_size, hidden1, ..., output_size]
        min_sparsity: Minimum node sparsity (fraction of hidden nodes OFF).
            Default 0.0 includes all valid patterns.
        use_tqdm: Whether to use tqdm to show progress

    Returns:
        List of circuits, one per valid node pattern, with full edges.
    """
    all_circuits = []
    patterns = enumerate_node_patterns(layer_widths, min_sparsity)

    if use_tqdm:
        total = count_node_patterns(layer_widths)
        patterns = tqdm(patterns, total=total, desc="Enumerating node patterns")

    for pattern in patterns:
        node_arrays = pattern.to_arrays()
        edge_masks = full_edge_config(pattern)
        circuit = Circuit(node_masks=node_arrays, edge_masks=edge_masks)
        all_circuits.append(circuit)

    return all_circuits


def enumerate_edge_variants(circuit: Circuit) -> list[Circuit]:
    """
    For a given node pattern, enumerate all valid edge configurations.

    Use after identifying best node patterns via subcircuit metrics.

    Args:
        circuit: A circuit with a specific node pattern

    Returns:
        List of circuits with all valid edge configurations.
    """
    # Convert circuit to NodePattern
    layer_masks = tuple(
        sum(int(b) << i for i, b in enumerate(mask)) for mask in circuit.node_masks
    )
    layer_widths = tuple(len(mask) for mask in circuit.node_masks)
    pattern = NodePattern(layer_masks=layer_masks, layer_widths=layer_widths)

    circuits = []
    for edge_masks in enumerate_edge_configs(pattern):
        new_circuit = Circuit(node_masks=circuit.node_masks, edge_masks=edge_masks)
        circuits.append(new_circuit)

    return circuits
