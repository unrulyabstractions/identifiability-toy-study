"""
Efficient subcircuit enumeration using bitwise operations.

Two-phase approach:
1. enumerate_node_patterns() - Generate all valid node masks (fast: O(product of layer masks))
2. enumerate_edge_configs() - For a given node pattern, enumerate edge configurations
"""

import itertools
from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass
class NodePattern:
    """A node mask pattern for each layer (as integer bitmasks)."""
    layer_masks: tuple[int, ...]  # Bitmask for each layer (1 = active)
    layer_widths: tuple[int, ...]  # Width of each layer

    def to_arrays(self) -> list[np.ndarray]:
        """Convert bitmasks to numpy arrays."""
        return [
            np.array([(mask >> i) & 1 for i in range(width)], dtype=np.int8)
            for mask, width in zip(self.layer_masks, self.layer_widths)
        ]

    def active_counts(self) -> list[int]:
        """Number of active nodes per layer."""
        return [bin(m).count('1') for m in self.layer_masks]

    def sparsity(self) -> float:
        """Fraction of hidden nodes that are OFF (0 = all active, 1 = none active)."""
        total = sum(self.layer_widths[1:-1])  # Hidden layers only
        active = sum(self.active_counts()[1:-1])
        return (total - active) / total if total > 0 else 0.0


def enumerate_node_patterns(
    layer_widths: list[int],
    min_sparsity: float = 0.0,
) -> Iterator[NodePattern]:
    """
    Enumerate all valid node patterns for a network.

    A valid pattern has at least 1 active node per hidden layer.
    Input and output layers are always fully active.

    Formula: (2^w - 1)^d valid patterns for d hidden layers of width w.

    Args:
        layer_widths: [input_size, hidden1, hidden2, ..., output_size]
        min_sparsity: Minimum sparsity to include (0.0 = all patterns)

    Yields:
        NodePattern objects representing valid node configurations.
    """
    if len(layer_widths) < 2:
        return

    input_width = layer_widths[0]
    output_width = layer_widths[-1]
    hidden_widths = layer_widths[1:-1]

    # Input layer: all active
    input_mask = (1 << input_width) - 1  # All bits set

    # Output layer: all active
    output_mask = (1 << output_width) - 1

    if not hidden_widths:
        # No hidden layers - just input->output
        yield NodePattern(
            layer_masks=(input_mask, output_mask),
            layer_widths=tuple(layer_widths),
        )
        return

    # Hidden layers: enumerate all masks with at least 1 bit set
    # For width w: masks are 1, 2, 3, ..., 2^w - 1
    hidden_mask_ranges = [range(1, 1 << w) for w in hidden_widths]

    total_hidden = sum(hidden_widths)

    # Compute full masks for each hidden layer (all nodes active)
    full_hidden_masks = tuple((1 << w) - 1 for w in hidden_widths)

    for hidden_masks in itertools.product(*hidden_mask_ranges):
        # Exclude the full circuit (all hidden nodes active in all layers)
        # A subcircuit must have at least one inactive node somewhere
        if hidden_masks == full_hidden_masks:
            continue

        # Optional sparsity filter
        if min_sparsity > 0:
            active = sum(bin(m).count('1') for m in hidden_masks)
            sparsity = (total_hidden - active) / total_hidden
            if sparsity < min_sparsity:
                continue

        yield NodePattern(
            layer_masks=(input_mask,) + hidden_masks + (output_mask,),
            layer_widths=tuple(layer_widths),
        )


def count_node_patterns(layer_widths: list[int]) -> int:
    """
    Count valid node patterns (subcircuits) without generating them.

    Formula: (2^w1 - 1) * (2^w2 - 1) * ... - 1 for hidden layer widths.
    The -1 excludes the full circuit (all nodes active in all layers).
    """
    hidden_widths = layer_widths[1:-1]
    if not hidden_widths:
        return 0  # No hidden layers = no subcircuits possible

    count = 1
    for w in hidden_widths:
        count *= (1 << w) - 1  # 2^w - 1
    return count - 1  # Exclude the full circuit


def enumerate_edge_configs(
    pattern: NodePattern,
    require_connectivity: bool = True,
) -> Iterator[list[np.ndarray]]:
    """
    Enumerate all valid edge configurations for a node pattern.

    Each active node must have at least one incoming edge (except input)
    and one outgoing edge (except output).

    Args:
        pattern: The node pattern to enumerate edges for
        require_connectivity: If True, enforce that each active node has
            at least one incoming and outgoing edge

    Yields:
        List of edge mask arrays, one per layer transition.
    """
    arrays = pattern.to_arrays()

    # Generate edge masks for each layer transition
    layer_edge_options = []
    for i in range(len(arrays) - 1):
        in_mask = arrays[i]
        out_mask = arrays[i + 1]
        edge_options = _enumerate_layer_edges(in_mask, out_mask, require_connectivity)
        if not edge_options:
            return  # No valid edge configs for this layer
        layer_edge_options.append(edge_options)

    # Product across all layers
    for edge_combo in itertools.product(*layer_edge_options):
        yield list(edge_combo)


def _enumerate_layer_edges(
    in_mask: np.ndarray,
    out_mask: np.ndarray,
    require_connectivity: bool,
) -> list[np.ndarray]:
    """
    Enumerate valid edge masks between two layers.

    Args:
        in_mask: Binary mask for input layer (1 = active)
        out_mask: Binary mask for output layer (1 = active)
        require_connectivity: If True, each active node needs connections

    Returns:
        List of valid edge mask arrays (shape: [out_size, in_size])
    """
    in_active = np.where(in_mask == 1)[0]
    out_active = np.where(out_mask == 1)[0]

    if len(in_active) == 0 or len(out_active) == 0:
        return []

    n_in = len(in_active)
    n_out = len(out_active)
    n_edges = n_in * n_out

    # Base edge mask (all inactive)
    base = np.zeros((len(out_mask), len(in_mask)), dtype=np.int8)

    if n_edges > 20:  # Too many combinations, just return full connectivity
        full = base.copy()
        full[np.ix_(out_active, in_active)] = 1
        return [full]

    valid_masks = []

    # Enumerate all 2^(n_edges) combinations
    for bits in range(1, 1 << n_edges):  # Skip all-zeros
        # Decode bits to edge pattern
        edge_pattern = np.array([(bits >> i) & 1 for i in range(n_edges)], dtype=np.int8)
        edge_pattern = edge_pattern.reshape(n_out, n_in)

        if require_connectivity:
            # Each output node needs at least one incoming edge
            if not np.all(edge_pattern.sum(axis=1) > 0):
                continue
            # Each input node needs at least one outgoing edge
            if not np.all(edge_pattern.sum(axis=0) > 0):
                continue

        # Build full edge mask
        mask = base.copy()
        mask[np.ix_(out_active, in_active)] = edge_pattern
        valid_masks.append(mask)

    return valid_masks


def full_edge_config(pattern: NodePattern) -> list[np.ndarray]:
    """
    Create a full-connectivity edge config for a node pattern.

    All edges between active nodes are enabled.
    """
    arrays = pattern.to_arrays()
    edge_masks = []
    for i in range(len(arrays) - 1):
        # Edge active iff both source and dest are active
        edge_mask = np.outer(arrays[i + 1], arrays[i]).astype(np.int8)
        edge_masks.append(edge_mask)
    return edge_masks


def pattern_to_circuit(pattern: NodePattern, edge_masks: list[np.ndarray] = None):
    """
    Convert a NodePattern to a Circuit object.

    Args:
        pattern: The node pattern
        edge_masks: Edge configuration (defaults to full connectivity)

    Returns:
        Circuit object
    """
    from .circuit import Circuit

    node_arrays = pattern.to_arrays()
    if edge_masks is None:
        edge_masks = full_edge_config(pattern)

    return Circuit(node_masks=node_arrays, edge_masks=edge_masks)


def enumerate_circuits(
    layer_widths: list[int],
    min_sparsity: float = 0.0,
    full_edges_only: bool = True,
):
    """
    Enumerate circuits for a network architecture.

    Args:
        layer_widths: [input_size, hidden1, ..., output_size]
        min_sparsity: Minimum node sparsity to include
        full_edges_only: If True, only return circuits with full edge connectivity
            (fast). If False, enumerate all edge configurations (can be slow).

    Yields:
        Circuit objects
    """
    from .circuit import Circuit

    for pattern in enumerate_node_patterns(layer_widths, min_sparsity):
        node_arrays = pattern.to_arrays()

        if full_edges_only:
            edge_masks = full_edge_config(pattern)
            yield Circuit(node_masks=node_arrays, edge_masks=edge_masks)
        else:
            for edge_masks in enumerate_edge_configs(pattern):
                yield Circuit(node_masks=node_arrays, edge_masks=edge_masks)


def enumerate_subcircuits(layer_widths: list[int], min_sparsity: float = 0.0):
    """Alias for enumerate_circuits with full edges only."""
    return enumerate_circuits(layer_widths, min_sparsity, full_edges_only=True)


def enumerate_subcircuits_with_constraint(
    layer_widths: list[int],
    min_sparsity: float = 0.0,
    require_connectivity: bool = True,
):
    """Enumerate subcircuits with full edge configs."""
    return enumerate_circuits(layer_widths, min_sparsity, full_edges_only=True)
