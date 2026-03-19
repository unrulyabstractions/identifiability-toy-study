"""
Efficient subcircuit enumeration using bitwise operations.

Two-phase approach:
1. enumerate_node_patterns() - Generate all valid node masks (fast: O(product of layer masks))
2. enumerate_edge_configs() - For a given node pattern, enumerate edge configurations
"""

import itertools
from collections.abc import Iterator
from dataclasses import dataclass

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
        return [bin(m).count("1") for m in self.layer_masks]

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
            active = sum(bin(m).count("1") for m in hidden_masks)
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
    require_connectivity: bool = True,
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
        edge_pattern = np.array(
            [(bits >> i) & 1 for i in range(n_edges)], dtype=np.int8
        )
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


# =============================================================================
# Subcircuit Index Utilities
# =============================================================================
# A subcircuit is uniquely identified by (node_mask_idx, edge_variant_rank).
# The flat subcircuit_idx is computed using the architecture (width, depth).


def make_subcircuit_idx(
    width: int, depth: int, node_mask_idx: int, edge_variant_rank: int
) -> int:
    """Create a flat subcircuit index from node pattern and edge variant rank.

    This is the canonical way to create a unique identifier for a subcircuit.
    Use parse_subcircuit_idx() to decompose back to (node_mask_idx, edge_variant_rank).

    IMPORTANT: edge_variant_rank is the OPTIMIZATION RANK (0=best, 1=2nd best, ...),
    NOT the original enumeration index. After edge optimization, variants are sorted
    by metrics, so rank 0 is always the best performing edge configuration.
    With full_edges_only=True, edge_variant_rank is always 0.

    The index space is deterministic based on architecture:
    - max_edge_variants = 2^(width * width * (depth - 1))
    - subcircuit_idx = node_mask_idx * max_edge_variants + edge_variant_rank

    Args:
        width: Width of hidden layers
        depth: Number of hidden layers
        node_mask_idx: Index of the node mask pattern in enumeration order
        edge_variant_rank: Rank of edge variant (0=best, 1=2nd best, ...)

    Returns:
        Flat subcircuit index
    """
    max_edge_variants = 2 ** (width * width * (depth - 1))
    return node_mask_idx * max_edge_variants + edge_variant_rank


def parse_subcircuit_idx(
    width: int, depth: int, subcircuit_idx: int
) -> tuple[int, int]:
    """Extract node pattern and edge variant rank from a flat subcircuit index.

    Args:
        width: Width of hidden layers
        depth: Number of hidden layers
        subcircuit_idx: Flat subcircuit index from make_subcircuit_idx()

    Returns:
        Tuple of (node_mask_idx, edge_variant_rank) where:
        - node_mask_idx: Index of the node pattern in enumeration order
        - edge_variant_rank: Rank of edge variant (0=best, 1=2nd best, ...)
    """
    max_edge_variants = 2 ** (width * width * (depth - 1))
    node_mask_idx = subcircuit_idx // max_edge_variants
    edge_variant_rank = subcircuit_idx % max_edge_variants
    return node_mask_idx, edge_variant_rank


# =============================================================================
# Subcircuit Counting with Connectivity Constraints
# =============================================================================
# A valid subcircuit is an edge subset where:
# - All input nodes have >= 1 outgoing edge
# - The output node has >= 1 incoming edge
# - Each hidden node is either absent (no edges) or present (>= 1 in AND >= 1 out)
#
# The subcircuit index is the decimal interpretation of the binary edge mask.


def f_cover(k: int, a: int) -> int:
    """Covering function: ways for `a` sources to cover all `k` targets.

    Each of `a` source nodes independently picks a NON-EMPTY subset of [k] targets.
    Returns the count where their union covers ALL k targets.

    Formula (inclusion-exclusion):
        f(k, a) = sum_{j=0}^{k} (-1)^{k-j} * C(k,j) * (2^j - 1)^a
    """
    from math import comb

    total = 0
    for j in range(k + 1):
        sign = (-1) ** (k - j)
        total += sign * comb(k, j) * ((2**j - 1) ** a)
    return total


def count_valid_subcircuits(input_size: int, width: int, depth: int) -> int:
    """Count valid subcircuits using DP over active hidden node counts.

    Network: [input_size] -> [width] x depth -> [1]

    By symmetry of hidden nodes, we only track HOW MANY are active (1..W),
    not which ones. Runs in O(W^3 * D) time.

    Args:
        input_size: Number of input nodes (F)
        width: Hidden layer width (W)
        depth: Number of hidden layers (D)

    Returns:
        Total count of valid subcircuits
    """
    from math import comb

    F, W, D = input_size, width, depth

    # First edge-layer: F inputs -> first hidden layer
    # state[a'] = number of edge configs producing a' active nodes in H1
    state = {}
    for a_prime in range(1, W + 1):
        val = comb(W, a_prime) * f_cover(a_prime, F)
        if val > 0:
            state[a_prime] = val

    # Middle edge-layers: hidden -> hidden (D-1 transitions)
    for _ in range(D - 1):
        new_state = {}
        for a, cnt in state.items():
            for a_prime in range(1, W + 1):
                trans = comb(W, a_prime) * f_cover(a_prime, a)
                if trans > 0:
                    new_state[a_prime] = new_state.get(a_prime, 0) + cnt * trans
        state = new_state

    # Last edge-layer: each active hidden node must connect to output
    # Valid iff a >= 1 (already guaranteed by state keys)
    return sum(state.values())


def get_edge_list(input_size: int, width: int, depth: int) -> list[tuple[int, int, int, int]]:
    """Get ordered list of edges for an architecture.

    Returns list of (src_layer, src_idx, dst_layer, dst_idx) tuples.
    Edge index in this list corresponds to bit position in subcircuit index.

    Network: [input_size] -> [width] x depth -> [1]
    """
    layer_sizes = [input_size] + [width] * depth + [1]
    edges = []
    for layer in range(len(layer_sizes) - 1):
        for src in range(layer_sizes[layer]):
            for dst in range(layer_sizes[layer + 1]):
                edges.append((layer, src, layer + 1, dst))
    return edges


def get_num_edges(input_size: int, width: int, depth: int) -> int:
    """Get total number of edges in architecture."""
    # input->hidden + (D-1)*hidden->hidden + hidden->output
    return input_size * width + (depth - 1) * width * width + width


def subcircuit_idx_to_edge_mask(subcircuit_idx: int, num_edges: int) -> list[int]:
    """Convert subcircuit index to binary edge mask.

    Args:
        subcircuit_idx: Decimal index (binary encodes which edges are active)
        num_edges: Total number of edges in architecture

    Returns:
        List of 0/1 values, one per edge
    """
    return [(subcircuit_idx >> i) & 1 for i in range(num_edges)]


def edge_mask_to_subcircuit_idx(edge_mask: list[int]) -> int:
    """Convert binary edge mask to subcircuit index.

    Args:
        edge_mask: List of 0/1 values, one per edge

    Returns:
        Decimal subcircuit index
    """
    idx = 0
    for i, bit in enumerate(edge_mask):
        if bit:
            idx |= (1 << i)
    return idx


def get_circuit_from_trial(
    trial,
    gate_name: str,
    subcircuit_idx: int,
    subcircuits: list = None,
) -> "Circuit":
    """Get a Circuit object from trial data given a subcircuit_idx.

    Looks up the stored circuit in trial.metrics.per_gate_circuits first.
    Falls back to base node pattern (full edges) if not found.

    Args:
        trial: Trial object with metrics
        gate_name: Name of the gate (e.g., "XOR")
        subcircuit_idx: Flat subcircuit index
        subcircuits: Optional list of base circuits (for fallback)

    Returns:
        Circuit object

    Raises:
        ValueError: If circuit cannot be found or reconstructed
    """
    from .circuit import Circuit

    # Get architecture params from trial
    width = trial.setup.architecture.get("width", 4)
    depth = trial.setup.architecture.get("depth", 2)

    # Try to get from stored circuits
    per_gate_circuits = trial.metrics.per_gate_circuits.get(gate_name, {})
    if subcircuit_idx in per_gate_circuits:
        circuit_dict = per_gate_circuits[subcircuit_idx]
        return Circuit.from_dict(circuit_dict)

    # Fallback: use base node pattern from subcircuits list
    if subcircuits is not None:
        node_mask_idx, _ = parse_subcircuit_idx(width, depth, subcircuit_idx)
        if 0 <= node_mask_idx < len(subcircuits):
            return subcircuits[node_mask_idx]

    raise ValueError(
        f"Circuit not found for subcircuit_idx={subcircuit_idx} "
        f"gate={gate_name}"
    )


def reconstruct_circuit_from_idx(
    subcircuit_idx: int,
    width: int,
    depth: int,
    input_size: int = 2,
    output_size: int = 1,
) -> "Circuit":
    """Reconstruct a Circuit by re-enumerating patterns up to the given index.

    This is expensive for large indices. Use get_circuit_from_trial() when
    trial data is available.

    Args:
        subcircuit_idx: Flat subcircuit index
        width: Hidden layer width
        depth: Number of hidden layers
        input_size: Number of inputs
        output_size: Number of outputs

    Returns:
        Circuit object
    """
    from .circuit import Circuit

    node_mask_idx, edge_variant_rank = parse_subcircuit_idx(width, depth, subcircuit_idx)

    # Enumerate node patterns until we find the right one
    layer_widths = [input_size] + [width] * depth + [output_size]
    patterns = list(enumerate_node_patterns(layer_widths))

    if node_mask_idx >= len(patterns):
        raise ValueError(
            f"node_mask_idx={node_mask_idx} exceeds number of patterns "
            f"({len(patterns)})"
        )

    pattern = patterns[node_mask_idx]
    node_arrays = pattern.to_arrays()

    # Enumerate edge configs for this pattern
    edge_configs = list(enumerate_edge_configs(pattern))

    if edge_variant_rank >= len(edge_configs):
        # Fall back to full edges if rank exceeds available configs
        edge_masks = full_edge_config(pattern)
    else:
        edge_masks = edge_configs[edge_variant_rank]

    return Circuit(node_masks=node_arrays, edge_masks=edge_masks)


def compare_circuits(circuit_a: "Circuit", circuit_b: "Circuit") -> dict:
    """Compare two circuits and return detailed metrics.

    Args:
        circuit_a: First circuit
        circuit_b: Second circuit

    Returns:
        Dict with comparison metrics including:
        - jaccard: Jaccard similarity (0-1)
        - a_subset_of_b: True if A is a subset of B
        - b_subset_of_a: True if B is a subset of A
        - intersection_size: Number of shared active components
        - a_only: Components active only in A
        - b_only: Components active only in B
    """
    metrics = circuit_a.overlap_metrics(circuit_b)

    return {
        "jaccard": metrics["jaccard"],
        "a_subset_of_b": metrics["is_subset"],
        "b_subset_of_a": metrics["is_superset"],
        "intersection_size": metrics["intersection_size"],
        "a_size": metrics["self_size"],
        "b_size": metrics["other_size"],
        "a_only": metrics["self_size"] - metrics["intersection_size"],
        "b_only": metrics["other_size"] - metrics["intersection_size"],
    }
