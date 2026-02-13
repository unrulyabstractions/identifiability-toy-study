"""Circuit precomputation utilities for experiments."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from src.infra import profile
from src.schema_class import SchemaClass

from .circuit import enumerate_circuits_for_architecture


@dataclass(frozen=True)
class SubcircuitKey:
    """Canonical key for identifying a subcircuit in trial results.

    A subcircuit is identified by two indices:
    - node_mask_idx: Index of the node pattern in enumeration order
    - edge_variant_rank: Rank of edge variant (0=best) from optimization

    The edge_variant_rank represents the OPTIMIZATION RANK, not an enumeration index.
    After edge optimization, variants are sorted by metrics, so rank 0 is always best.
    With full_edges_only=True, edge_variant_rank is always 0.
    """

    node_mask_idx: int
    edge_variant_rank: int = 0

    def to_flat(self, width: int, depth: int) -> int:
        """Convert to flat index for serialization.

        Args:
            width: Width of hidden layers
            depth: Number of hidden layers

        Returns:
            Flat subcircuit index
        """
        max_edge_variants = 2 ** (width * width * (depth - 1))
        return self.node_mask_idx * max_edge_variants + self.edge_variant_rank

    @classmethod
    def from_flat(cls, flat_idx: int, width: int, depth: int) -> "SubcircuitKey":
        """Parse flat index back to SubcircuitKey.

        Args:
            flat_idx: Flat subcircuit index
            width: Width of hidden layers
            depth: Number of hidden layers

        Returns:
            SubcircuitKey with node_mask_idx and edge_variant_rank
        """
        max_edge_variants = 2 ** (width * width * (depth - 1))
        node_mask_idx = flat_idx // max_edge_variants
        edge_variant_rank = flat_idx % max_edge_variants
        return cls(node_mask_idx, edge_variant_rank)

    def __str__(self) -> str:
        return f"Node#{self.node_mask_idx}/EdgeVariant#{self.edge_variant_rank}"


@dataclass(frozen=True)
class SubcircuitIndex:
    """Index a single maskable node or edge in one of k separated MLPs.

    Architecture: [input_size] + [width]*depth + [1]

    Layers (0-indexed):
        0           = input   (size input_size)
        1 … depth   = hidden  (size width each)
        depth+1     = output  (size 1)

    node_idx (flat int):
        node_idx = hidden_layer * width + neuron
        hidden_layer ∈ [0, depth),  neuron ∈ [0, width)
        Total: depth * width

    edge_idx (flat int), prefix-stable ordering:
        Section 0: hidden→hidden   [(depth-1) * width²]
        Section 1: hidden→output   [width]
        Section 2: input→hidden    [input_size * width]   ← last, for stability
    """

    input_size: int
    width: int
    depth: int
    k: int = 0
    node_idx: Optional[int] = None
    edge_idx: Optional[int] = None

    def __post_init__(self):
        if (self.node_idx is None) == (self.edge_idx is None):
            raise ValueError("Provide exactly one of node_idx or edge_idx")

    # ── section sizes ──────────────────────────────────────────────

    @property
    def n_hidden_nodes(self) -> int:
        return self.depth * self.width

    @property
    def n_hh_edges(self) -> int:
        return max(0, self.depth - 1) * self.width * self.width

    @property
    def n_ho_edges(self) -> int:
        return self.width

    @property
    def n_internal_edges(self) -> int:
        """Edges that don't depend on input_size."""
        return self.n_hh_edges + self.n_ho_edges

    @property
    def n_input_edges(self) -> int:
        return self.input_size * self.width

    @property
    def n_nodes(self) -> int:
        return self.n_hidden_nodes

    @property
    def n_edges(self) -> int:
        return self.n_internal_edges + self.n_input_edges

    @property
    def total(self) -> int:
        return self.n_nodes + self.n_edges

    # ── coordinate helpers ─────────────────────────────────────────

    def node_coords(self) -> Tuple[int, int]:
        """(hidden_layer, neuron) from flat node_idx."""
        assert self.node_idx is not None
        return divmod(self.node_idx, self.width)

    def edge_coords(self) -> Tuple[int, int, int]:
        """(src_layer, src_neuron, dst_neuron) from flat edge_idx.

        src_layer=0        : input → hidden_0
        src_layer=1..dep-1 : hidden_{l-1} → hidden_l
        src_layer=depth    : hidden_{depth-1} → output
        """
        assert self.edge_idx is not None
        idx = self.edge_idx

        # Section 0: h→h
        if idx < self.n_hh_edges:
            group, rem = divmod(idx, self.width * self.width)
            src, dst = divmod(rem, self.width)
            return (group + 1, src, dst)

        idx -= self.n_hh_edges

        # Section 1: h→output
        if idx < self.n_ho_edges:
            return (self.depth, idx, 0)

        idx -= self.n_ho_edges

        # Section 2: input→h
        src, dst = divmod(idx, self.width)
        return (0, src, dst)

    # ── flat subcircuit index ──────────────────────────────────────

    def get_subcircuit_idx(self) -> int:
        """Unified flat index. k is ignored (same mapping for all heads).

        Layout:
            [hidden nodes | h→h edges | h→output edges | input→h edges]
            ←──────────── input_size independent ──────→ ←─ appended ─→
        """
        if self.node_idx is not None:
            return self.node_idx
        return self.n_hidden_nodes + self.edge_idx

    # ── mask → index extraction ────────────────────────────────────

    @staticmethod
    def calculate_index(
        width: int,
        depth: int,
        input_size: int,
        node_mask: List[List[int]],
        edge_mask: List,
    ) -> Tuple[List[int], List[int]]:
        """Extract flat indices where mask == 1.

        Parameters
        ----------
        node_mask : [depth][width]
            node_mask[l][n] — hidden layer l, neuron n.

        edge_mask : [depth+1] groups, ordered by *source* layer:
            edge_mask[0]          : input → h_0,          shape [input_size][width]
            edge_mask[1..depth-1] : h_{l-1} → h_l,        shape [width][width]
            edge_mask[depth]      : h_{depth-1} → output,  shape [width] or [width][1]

        Returns
        -------
        (node_indices, edge_indices) : each a sorted list of flat ints
            Edge indices follow the prefix-stable ordering:
            [h→h | h→output | input→h]
        """
        n_hh = max(0, depth - 1) * width * width

        # ── nodes ──
        node_indices = []
        for l in range(depth):
            for n in range(width):
                if node_mask[l][n]:
                    node_indices.append(l * width + n)

        # ── edges ──
        edge_indices = []

        # Section 0: h→h  (edge_mask groups 1 … depth-1)
        for g in range(1, depth):
            base = (g - 1) * width * width
            for s in range(width):
                for d in range(width):
                    if edge_mask[g][s][d]:
                        edge_indices.append(base + s * width + d)

        # Section 1: h→output  (edge_mask group depth)
        last_group = edge_mask[depth]
        for s in range(width):
            val = last_group[s]
            if isinstance(val, list):
                val = val[0]
            if val:
                edge_indices.append(n_hh + s)

        # Section 2: input→h  (edge_mask group 0, ordered by input neuron)
        for s in range(input_size):
            for d in range(width):
                if edge_mask[0][s][d]:
                    edge_indices.append(n_hh + width + s * width + d)

        return node_indices, edge_indices


class SubcircuitSpecification(SchemaClass):
    index: SubcircuitIndex
    structure: "CircuitStructure"

    # TODO Define useful properties for querying and suck


def compute_node_mask_idx(node_masks: List[List[int]], width: int, depth: int) -> int:
    """Compute flat node_mask_idx from node masks.

    The node_mask_idx uniquely identifies which hidden nodes are active.
    Input and output layers are always fully active and not included.

    Args:
        node_masks: List of masks per layer [input, hidden1, ..., hiddenN, output]
        width: Hidden layer width
        depth: Number of hidden layers

    Returns:
        Flat node_mask_idx
    """
    # Extract hidden layer masks (skip input and output)
    hidden_masks = node_masks[1:-1] if len(node_masks) > 2 else []

    # Convert each hidden layer mask to an integer (bitmask)
    # Then combine into a single index
    idx = 0
    multiplier = 1
    for layer_mask in hidden_masks:
        # Convert layer mask to integer (e.g., [1, 0, 1] -> 5)
        layer_int = sum(bit << i for i, bit in enumerate(layer_mask))
        idx += layer_int * multiplier
        multiplier *= (1 << width)  # 2^width possible masks per layer

    return idx


def compute_edge_mask_idx(
    edge_masks: List, width: int, depth: int, input_size: int
) -> int:
    """Compute flat edge_mask_idx from edge masks.

    The edge_mask_idx uniquely identifies which edges are active.
    Uses prefix-stable ordering: [h→h | h→output | input→h]

    Args:
        edge_masks: List of edge mask matrices per layer transition
        width: Hidden layer width
        depth: Number of hidden layers
        input_size: Number of inputs

    Returns:
        Flat edge_mask_idx
    """
    # Use SubcircuitIndex.calculate_index to get the edge indices
    # Then convert the pattern of active edges to a flat index

    # Get the list of active edge indices
    n_hh = max(0, depth - 1) * width * width
    n_ho = width
    n_ih = input_size * width
    total_edges = n_hh + n_ho + n_ih

    # Build a bitmask of active edges
    edge_bitmask = 0

    # Section 0: h→h (edge_masks groups 1 … depth-1)
    edge_offset = 0
    for g in range(1, depth):
        for s in range(width):
            for d in range(width):
                if g < len(edge_masks) and s < len(edge_masks[g]) and d < len(edge_masks[g][s]):
                    if edge_masks[g][s][d]:
                        edge_bitmask |= (1 << edge_offset)
                edge_offset += 1

    # Section 1: h→output (edge_masks group depth)
    if depth < len(edge_masks):
        last_group = edge_masks[depth]
        for s in range(width):
            if s < len(last_group):
                val = last_group[s]
                if isinstance(val, list):
                    val = val[0] if val else 0
                if val:
                    edge_bitmask |= (1 << edge_offset)
            edge_offset += 1

    # Section 2: input→h (edge_masks group 0)
    if len(edge_masks) > 0:
        for s in range(input_size):
            for d in range(width):
                if s < len(edge_masks[0]) and d < len(edge_masks[0][s]):
                    if edge_masks[0][s][d]:
                        edge_bitmask |= (1 << edge_offset)
                edge_offset += 1

    return edge_bitmask


def edge_mask_idx_to_bitmask(edge_mask_idx: int) -> int:
    """Convert edge_mask_idx back to bitmask (identity for now)."""
    return edge_mask_idx


def get_compatible_edge_masks_for_node(
    node_masks: List[List[int]], width: int, depth: int, input_size: int
) -> int:
    """Get the edge_mask_idx for full connectivity given a node pattern.

    With full_edges_only=True, each node pattern has exactly one compatible
    edge pattern (full connectivity between active nodes).

    Args:
        node_masks: List of masks per layer
        width: Hidden layer width
        depth: Number of hidden layers
        input_size: Number of inputs

    Returns:
        edge_mask_idx for full connectivity between active nodes
    """
    # Build full edge masks from node masks
    edge_masks = []
    for i in range(len(node_masks) - 1):
        src_mask = node_masks[i]
        dst_mask = node_masks[i + 1]
        # Edge is active iff both source and dest nodes are active
        edge_layer = [
            [src_mask[s] * dst_mask[d] for d in range(len(dst_mask))]
            for s in range(len(src_mask))
        ]
        edge_masks.append(edge_layer)

    return compute_edge_mask_idx(edge_masks, width, depth, input_size)


def get_layer_widths(
    input_size: int, width: int, depth: int, output_size: int = 1
) -> list[int]:
    """Compute layer widths for an MLP architecture.

    Args:
        input_size: Number of input features
        width: Hidden layer width
        depth: Number of hidden layers
        output_size: Number of output units (gates). For multi-gate models,
            this should match the number of gates so masks can be properly
            adapted for each gate.
    """
    return [input_size] + [width] * depth + [output_size]


def precompute_circuits_for_architecture(
    width: int,
    depth: int,
    input_size: int,
    logger=None,
) -> tuple[list, list]:
    """Pre-compute circuits for a single architecture.

    Output size is always 1 because we decompose per gate, so each gate's MLP
    has the same number of nodes and edges, keeping indices aligned.

    Args:
        width: Hidden layer width
        depth: Number of hidden layers
        input_size: Number of input features
        logger: Optional logger for progress messages

    Returns:
        Tuple of (subcircuits, subcircuit_structures) where:
        - subcircuits: List of Circuit objects
        - subcircuit_structures: List of structure analysis dicts
    """
    layer_widths = get_layer_widths(input_size, width, depth, output_size=1)
    logger and logger.info(f"Pre-computing circuits for width={width}, depth={depth}")

    with profile("enumerate_circuits"):
        subcircuits = enumerate_circuits_for_architecture(
            layer_widths, min_sparsity=0.0, use_tqdm=True
        )

    with profile("analyze_structures"):
        subcircuit_structures = [s.analyze_structure() for s in subcircuits]

    logger and logger.info(f"  Found {len(subcircuits)} subcircuits")

    return subcircuits, subcircuit_structures
