"""Subcircuit filtering and selection functions."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.circuit import Circuit, CircuitStructure
    from src.schemas import IdentifiabilityConstraints, SubcircuitMetrics


def _node_masks_key(circuit: "Circuit") -> tuple:
    """Convert node_masks to a hashable key for grouping by activation pattern."""
    return tuple(tuple(m.tolist()) for m in circuit.node_masks)


def filter_subcircuits(
    constraints: "IdentifiabilityConstraints",
    subcircuit_metrics: list["SubcircuitMetrics"],
    subcircuits: list["Circuit"],
    subcircuit_structures: list["CircuitStructure"],
    max_subcircuits: int = 1,
) -> list[int]:
    """Filter subcircuits by epsilon thresholds, then select diverse top-k.

    Steps:
    1. Filter by bit_similarity and accuracy using epsilon threshold
    2. Sort by (accuracy DESC, bit_similarity DESC, node_sparsity DESC)
    3. Select up to max_subcircuits, diversifying by jaccard distance

    Note: Edge masks are not directly filtered here since circuits with
    the same node pattern but different edges are functionally equivalent
    for initial filtering. Edge exploration happens in a later stage.

    Args:
        constraints: Epsilon thresholds for filtering
        subcircuit_metrics: Per-subcircuit accuracy/similarity metrics
        subcircuits: Circuit objects (for jaccard calculation)
        subcircuit_structures: Circuit structure info (for sparsity)
        max_subcircuits: Maximum number of subcircuits to return

    Returns:
        List of subcircuit indices (up to max_subcircuits), diversified by overlap
    """
    metrics_by_idx = {m.idx: m for m in subcircuit_metrics}

    # First pass: filter by epsilon thresholds
    passing_indices = []
    for result in subcircuit_metrics:
        # Must pass BOTH bit_similarity AND accuracy thresholds
        if 1.0 - result.bit_similarity > constraints.epsilon:
            continue
        if 1.0 - result.accuracy > constraints.epsilon:
            continue
        passing_indices.append(result.idx)

    if not passing_indices:
        return []

    # Sort by quality: (accuracy DESC, bit_similarity DESC, node_sparsity DESC)
    # Higher is better for all metrics
    sorted_indices = sorted(
        passing_indices,
        key=lambda idx: (
            metrics_by_idx[idx].accuracy,
            metrics_by_idx[idx].bit_similarity,
            subcircuit_structures[idx].node_sparsity,
        ),
        reverse=True,
    )

    if max_subcircuits == 1:
        return [sorted_indices[0]]

    # Greedy selection: pick best, then diversify
    # For ties in quality, prefer less overlap with already-selected
    selected = [sorted_indices[0]]

    for candidate_idx in sorted_indices[1:]:
        if len(selected) >= max_subcircuits:
            break

        # Calculate max overlap with any already-selected subcircuit
        # (lower is better - we want diversity)
        max_overlap = max(
            subcircuits[candidate_idx].overlap_jaccard(subcircuits[s]) for s in selected
        )

        # Check if this candidate is a quality tie with the last selected
        candidate = metrics_by_idx[candidate_idx]
        last = metrics_by_idx[selected[-1]]
        is_tie = (
            abs(candidate.accuracy - last.accuracy) < 1e-6
            and abs(candidate.bit_similarity - last.bit_similarity) < 1e-6
        )

        if is_tie:
            # For ties, only add if sufficiently different (jaccard < 0.8)
            if max_overlap < 0.8:
                selected.append(candidate_idx)
        else:
            # Not a tie - just add (it's lower quality but still passes)
            selected.append(candidate_idx)

    return selected
