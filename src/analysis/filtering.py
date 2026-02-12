"""Subcircuit filtering and selection functions."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.circuit import Circuit, CircuitStructure
    from src.experiment_config import IdentifiabilityConstraints
    from src.schemas import SubcircuitMetrics


def _node_masks_key(circuit: "Circuit") -> tuple:
    """Convert node_masks to a hashable key for grouping by activation pattern."""
    return tuple(tuple(m.tolist()) for m in circuit.node_masks)


@dataclass
class FilterResult:
    """Result of subcircuit filtering with metadata."""

    indices: list[int]  # Selected subcircuit indices
    n_passing: int  # Number passing epsilon threshold
    n_total: int  # Total number of subcircuits
    best_metrics: "SubcircuitMetrics | None"  # Metrics of the best passing subcircuit
    best_failing: "SubcircuitMetrics | None"  # Metrics of the best failing subcircuit


def filter_subcircuits(
    constraints: "IdentifiabilityConstraints",
    subcircuit_metrics: list["SubcircuitMetrics"],
    subcircuits: list["Circuit"],
    subcircuit_structures: list["CircuitStructure"],
    min_subcircuits: int = 1,
    max_subcircuits: int = 1,
) -> FilterResult:
    """Filter subcircuits by epsilon thresholds, then select diverse top-k.

    Steps:
    1. Filter by bit_similarity and accuracy using epsilon threshold
    2. Sort by (accuracy DESC, bit_similarity DESC, node_sparsity DESC)
    3. Select between min_subcircuits and max_subcircuits, diversifying by jaccard distance
    4. If fewer than min pass threshold, include best non-passing ones

    Note: Edge masks are not directly filtered here since circuits with
    the same node pattern but different edges are functionally equivalent
    for initial filtering. Edge exploration happens in a later stage.

    Args:
        constraints: Epsilon thresholds for filtering
        subcircuit_metrics: Per-subcircuit accuracy/similarity metrics
        subcircuits: Circuit objects (for jaccard calculation)
        subcircuit_structures: Circuit structure info (for sparsity)
        min_subcircuits: Minimum number of subcircuits to return (if available)
        max_subcircuits: Maximum number of subcircuits to return

    Returns:
        FilterResult with selected indices and metadata
    """
    if not subcircuit_metrics:
        return FilterResult(indices=[], n_passing=0, n_total=0, best_metrics=None, best_failing=None)

    # Ensure at least min_subcircuits (default 1) and respect max
    effective_min = max(1, min_subcircuits)
    effective_max = max(effective_min, max_subcircuits)

    metrics_by_idx = {m.idx: m for m in subcircuit_metrics}
    n_total = len(subcircuit_metrics)

    # First pass: filter by epsilon thresholds
    # Note: NaN comparisons return False, so we explicitly check for NaN to exclude them
    import math
    passing_indices = []
    for result in subcircuit_metrics:
        # Exclude NaN values (they shouldn't pass the filter)
        if math.isnan(result.bit_similarity) or math.isnan(result.accuracy):
            continue
        if 1.0 - result.bit_similarity > constraints.epsilon:
            continue
        if 1.0 - result.accuracy > constraints.epsilon:
            continue
        passing_indices.append(result.idx)

    n_passing = len(passing_indices)
    passing_set = set(passing_indices)

    # Sort ALL by quality for fallback (accuracy DESC, bit_similarity DESC, sparsity DESC)
    all_sorted = sorted(
        [m.idx for m in subcircuit_metrics],
        key=lambda idx: (
            metrics_by_idx[idx].accuracy,
            metrics_by_idx[idx].bit_similarity,
            subcircuit_structures[idx].node_sparsity,
        ),
        reverse=True,
    )

    # Find best failing subcircuit (first in all_sorted that's not in passing)
    best_failing = None
    for idx in all_sorted:
        if idx not in passing_set:
            best_failing = metrics_by_idx[idx]
            break

    # Sort passing by quality
    sorted_passing = sorted(
        passing_indices,
        key=lambda idx: (
            metrics_by_idx[idx].accuracy,
            metrics_by_idx[idx].bit_similarity,
            subcircuit_structures[idx].node_sparsity,
        ),
        reverse=True,
    )

    # Greedy selection from passing indices
    selected = []
    for candidate_idx in sorted_passing:
        if len(selected) >= effective_max:
            break

        if not selected:
            selected.append(candidate_idx)
            continue

        # Calculate max overlap with any already-selected subcircuit
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

    # If we have fewer than effective_min, add best non-passing ones
    if len(selected) < effective_min:
        for idx in all_sorted:
            if len(selected) >= effective_min:
                break
            if idx not in selected:
                selected.append(idx)

    # Handle empty selection (shouldn't happen but be safe)
    if not selected and all_sorted:
        selected = all_sorted[:effective_min]

    return FilterResult(
        indices=selected,
        n_passing=n_passing,
        n_total=n_total,
        best_metrics=metrics_by_idx[selected[0]] if selected else None,
        best_failing=best_failing,
    )
