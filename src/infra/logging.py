"""Logging utilities for detailed pipeline output.

Provides structured logging functions that format data cleanly without
cluttering the main code with print logic.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.schemas import FaithfulnessMetrics, SubcircuitMetrics


def log_section(title: str, width: int = 60) -> None:
    """Print a section header."""
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")


def log_subsection(title: str) -> None:
    """Print a subsection header."""
    print(f"\n  ┌─ {title}")


def log_kv(key: str, value: str, indent: int = 4) -> None:
    """Print a key-value pair."""
    prefix = " " * indent
    print(f"{prefix}{key}: {value}")


def log_subcircuit_metrics(
    label: str,
    metrics: "SubcircuitMetrics",
    indent: int = 4,
) -> None:
    """Log subcircuit metrics in a compact format."""
    prefix = " " * indent
    print(
        f"{prefix}{label} (idx={metrics.idx}): "
        f"acc={metrics.accuracy:.2%}  "
        f"bit_sim={metrics.bit_similarity:.2%}  "
        f"best_sim={metrics.best_similarity:.2%}"
    )


def log_filtering_result(
    gate_name: str,
    gate_acc: float,
    n_passing: int,
    n_total: int,
    epsilon: float,
    selected_indices: list[int],
    best_passing: "SubcircuitMetrics | None",
    best_failing: "SubcircuitMetrics | None",
) -> None:
    """Log comprehensive filtering results for a gate."""
    n_selected = len(selected_indices)

    # Status indicator
    if n_passing == 0:
        status = "⚠ FALLBACK"
    elif n_passing < n_selected:
        status = "◐ PARTIAL"
    else:
        status = "✓ PASS"

    print(f"\n  {gate_name} Summary:")
    print(f"    Gate accuracy:     {gate_acc:.2%}")
    print(f"    Subcircuits:       {n_passing}/{n_total} pass ε={epsilon}")
    print(f"    Selected:          {n_selected} [{status}]")

    if best_passing:
        log_subcircuit_metrics("Best passing", best_passing)

    if best_failing and n_passing < n_total:
        log_subcircuit_metrics("Best failing", best_failing)


def log_edge_variant_summary(
    orig_idx: int,
    n_variants: int,
    best_acc: float,
    best_bit_sim: float,
    worst_acc: float,
    worst_bit_sim: float,
) -> None:
    """Log edge variant optimization results for a subcircuit."""
    print(
        f"    SC #{orig_idx}: {n_variants} variants  "
        f"best=(acc={best_acc:.2%}, sim={best_bit_sim:.2%})  "
        f"worst=(acc={worst_acc:.2%}, sim={worst_bit_sim:.2%})"
    )


def parse_subcircuit_key(key: int, width: int, depth: int) -> tuple[int, int]:
    """Parse a flat subcircuit index into (node_mask_idx, edge_variant_rank).

    Args:
        key: Flat subcircuit index from make_subcircuit_idx()
        width: Width of hidden layers
        depth: Number of hidden layers

    Returns:
        Tuple of (node_mask_idx, edge_variant_rank) where edge_variant_rank
        is the optimization rank (0=best, 1=2nd best, ...)
    """
    from src.circuit import parse_subcircuit_idx

    return parse_subcircuit_idx(width, depth, key)


def format_subcircuit_key(key: int, width: int, depth: int) -> str:
    """Format a subcircuit key for display."""
    node_mask_idx, edge_variant_rank = parse_subcircuit_key(key, width, depth)
    return f"Node#{node_mask_idx}/EdgeVariant#{edge_variant_rank}"


def log_faithfulness_metrics(
    subcircuit_key: int,
    faith: "FaithfulnessMetrics",
    width: int,
    depth: int,
    indent: int = 4,
) -> None:
    """Log faithfulness metrics for a subcircuit."""
    prefix = " " * indent
    obs = faith.observational
    inter = faith.interventional
    cf = faith.counterfactual

    key_str = format_subcircuit_key(subcircuit_key, width, depth)
    print(f"{prefix}{key_str} Faithfulness:")

    # Observational
    if obs:
        noise = obs.noise
        if noise:
            sim = noise.similarity
            print(
                f"{prefix}  Observational: {obs.overall_observational:.3f}  "
                f"(noise: bit_sim={sim.bit:.3f}, sc_acc={noise.subcircuit_accuracy:.2%})"
            )
        else:
            print(f"{prefix}  Observational: {obs.overall_observational:.3f}")

    # Interventional
    if inter:
        print(
            f"{prefix}  Interventional: {inter.overall_interventional:.3f}  "
            f"(in={inter.mean_in_circuit_similarity:.3f}, "
            f"out={inter.mean_out_circuit_similarity:.3f})"
        )

    # Counterfactual
    if cf:
        print(
            f"{prefix}  Counterfactual: {cf.overall_counterfactual:.3f}  "
            f"(suff={cf.mean_sufficiency:.3f}, comp={cf.mean_completeness:.3f}, "
            f"nec={cf.mean_necessity:.3f}, ind={cf.mean_independence:.3f})"
        )

    # Overall
    print(f"{prefix}  Overall: {faith.overall_faithfulness:.3f}")


def log_gate_faithfulness_summary(
    gate_name: str,
    subcircuit_indices: list[int],
    faithfulness_results: list["FaithfulnessMetrics"],
    width: int,
    depth: int,
) -> None:
    """Log faithfulness summary for all subcircuits of a gate."""
    if not faithfulness_results:
        return

    print(f"\n  {gate_name} Faithfulness Results:")
    for idx, faith in zip(subcircuit_indices, faithfulness_results):
        log_faithfulness_metrics(idx, faith, width, depth)
