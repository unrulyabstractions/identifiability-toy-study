"""Circuit precomputation utilities for experiments."""

from itertools import product

from src.infra import profile

from .circuit import enumerate_circuits_for_architecture


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


def precompute_circuits_for_architectures(
    widths: list[int],
    depths: list[int],
    input_size: int,
    output_size: int = 1,
    logger=None,
) -> dict[tuple[int, int], tuple[list, list]]:
    """Pre-compute circuits for all unique (width, depth) combinations.

    Args:
        widths: List of hidden layer widths to enumerate
        depths: List of hidden layer depths to enumerate
        input_size: Number of input features
        output_size: Number of output units (gates). For multi-gate models,
            this should be the maximum number of gates across trials.
        logger: Optional logger for progress messages

    Returns:
        Dict mapping (width, depth) -> (subcircuits, subcircuit_structures)
    """
    circuits_cache = {}

    for width, depth in product(widths, depths):
        key = (width, depth)
        if key in circuits_cache:
            continue

        layer_widths = get_layer_widths(input_size, width, depth, output_size)
        logger and logger.info(
            f"Pre-computing circuits for width={width}, depth={depth}"
        )

        with profile("enumerate_circuits"):
            subcircuits = enumerate_circuits_for_architecture(
                layer_widths, min_sparsity=0.0, use_tqdm=True
            )

        with profile("analyze_structures"):
            subcircuit_structures = [s.analyze_structure() for s in subcircuits]

        circuits_cache[key] = (subcircuits, subcircuit_structures)
        logger and logger.info(f"  Found {len(subcircuits)} subcircuits")

    return circuits_cache
