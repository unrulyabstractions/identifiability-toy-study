"""Test node mask enumeration and circuit generation."""

import pytest
import numpy as np
from identifiability_toy_study.common.neural_model import MLP
from identifiability_toy_study.common.circuit import enumerate_all_valid_circuit, enumerate_edge_variants


# Node masks: 2^(w*d) total combinations for hidden layers
# Valid circuits: (2^w - 1)^d (each hidden layer needs at least 1 active node)


@pytest.mark.parametrize("width,depth", [
    (w, d) for w in range(1, 6) for d in range(1, 5)
    if 2 ** (w * d) <= 100000  # Skip very large cases for speed
])
def test_node_mask_count(width, depth):
    """Verify enumerate_valid_node_masks returns 2^(w*d) masks."""
    hidden_sizes = [width] * depth
    model = MLP(hidden_sizes=hidden_sizes, input_size=2, output_size=1, device="cpu")
    expected = 2 ** (width * depth)

    masks = model.enumerate_valid_node_masks(fix_output=True)

    assert len(masks) == expected, (
        f"width={width}, depth={depth}: expected {expected} masks, got {len(masks)}"
    )


def test_specific_case_w4_d3():
    """Test the specific case mentioned: w=4, d=3 => 2^12 = 4096 masks."""
    model = MLP(hidden_sizes=[4, 4, 4], input_size=2, output_size=1, device="cpu")
    masks = model.enumerate_valid_node_masks(fix_output=True)
    assert len(masks) == 4096, f"Expected 4096 masks for w=4,d=3, got {len(masks)}"


def test_multi_output_same_count():
    """Output size shouldn't affect hidden layer mask count when fix_output=True."""
    model_1 = MLP(hidden_sizes=[4, 4, 4], input_size=2, output_size=1, device="cpu")
    model_2 = MLP(hidden_sizes=[4, 4, 4], input_size=2, output_size=2, device="cpu")

    masks_1 = model_1.enumerate_valid_node_masks(fix_output=True)
    masks_2 = model_2.enumerate_valid_node_masks(fix_output=True)

    assert len(masks_1) == len(masks_2) == 4096


def test_formula_2_power_wd():
    """Directly verify the formula 2^(w*d) for several cases."""
    test_cases = [
        (1, 1, 2),      # 2^1 = 2
        (2, 1, 4),      # 2^2 = 4
        (2, 2, 16),     # 2^4 = 16
        (3, 2, 64),     # 2^6 = 64
        (4, 2, 256),    # 2^8 = 256
        (4, 3, 4096),   # 2^12 = 4096
        (3, 3, 512),    # 2^9 = 512
    ]

    for width, depth, expected in test_cases:
        model = MLP(hidden_sizes=[width] * depth, input_size=2, output_size=1, device="cpu")
        masks = model.enumerate_valid_node_masks(fix_output=True)
        assert len(masks) == expected, (
            f"w={width}, d={depth}: expected 2^{width*depth}={expected}, got {len(masks)}"
        )


@pytest.mark.parametrize("width,depth", [
    (2, 2), (3, 2), (3, 3), (4, 2), (4, 3),
])
def test_valid_circuit_count(width, depth):
    """Verify enumerate_all_valid_circuit returns (2^w - 1)^d valid circuits.

    Valid circuits must have at least 1 active node per hidden layer.
    """
    model = MLP(hidden_sizes=[width] * depth, input_size=2, output_size=1, device="cpu")
    expected = (2**width - 1) ** depth

    circuits = enumerate_all_valid_circuit(model, use_tqdm=False)

    assert len(circuits) == expected, (
        f"w={width}, d={depth}: expected (2^{width}-1)^{depth}={expected}, got {len(circuits)}"
    )


def test_all_circuits_have_valid_connectivity():
    """All enumerated circuits must have at least 1 active node per hidden layer."""
    model = MLP(hidden_sizes=[4, 4, 4], input_size=2, output_size=1, device="cpu")
    circuits = enumerate_all_valid_circuit(model, use_tqdm=False)

    for i, circuit in enumerate(circuits):
        for layer_idx, layer in enumerate(circuit.node_masks[1:-1]):  # Hidden layers
            active_count = np.sum(layer)
            assert active_count > 0, (
                f"Circuit {i} has no active nodes in hidden layer {layer_idx}"
            )


def test_edge_variants_sparse_circuit():
    """Sparse circuits (few active nodes) should have few edge variants."""
    model = MLP(hidden_sizes=[4, 4, 4], input_size=2, output_size=1, device="cpu")
    circuits = enumerate_all_valid_circuit(model, use_tqdm=False)

    # Find a sparse circuit (1 active node per layer)
    sparse = [c for c in circuits if c.sparsity()[0] >= 0.7]
    assert len(sparse) > 0, "Should have sparse circuits"

    edge_variants = enumerate_edge_variants(sparse[0])
    # With 1 active node per layer, only 1 edge variant (the full connection)
    assert len(edge_variants) >= 1


def test_edge_variants_dense_circuit():
    """Dense circuits (many active nodes) can have many edge variants."""
    model = MLP(hidden_sizes=[3, 3], input_size=2, output_size=1, device="cpu")
    circuits = enumerate_all_valid_circuit(model, use_tqdm=False)

    # Find the densest circuit (all nodes active)
    dense = [c for c in circuits if c.sparsity()[0] == 0.0]
    if dense:
        edge_variants = enumerate_edge_variants(dense[0])
        # With all nodes active, many edge combinations
        assert len(edge_variants) > 1
