"""Test batched evaluation correctness and performance."""

import pytest
import torch
import numpy as np
import time
from src.common.neural_model import MLP
from src.common.circuit import enumerate_all_valid_circuit
from src.common.batched_eval import (
    adapt_masks_for_gate,
    batch_evaluate_edge_variants,
    batch_evaluate_subcircuits,
    batch_compute_metrics,
    precompute_circuit_masks,
    precompute_circuit_masks_base,
)


def sequential_evaluate(model, circuits, x, gate_idx=0):
    """Sequential evaluation for comparison."""
    results = []
    with torch.no_grad():
        for circuit in circuits:
            intervention = circuit.to_intervention(model.device)
            y = model(x, intervention=intervention)
            if y.shape[-1] > 1:
                y = y[:, gate_idx:gate_idx+1]
            results.append(y.cpu().numpy())
    return np.stack(results)


@pytest.fixture
def model_and_circuits():
    """Create a model and enumerate circuits."""
    model = MLP(hidden_sizes=[4, 4], input_size=2, output_size=1, device="cpu")
    circuits = enumerate_all_valid_circuit(model, use_tqdm=False)
    return model, circuits


def test_batch_evaluate_matches_sequential(model_and_circuits):
    """Verify batched evaluation produces same results as sequential."""
    model, circuits = model_and_circuits
    x = torch.randn(10, 2)

    # Sequential evaluation
    seq_results = sequential_evaluate(model, circuits, x, gate_idx=0)

    # Batched evaluation
    batch_results = batch_evaluate_subcircuits(model, circuits, x, gate_idx=0)
    batch_results = batch_results.cpu().numpy()

    # Compare
    np.testing.assert_allclose(
        batch_results, seq_results,
        rtol=1e-5, atol=1e-5,
        err_msg="Batched and sequential results differ"
    )


def test_batch_compute_metrics_shapes(model_and_circuits):
    """Verify batch_compute_metrics returns correct shapes."""
    model, circuits = model_and_circuits
    x = torch.randn(10, 2)
    y_target = torch.randint(0, 2, (10, 1)).float()

    with torch.no_grad():
        y_pred = model(x)

    accs, logit_sims, bit_sims, best_sims = batch_compute_metrics(
        model, circuits, x, y_target, y_pred, gate_idx=0
    )

    n_circuits = len(circuits)
    assert accs.shape == (n_circuits,)
    assert logit_sims.shape == (n_circuits,)
    assert bit_sims.shape == (n_circuits,)
    assert best_sims.shape == (n_circuits,)


def test_batch_compute_metrics_range(model_and_circuits):
    """Verify metrics are in valid ranges."""
    model, circuits = model_and_circuits
    x = torch.randn(20, 2)
    y_target = torch.randint(0, 2, (20, 1)).float()

    with torch.no_grad():
        y_pred = model(x)

    accs, logit_sims, bit_sims, best_sims = batch_compute_metrics(
        model, circuits, x, y_target, y_pred, gate_idx=0
    )

    # Accuracy and bit_similarity should be in [0, 1]
    assert np.all(accs >= 0) and np.all(accs <= 1)
    assert np.all(bit_sims >= 0) and np.all(bit_sims <= 1)
    assert np.all(best_sims >= 0) and np.all(best_sims <= 1)
    # Logit similarity can be negative if MSE > 1


@pytest.mark.parametrize("width,depth", [(3, 2), (4, 2)])
def test_batch_speedup(width, depth):
    """Measure speedup from batched evaluation."""
    model = MLP(hidden_sizes=[width] * depth, input_size=2, output_size=1, device="cpu")
    circuits = enumerate_all_valid_circuit(model, use_tqdm=False)
    x = torch.randn(100, 2)

    # Warmup
    _ = batch_evaluate_subcircuits(model, circuits[:10], x)

    # Time sequential
    start = time.time()
    for _ in range(3):
        _ = sequential_evaluate(model, circuits, x)
    seq_time = (time.time() - start) / 3

    # Time batched
    start = time.time()
    for _ in range(3):
        _ = batch_evaluate_subcircuits(model, circuits, x)
    batch_time = (time.time() - start) / 3

    speedup = seq_time / batch_time if batch_time > 0 else float('inf')
    print(f"\nw={width}, d={depth}: {len(circuits)} circuits")
    print(f"  Sequential: {seq_time*1000:.1f}ms")
    print(f"  Batched: {batch_time*1000:.1f}ms")
    print(f"  Speedup: {speedup:.1f}x")

    # Batched should be at least as fast (allow some variance)
    assert batch_time <= seq_time * 1.5, f"Batched slower than expected: {batch_time} vs {seq_time}"


def test_multi_output_model():
    """Test with multi-output model."""
    model = MLP(hidden_sizes=[3, 3], input_size=2, output_size=2, device="cpu")
    circuits = enumerate_all_valid_circuit(model, use_tqdm=False)
    x = torch.randn(10, 2)

    # Test gate 0
    result0 = batch_evaluate_subcircuits(model, circuits, x, gate_idx=0)
    assert result0.shape == (len(circuits), 10, 1)

    # Test gate 1
    result1 = batch_evaluate_subcircuits(model, circuits, x, gate_idx=1)
    assert result1.shape == (len(circuits), 10, 1)

    # Results should differ
    assert not torch.allclose(result0, result1)


def test_precomputed_masks_match_without():
    """Verify precomputed masks give same results as without."""
    model = MLP(hidden_sizes=[4, 4], input_size=2, output_size=1, device="cpu")
    circuits = enumerate_all_valid_circuit(model, use_tqdm=False)
    x = torch.randn(20, 2)

    # Without precomputed
    result_no_precompute = batch_evaluate_subcircuits(model, circuits, x, gate_idx=0)

    # With precomputed
    precomputed = precompute_circuit_masks(circuits, len(model.layers), gate_idx=0, device="cpu")
    result_with_precompute = batch_evaluate_subcircuits(
        model, circuits, x, gate_idx=0, precomputed_masks=precomputed
    )

    # Should match exactly
    np.testing.assert_allclose(
        result_with_precompute.cpu().numpy(),
        result_no_precompute.cpu().numpy(),
        rtol=1e-5, atol=1e-5,
    )


def test_precomputed_masks_multi_gate():
    """Test precomputed masks work correctly with multi-output models."""
    model = MLP(hidden_sizes=[3, 3], input_size=2, output_size=2, device="cpu")
    circuits = enumerate_all_valid_circuit(model, use_tqdm=False)
    x = torch.randn(10, 2)

    # Precompute for gate 0
    precomputed_g0 = precompute_circuit_masks(circuits, len(model.layers), gate_idx=0, device="cpu")
    result_g0 = batch_evaluate_subcircuits(model, circuits, x, gate_idx=0, precomputed_masks=precomputed_g0)

    # Precompute for gate 1
    precomputed_g1 = precompute_circuit_masks(circuits, len(model.layers), gate_idx=1, device="cpu")
    result_g1 = batch_evaluate_subcircuits(model, circuits, x, gate_idx=1, precomputed_masks=precomputed_g1)

    # Results should differ
    assert not torch.allclose(result_g0, result_g1)


def test_eval_device_parameter():
    """Test that eval_device parameter works correctly."""
    model = MLP(hidden_sizes=[3, 3], input_size=2, output_size=1, device="cpu")
    circuits = enumerate_all_valid_circuit(model, use_tqdm=False)
    x_cpu = torch.randn(10, 2)

    # Evaluate on CPU
    result_cpu = batch_evaluate_subcircuits(model, circuits, x_cpu, eval_device="cpu")

    # If MPS available, test MPS evaluation
    if torch.backends.mps.is_available():
        result_mps = batch_evaluate_subcircuits(model, circuits, x_cpu, eval_device="mps")
        # Results should be on MPS device
        assert result_mps.device.type == "mps"
        # Results should match (within tolerance)
        np.testing.assert_allclose(
            result_mps.cpu().numpy(),
            result_cpu.numpy(),
            rtol=1e-4, atol=1e-4,
        )


def test_batch_compute_metrics_with_eval_device():
    """Test batch_compute_metrics with explicit eval_device."""
    model = MLP(hidden_sizes=[3, 3], input_size=2, output_size=1, device="cpu")
    circuits = enumerate_all_valid_circuit(model, use_tqdm=False)
    x = torch.randn(10, 2)
    y_target = torch.randint(0, 2, (10, 1)).float()

    with torch.no_grad():
        y_pred = model(x)

    # Test with CPU
    accs_cpu, logits_cpu, bits_cpu, bests_cpu = batch_compute_metrics(
        model, circuits, x, y_target, y_pred, eval_device="cpu"
    )

    # If MPS available, test with MPS
    if torch.backends.mps.is_available():
        accs_mps, logits_mps, bits_mps, bests_mps = batch_compute_metrics(
            model, circuits, x, y_target, y_pred, eval_device="mps"
        )
        # Results should match
        np.testing.assert_allclose(accs_cpu, accs_mps, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(bits_cpu, bits_mps, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(bests_cpu, bests_mps, rtol=1e-4, atol=1e-4)


def test_base_masks_with_adapt_matches_direct():
    """Verify base masks + adapt gives same results as direct precompute."""
    model = MLP(hidden_sizes=[3, 3], input_size=2, output_size=2, device="cpu")
    circuits = enumerate_all_valid_circuit(model, use_tqdm=False)
    x = torch.randn(10, 2)

    # Direct computation for each gate
    direct_g0 = precompute_circuit_masks(circuits, len(model.layers), gate_idx=0, device="cpu")
    direct_g1 = precompute_circuit_masks(circuits, len(model.layers), gate_idx=1, device="cpu")

    # Base + adapt approach
    base_masks = precompute_circuit_masks_base(circuits, len(model.layers), device="cpu")
    adapted_g0 = adapt_masks_for_gate(base_masks, gate_idx=0, output_size=2)
    adapted_g1 = adapt_masks_for_gate(base_masks, gate_idx=1, output_size=2)

    # Compare masks
    for layer_idx in range(len(model.layers)):
        np.testing.assert_allclose(
            direct_g0[layer_idx].numpy(),
            adapted_g0[layer_idx].numpy(),
            rtol=1e-5, atol=1e-5,
            err_msg=f"Gate 0, layer {layer_idx} masks differ"
        )
        np.testing.assert_allclose(
            direct_g1[layer_idx].numpy(),
            adapted_g1[layer_idx].numpy(),
            rtol=1e-5, atol=1e-5,
            err_msg=f"Gate 1, layer {layer_idx} masks differ"
        )


def test_base_masks_evaluate_matches():
    """Verify evaluation results match between base+adapt and direct precompute."""
    model = MLP(hidden_sizes=[3, 3], input_size=2, output_size=2, device="cpu")
    circuits = enumerate_all_valid_circuit(model, use_tqdm=False)
    x = torch.randn(15, 2)

    # Direct precompute approach
    direct_g0 = precompute_circuit_masks(circuits, len(model.layers), gate_idx=0, device="cpu")
    result_direct_g0 = batch_evaluate_subcircuits(model, circuits, x, gate_idx=0, precomputed_masks=direct_g0)

    # Base + adapt approach
    base_masks = precompute_circuit_masks_base(circuits, len(model.layers), device="cpu")
    adapted_g0 = adapt_masks_for_gate(base_masks, gate_idx=0, output_size=2)
    result_adapted_g0 = batch_evaluate_subcircuits(model, circuits, x, gate_idx=0, precomputed_masks=adapted_g0)

    np.testing.assert_allclose(
        result_direct_g0.numpy(),
        result_adapted_g0.numpy(),
        rtol=1e-5, atol=1e-5,
        err_msg="Evaluation results differ between direct and adapt approaches"
    )


def test_single_output_adapt_returns_base():
    """For single-output models, adapt should return unchanged masks."""
    model = MLP(hidden_sizes=[3, 3], input_size=2, output_size=1, device="cpu")
    circuits = enumerate_all_valid_circuit(model, use_tqdm=False)

    base_masks = precompute_circuit_masks_base(circuits, len(model.layers), device="cpu")
    adapted = adapt_masks_for_gate(base_masks, gate_idx=0, output_size=1)

    # Should return same masks (no slicing needed)
    assert len(adapted) == len(base_masks)
    for layer_idx in range(len(model.layers)):
        assert torch.equal(adapted[layer_idx], base_masks[layer_idx])


def test_batch_evaluate_edge_variants_returns_results():
    """Verify edge variant evaluation returns correct structure."""
    model = MLP(hidden_sizes=[3, 3], input_size=2, output_size=1, device="cpu")
    circuits = enumerate_all_valid_circuit(model, use_tqdm=False)
    x = torch.randn(10, 2)
    y_target = torch.randint(0, 2, (10, 1)).float()

    with torch.no_grad():
        y_pred = model(x)

    # Select a few circuits to test edge variants
    test_circuits = circuits[:3]

    results = batch_evaluate_edge_variants(
        model, test_circuits, x, y_target, y_pred, gate_idx=0
    )

    assert len(results) == len(test_circuits)

    for orig_idx, opt_circuit, acc, logit_sim, bit_sim in results:
        assert 0 <= orig_idx < len(test_circuits)
        assert opt_circuit is not None
        assert 0 <= acc <= 1
        assert 0 <= bit_sim <= 1
        # logit_sim can be negative if MSE > 1


def test_batch_evaluate_edge_variants_preserves_node_pattern():
    """Edge optimization should preserve the original node pattern."""
    model = MLP(hidden_sizes=[3, 3], input_size=2, output_size=1, device="cpu")
    circuits = enumerate_all_valid_circuit(model, use_tqdm=False)
    x = torch.randn(10, 2)
    y_target = torch.randint(0, 2, (10, 1)).float()

    with torch.no_grad():
        y_pred = model(x)

    test_circuits = circuits[:5]

    results = batch_evaluate_edge_variants(
        model, test_circuits, x, y_target, y_pred, gate_idx=0
    )

    for orig_idx, opt_circuit, _, _, _ in results:
        original = test_circuits[orig_idx]
        # Node masks should match exactly
        assert len(opt_circuit.node_masks) == len(original.node_masks)
        for opt_nm, orig_nm in zip(opt_circuit.node_masks, original.node_masks):
            np.testing.assert_array_equal(opt_nm, orig_nm)


def test_batch_evaluate_edge_variants_multi_gate():
    """Test edge variant evaluation with multi-gate model."""
    model = MLP(hidden_sizes=[3, 3], input_size=2, output_size=2, device="cpu")
    circuits = enumerate_all_valid_circuit(model, use_tqdm=False)
    x = torch.randn(10, 2)
    y_target = torch.randint(0, 2, (10, 1)).float()

    with torch.no_grad():
        y_pred = model(x)[:, :1]  # Use first gate output

    test_circuits = circuits[:3]

    # Test gate 0
    results_g0 = batch_evaluate_edge_variants(
        model, test_circuits, x, y_target, y_pred, gate_idx=0
    )

    # Test gate 1
    results_g1 = batch_evaluate_edge_variants(
        model, test_circuits, x, y_target, y_pred, gate_idx=1
    )

    # Both should return same number of results
    assert len(results_g0) == len(results_g1) == len(test_circuits)
