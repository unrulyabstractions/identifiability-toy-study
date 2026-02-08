"""
Test that batched operations produce identical results to sequential operations.

These tests verify correctness by comparing:
1. Batched circuit evaluation vs sequential model forward passes
2. Batched metrics vs sequential metric computation
3. Parallel structure analysis vs sequential
4. Different devices (CPU vs MPS) produce same results
"""

import numpy as np
import pytest
import torch
from src.math import calculate_best_match_rate, logits_to_binary

from src.circuit import (
    batch_compute_metrics,
    batch_evaluate_subcircuits,
    precompute_circuit_masks,
    enumerate_all_valid_circuit,
)
from src.model import MLP
from src.infra import ParallelTasks

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def small_model():
    """Small model for fast tests."""
    return MLP(hidden_sizes=[3, 3], input_size=2, output_size=1, device="cpu")


@pytest.fixture
def medium_model():
    """Medium model (w=4, d=2) for more thorough tests."""
    return MLP(hidden_sizes=[4, 4], input_size=2, output_size=1, device="cpu")


@pytest.fixture
def multi_output_model():
    """Multi-output model for testing gate_idx handling."""
    return MLP(hidden_sizes=[3, 3], input_size=2, output_size=2, device="cpu")


# =============================================================================
# Sequential reference implementations
# =============================================================================


def sequential_evaluate_circuits(model, circuits, x, gate_idx=0):
    """Reference: evaluate circuits one at a time using model forward pass."""
    results = []
    with torch.inference_mode():
        for circuit in circuits:
            intervention = circuit.to_intervention(model.device)
            y = model(x, intervention=intervention)
            if y.shape[-1] > 1:
                y = y[:, gate_idx : gate_idx + 1]
            results.append(y)
    return torch.stack(results)  # [n_circuits, batch, 1]


def sequential_compute_metrics(model, circuits, x, y_target, y_pred, gate_idx=0):
    """Reference: compute metrics for each circuit sequentially."""
    accuracies = []
    logit_similarities = []
    bit_similarities = []
    best_similarities = []

    # y_target is already binary 0/1 (ground truth), y_pred is logits
    bit_target = y_target.float()  # [n_samples, 1] - already 0/1
    bit_pred = logits_to_binary(y_pred)  # [n_samples, 1] - threshold logits at 0

    with torch.inference_mode():
        for circuit in circuits:
            intervention = circuit.to_intervention(model.device)
            y_circuit = model(x, intervention=intervention)
            if y_circuit.shape[-1] > 1:
                y_circuit = y_circuit[:, gate_idx : gate_idx + 1]

            bit_circuit = logits_to_binary(y_circuit)

            # Accuracy
            correct = bit_circuit.eq(bit_target)
            acc = correct.float().mean().item()
            accuracies.append(acc)

            # Bit similarity
            same = bit_circuit.eq(bit_pred)
            bit_sim = same.float().mean().item()
            bit_similarities.append(bit_sim)

            # Best similarity - uses calculate_best_match_rate for modularity
            best_sim = calculate_best_match_rate(y_pred, y_circuit).item()
            best_similarities.append(best_sim)

            # Logit similarity (R²-like: 1 - mse/var)
            mse = ((y_circuit - y_pred) ** 2).mean()
            var = y_pred.var().clamp(min=1e-8)
            logit_sim = (1 - mse / var).item()
            logit_similarities.append(logit_sim)

    return (
        np.array(accuracies),
        np.array(logit_similarities),
        np.array(bit_similarities),
        np.array(best_similarities),
    )


def sequential_structure_analysis(circuits):
    """Reference: analyze structures sequentially."""
    return [c.analyze_structure() for c in circuits]


# =============================================================================
# Test: Batched evaluation equals sequential
# =============================================================================


class TestBatchedEvalEqualsSequential:
    """Verify batched evaluation produces same results as sequential."""

    def test_small_model_basic(self, small_model):
        """Basic test with small model."""
        circuits = enumerate_all_valid_circuit(small_model, use_tqdm=False)
        x = torch.randn(10, 2)

        seq_result = sequential_evaluate_circuits(small_model, circuits, x)
        batch_result = batch_evaluate_subcircuits(small_model, circuits, x)

        np.testing.assert_allclose(
            batch_result.numpy(),
            seq_result.numpy(),
            rtol=1e-5,
            atol=1e-5,
            err_msg="Batched eval differs from sequential",
        )

    def test_medium_model(self, medium_model):
        """Test with more circuits (225 for w=4, d=2)."""
        circuits = enumerate_all_valid_circuit(medium_model, use_tqdm=False)
        x = torch.randn(20, 2)

        seq_result = sequential_evaluate_circuits(medium_model, circuits, x)
        batch_result = batch_evaluate_subcircuits(medium_model, circuits, x)

        np.testing.assert_allclose(
            batch_result.numpy(),
            seq_result.numpy(),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_multi_output_gate0(self, multi_output_model):
        """Test multi-output model, gate 0."""
        circuits = enumerate_all_valid_circuit(multi_output_model, use_tqdm=False)
        x = torch.randn(10, 2)

        seq_result = sequential_evaluate_circuits(
            multi_output_model, circuits, x, gate_idx=0
        )
        batch_result = batch_evaluate_subcircuits(
            multi_output_model, circuits, x, gate_idx=0
        )

        np.testing.assert_allclose(
            batch_result.numpy(),
            seq_result.numpy(),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_multi_output_gate1(self, multi_output_model):
        """Test multi-output model, gate 1."""
        circuits = enumerate_all_valid_circuit(multi_output_model, use_tqdm=False)
        x = torch.randn(10, 2)

        seq_result = sequential_evaluate_circuits(
            multi_output_model, circuits, x, gate_idx=1
        )
        batch_result = batch_evaluate_subcircuits(
            multi_output_model, circuits, x, gate_idx=1
        )

        np.testing.assert_allclose(
            batch_result.numpy(),
            seq_result.numpy(),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_with_precomputed_masks(self, medium_model):
        """Test that precomputed masks give same result."""
        circuits = enumerate_all_valid_circuit(medium_model, use_tqdm=False)
        x = torch.randn(15, 2)

        # Without precompute
        result_no_precompute = batch_evaluate_subcircuits(medium_model, circuits, x)

        # With precompute
        precomputed = precompute_circuit_masks(
            circuits, len(medium_model.layers), gate_idx=0, device="cpu"
        )
        result_precompute = batch_evaluate_subcircuits(
            medium_model, circuits, x, precomputed_masks=precomputed
        )

        np.testing.assert_allclose(
            result_precompute.numpy(),
            result_no_precompute.numpy(),
            rtol=1e-5,
            atol=1e-5,
        )

    @pytest.mark.parametrize("batch_size", [1, 5, 10, 32, 64])
    def test_various_batch_sizes(self, small_model, batch_size):
        """Test with various batch sizes."""
        circuits = enumerate_all_valid_circuit(small_model, use_tqdm=False)
        x = torch.randn(batch_size, 2)

        seq_result = sequential_evaluate_circuits(small_model, circuits, x)
        batch_result = batch_evaluate_subcircuits(small_model, circuits, x)

        np.testing.assert_allclose(
            batch_result.numpy(),
            seq_result.numpy(),
            rtol=1e-5,
            atol=1e-5,
        )


# =============================================================================
# Test: Batched metrics equals sequential
# =============================================================================


class TestBatchedMetricsEqualsSequential:
    """Verify batched metrics computation matches sequential."""

    def test_accuracy_matches(self, small_model):
        """Test accuracy computation matches."""
        circuits = enumerate_all_valid_circuit(small_model, use_tqdm=False)
        x = torch.randn(20, 2)
        y_target = torch.randint(0, 2, (20, 1)).float()

        with torch.no_grad():
            y_pred = small_model(x)

        seq_accs, _, _, _ = sequential_compute_metrics(
            small_model, circuits, x, y_target, y_pred
        )
        batch_accs, _, _, _ = batch_compute_metrics(
            small_model, circuits, x, y_target, y_pred
        )

        np.testing.assert_allclose(batch_accs, seq_accs, rtol=1e-5, atol=1e-5)

    def test_bit_similarity_matches(self, small_model):
        """Test bit similarity computation matches."""
        circuits = enumerate_all_valid_circuit(small_model, use_tqdm=False)
        x = torch.randn(20, 2)
        y_target = torch.randint(0, 2, (20, 1)).float()

        with torch.no_grad():
            y_pred = small_model(x)

        _, _, seq_bits, _ = sequential_compute_metrics(
            small_model, circuits, x, y_target, y_pred
        )
        _, _, batch_bits, _ = batch_compute_metrics(
            small_model, circuits, x, y_target, y_pred
        )

        np.testing.assert_allclose(batch_bits, seq_bits, rtol=1e-5, atol=1e-5)

    def test_logit_similarity_matches(self, small_model):
        """Test logit similarity computation matches."""
        circuits = enumerate_all_valid_circuit(small_model, use_tqdm=False)
        x = torch.randn(20, 2)
        y_target = torch.randint(0, 2, (20, 1)).float()

        with torch.no_grad():
            y_pred = small_model(x)

        _, seq_logits, _, _ = sequential_compute_metrics(
            small_model, circuits, x, y_target, y_pred
        )
        _, batch_logits, _, _ = batch_compute_metrics(
            small_model, circuits, x, y_target, y_pred
        )

        np.testing.assert_allclose(batch_logits, seq_logits, rtol=1e-4, atol=1e-4)

    def test_all_metrics_medium_model(self, medium_model):
        """Test all metrics with medium model."""
        circuits = enumerate_all_valid_circuit(medium_model, use_tqdm=False)
        x = torch.randn(30, 2)
        y_target = torch.randint(0, 2, (30, 1)).float()

        with torch.no_grad():
            y_pred = medium_model(x)

        seq_accs, seq_logits, seq_bits, _ = sequential_compute_metrics(
            medium_model, circuits, x, y_target, y_pred
        )
        batch_accs, batch_logits, batch_bits, _ = batch_compute_metrics(
            medium_model, circuits, x, y_target, y_pred
        )

        np.testing.assert_allclose(batch_accs, seq_accs, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(batch_bits, seq_bits, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(batch_logits, seq_logits, rtol=1e-4, atol=1e-4)


# =============================================================================
# Test: Parallel structure analysis equals sequential
# =============================================================================


class TestParallelStructureEqualsSequential:
    """Verify parallel structure analysis matches sequential."""

    def test_structure_analysis_matches(self, medium_model):
        """Test parallel structure analysis gives same results."""
        circuits = enumerate_all_valid_circuit(medium_model, use_tqdm=False)

        # Sequential
        seq_structures = sequential_structure_analysis(circuits)

        # Parallel
        with ParallelTasks(max_workers=4) as tasks:
            futures = [tasks.submit(c.analyze_structure) for c in circuits]
        parallel_structures = [f.result() for f in futures]

        # Compare
        assert len(seq_structures) == len(parallel_structures)
        for i, (seq, par) in enumerate(zip(seq_structures, parallel_structures)):
            assert seq.node_sparsity == par.node_sparsity, (
                f"Circuit {i}: node_sparsity mismatch"
            )
            assert seq.edge_sparsity == par.edge_sparsity, (
                f"Circuit {i}: edge_sparsity mismatch"
            )
            assert seq.width == par.width, f"Circuit {i}: width mismatch"
            assert seq.depth == par.depth, f"Circuit {i}: depth mismatch"


# =============================================================================
# Test: CPU vs MPS equivalence (if available)
# =============================================================================


class TestCpuMpsEquivalence:
    """Verify CPU and MPS produce same results."""

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(), reason="MPS not available"
    )
    def test_eval_cpu_vs_mps(self, small_model):
        """Test batched eval gives same results on CPU vs MPS."""
        circuits = enumerate_all_valid_circuit(small_model, use_tqdm=False)
        x = torch.randn(10, 2)

        # CPU
        result_cpu = batch_evaluate_subcircuits(
            small_model, circuits, x, eval_device="cpu"
        )

        # MPS
        result_mps = batch_evaluate_subcircuits(
            small_model, circuits, x, eval_device="mps"
        )

        np.testing.assert_allclose(
            result_mps.cpu().numpy(),
            result_cpu.numpy(),
            rtol=1e-4,
            atol=1e-4,
            err_msg="MPS and CPU results differ",
        )

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(), reason="MPS not available"
    )
    def test_metrics_cpu_vs_mps(self, small_model):
        """Test metrics computation gives same results on CPU vs MPS."""
        circuits = enumerate_all_valid_circuit(small_model, use_tqdm=False)
        x = torch.randn(15, 2)
        y_target = torch.randint(0, 2, (15, 1)).float()

        with torch.no_grad():
            y_pred = small_model(x)

        # CPU
        accs_cpu, logits_cpu, bits_cpu, _ = batch_compute_metrics(
            small_model, circuits, x, y_target, y_pred, eval_device="cpu"
        )

        # MPS
        accs_mps, logits_mps, bits_mps, _ = batch_compute_metrics(
            small_model, circuits, x, y_target, y_pred, eval_device="mps"
        )

        np.testing.assert_allclose(accs_mps, accs_cpu, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(bits_mps, bits_cpu, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(logits_mps, logits_cpu, rtol=1e-3, atol=1e-3)

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(), reason="MPS not available"
    )
    def test_precomputed_masks_cpu_vs_mps(self, medium_model):
        """Test precomputed masks work correctly on both devices."""
        circuits = enumerate_all_valid_circuit(medium_model, use_tqdm=False)
        x = torch.randn(10, 2)

        # Precompute on each device
        masks_cpu = precompute_circuit_masks(
            circuits, len(medium_model.layers), device="cpu"
        )
        masks_mps = precompute_circuit_masks(
            circuits, len(medium_model.layers), device="mps"
        )

        # Verify masks are equivalent
        for i, (m_cpu, m_mps) in enumerate(zip(masks_cpu, masks_mps)):
            np.testing.assert_allclose(
                m_mps.cpu().numpy(),
                m_cpu.numpy(),
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"Layer {i} masks differ",
            )

        # Verify evaluation with precomputed masks
        result_cpu = batch_evaluate_subcircuits(
            medium_model, circuits, x, precomputed_masks=masks_cpu, eval_device="cpu"
        )
        result_mps = batch_evaluate_subcircuits(
            medium_model, circuits, x, precomputed_masks=masks_mps, eval_device="mps"
        )

        np.testing.assert_allclose(
            result_mps.cpu().numpy(),
            result_cpu.numpy(),
            rtol=1e-4,
            atol=1e-4,
        )


# =============================================================================
# Test: Edge cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_circuit(self, small_model):
        """Test with just one circuit."""
        circuits = enumerate_all_valid_circuit(small_model, use_tqdm=False)[:1]
        x = torch.randn(5, 2)

        seq_result = sequential_evaluate_circuits(small_model, circuits, x)
        batch_result = batch_evaluate_subcircuits(small_model, circuits, x)

        np.testing.assert_allclose(
            batch_result.numpy(),
            seq_result.numpy(),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_single_sample(self, small_model):
        """Test with batch size of 1."""
        circuits = enumerate_all_valid_circuit(small_model, use_tqdm=False)
        x = torch.randn(1, 2)

        seq_result = sequential_evaluate_circuits(small_model, circuits, x)
        batch_result = batch_evaluate_subcircuits(small_model, circuits, x)

        np.testing.assert_allclose(
            batch_result.numpy(),
            seq_result.numpy(),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_deterministic_results(self, small_model):
        """Test that results are deterministic across runs."""
        circuits = enumerate_all_valid_circuit(small_model, use_tqdm=False)
        x = torch.randn(10, 2)

        result1 = batch_evaluate_subcircuits(small_model, circuits, x)
        result2 = batch_evaluate_subcircuits(small_model, circuits, x)

        np.testing.assert_array_equal(
            result1.numpy(), result2.numpy(), err_msg="Results not deterministic"
        )

    def test_different_inputs_different_outputs(self, small_model):
        """Test that different inputs produce different outputs."""
        circuits = enumerate_all_valid_circuit(small_model, use_tqdm=False)
        x1 = torch.zeros(5, 2)
        x2 = torch.ones(5, 2)

        result1 = batch_evaluate_subcircuits(small_model, circuits, x1)
        result2 = batch_evaluate_subcircuits(small_model, circuits, x2)

        # Should not be equal
        assert not np.allclose(result1.numpy(), result2.numpy())
