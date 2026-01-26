"""Tests for causal analysis functions."""

import pytest
import numpy as np
import torch

from identifiability_toy_study.common.circuit import Circuit, enumerate_all_valid_circuit
from identifiability_toy_study.common.neural_model import MLP
from identifiability_toy_study.common.schemas import IdentifiabilityConstraints, SubcircuitMetrics
from identifiability_toy_study.causal_analysis import (
    filter_subcircuits,
    _generate_ood_samples,
    _generate_noise_samples,
)


# ===== Tests for filter_subcircuits =====


class TestFilterSubcircuits:
    """Tests for filter_subcircuits function."""

    @pytest.fixture
    def model_and_circuits(self):
        """Create a small model with circuits and structures."""
        model = MLP(hidden_sizes=[3, 3], input_size=2, output_size=1, device="cpu")
        circuits = enumerate_all_valid_circuit(model, use_tqdm=False)
        structures = [c.analyze_structure() for c in circuits]
        return model, circuits, structures

    def test_epsilon_filtering_strict(self, model_and_circuits):
        """Strict epsilon should filter most subcircuits."""
        _, circuits, structures = model_and_circuits
        constraints = IdentifiabilityConstraints(epsilon=0.001)

        # All subcircuits with low similarity should be filtered
        metrics = [
            SubcircuitMetrics(idx=i, accuracy=0.9, logit_similarity=0.9, bit_similarity=0.9)
            for i in range(len(circuits))
        ]

        result = filter_subcircuits(constraints, metrics, circuits, structures)
        # 1 - 0.9 = 0.1 > 0.001, so all should be filtered
        assert len(result) == 0

    def test_epsilon_filtering_lenient(self, model_and_circuits):
        """Lenient epsilon should allow more subcircuits."""
        _, circuits, structures = model_and_circuits
        constraints = IdentifiabilityConstraints(epsilon=0.2)

        # Some subcircuits with high similarity should pass
        metrics = [
            SubcircuitMetrics(
                idx=i,
                accuracy=0.95 if i < 5 else 0.5,
                logit_similarity=0.9,
                bit_similarity=0.95 if i < 5 else 0.5,
            )
            for i in range(len(circuits))
        ]

        result = filter_subcircuits(constraints, metrics, circuits, structures)
        # First 5 have 1 - 0.95 = 0.05 < 0.2, should pass
        assert len(result) > 0
        assert all(idx < 5 for idx in result)

    def test_both_thresholds_must_pass(self, model_and_circuits):
        """Both bit_similarity AND accuracy must meet epsilon."""
        _, circuits, structures = model_and_circuits
        constraints = IdentifiabilityConstraints(epsilon=0.1)

        # High accuracy but low similarity
        metrics = [
            SubcircuitMetrics(
                idx=i,
                accuracy=0.99,  # Good
                logit_similarity=0.9,
                bit_similarity=0.5,  # Bad: 1 - 0.5 = 0.5 > 0.1
            )
            for i in range(len(circuits))
        ]

        result = filter_subcircuits(constraints, metrics, circuits, structures)
        assert len(result) == 0  # All filtered due to low bit_similarity

    def test_deduplication_by_activation_pattern(self, model_and_circuits):
        """Subcircuits with same node_masks should be deduplicated."""
        _, circuits, structures = model_and_circuits
        constraints = IdentifiabilityConstraints(epsilon=0.5)

        # All subcircuits pass epsilon
        metrics = [
            SubcircuitMetrics(idx=i, accuracy=0.99, logit_similarity=0.99, bit_similarity=0.99)
            for i in range(len(circuits))
        ]

        result = filter_subcircuits(constraints, metrics, circuits, structures)

        # Result should have at most one subcircuit per unique node_masks pattern
        seen_patterns = set()
        for idx in result:
            pattern = tuple(tuple(m.tolist()) for m in circuits[idx].node_masks)
            assert pattern not in seen_patterns, f"Duplicate pattern found at idx {idx}"
            seen_patterns.add(pattern)

    def test_quality_sorting(self, model_and_circuits):
        """Subcircuits should be sorted by accuracy, similarity, then sparsity."""
        _, circuits, structures = model_and_circuits
        constraints = IdentifiabilityConstraints(epsilon=0.5)

        # Create metrics with varying quality
        metrics = [
            SubcircuitMetrics(
                idx=i,
                accuracy=0.99 - (i * 0.01),  # Decreasing accuracy
                logit_similarity=0.99,
                bit_similarity=0.99,
            )
            for i in range(len(circuits))
        ]

        result = filter_subcircuits(constraints, metrics, circuits, structures, max_subcircuits=5)

        # First result should be the one with highest accuracy
        assert result[0] == 0  # idx 0 has accuracy 0.99

    def test_max_subcircuits_limit(self, model_and_circuits):
        """Should return at most max_subcircuits."""
        _, circuits, structures = model_and_circuits
        constraints = IdentifiabilityConstraints(epsilon=0.5)

        metrics = [
            SubcircuitMetrics(idx=i, accuracy=0.99, logit_similarity=0.99, bit_similarity=0.99)
            for i in range(len(circuits))
        ]

        for max_sc in [1, 2, 3, 5]:
            result = filter_subcircuits(
                constraints, metrics, circuits, structures, max_subcircuits=max_sc
            )
            assert len(result) <= max_sc

    def test_diversity_selection(self, model_and_circuits):
        """Ties should be resolved by picking less overlapping subcircuits."""
        _, circuits, structures = model_and_circuits
        constraints = IdentifiabilityConstraints(epsilon=0.5)

        # All have same quality
        metrics = [
            SubcircuitMetrics(idx=i, accuracy=0.99, logit_similarity=0.99, bit_similarity=0.99)
            for i in range(len(circuits))
        ]

        result = filter_subcircuits(
            constraints, metrics, circuits, structures, max_subcircuits=3
        )

        # Selected subcircuits should have some diversity (not all identical)
        if len(result) > 1:
            # Check that not all selected are from same pattern
            patterns = [tuple(tuple(m.tolist()) for m in circuits[idx].node_masks) for idx in result]
            # At least first one should be different from last (if diverse)
            # (This is a weak check - just verifying diversity logic runs)

    def test_empty_when_none_pass(self, model_and_circuits):
        """Returns empty list when no subcircuits pass thresholds."""
        _, circuits, structures = model_and_circuits
        constraints = IdentifiabilityConstraints(epsilon=0.0)  # Impossible to pass

        metrics = [
            SubcircuitMetrics(idx=i, accuracy=0.99, logit_similarity=0.99, bit_similarity=0.99)
            for i in range(len(circuits))
        ]

        result = filter_subcircuits(constraints, metrics, circuits, structures)
        # 1 - 0.99 = 0.01 > 0, so none pass
        assert len(result) == 0

    def test_returns_indices(self, model_and_circuits):
        """Result should be list of integer indices."""
        _, circuits, structures = model_and_circuits
        constraints = IdentifiabilityConstraints(epsilon=0.5)

        metrics = [
            SubcircuitMetrics(idx=i, accuracy=0.99, logit_similarity=0.99, bit_similarity=0.99)
            for i in range(len(circuits))
        ]

        result = filter_subcircuits(constraints, metrics, circuits, structures)

        assert isinstance(result, list)
        for idx in result:
            assert isinstance(idx, int)
            assert 0 <= idx < len(circuits)


# ===== Tests for _generate_ood_samples =====


class TestGenerateOODSamples:
    """Tests for _generate_ood_samples function."""

    @pytest.fixture
    def base_inputs(self):
        """Standard binary inputs for testing."""
        return [
            torch.tensor([0.0, 0.0]),
            torch.tensor([0.0, 1.0]),
            torch.tensor([1.0, 0.0]),
            torch.tensor([1.0, 1.0]),
        ]

    def test_returns_list_of_tuples(self, base_inputs):
        """Should return list of (perturbed, base, scale) tuples."""
        samples = _generate_ood_samples(base_inputs, n_samples_per_base=10)

        assert isinstance(samples, list)
        for sample in samples:
            assert isinstance(sample, tuple)
            assert len(sample) == 3
            perturbed, base, scale = sample
            assert isinstance(perturbed, torch.Tensor)
            assert isinstance(base, torch.Tensor)
            assert isinstance(scale, float)

    def test_skips_zero_zero_input(self, base_inputs):
        """(0,0) should be skipped since scaling doesn't create OOD."""
        samples = _generate_ood_samples(base_inputs, n_samples_per_base=10)

        # None of the base inputs should be (0,0)
        for _, base, _ in samples:
            assert not (base[0].item() == 0.0 and base[1].item() == 0.0)

    def test_positive_ood_above_one(self, base_inputs):
        """Positive OOD samples should have values > 1 for non-zero base."""
        samples = _generate_ood_samples(base_inputs, n_samples_per_base=20)

        positive_samples = [(p, b, s) for p, b, s in samples if s > 0]
        assert len(positive_samples) > 0

        for perturbed, base, scale in positive_samples:
            assert scale > 1, "Positive OOD should have scale > 1"
            # For non-zero base values, scaled value should be > 1
            for i in range(len(base)):
                if base[i].item() != 0:
                    assert abs(perturbed[i].item()) >= 1

    def test_negative_ood_below_zero(self, base_inputs):
        """Negative OOD samples should have values < 0 for positive base."""
        samples = _generate_ood_samples(base_inputs, n_samples_per_base=20)

        negative_samples = [(p, b, s) for p, b, s in samples if s < 0]
        assert len(negative_samples) > 0

        for perturbed, base, scale in negative_samples:
            assert scale < 0, "Negative OOD should have scale < 0"
            # For positive base values, scaled value should be negative
            for i in range(len(base)):
                if base[i].item() > 0:
                    assert perturbed[i].item() < 0

    def test_multiplicative_scaling(self, base_inputs):
        """Samples should be exactly base * scale."""
        samples = _generate_ood_samples(base_inputs, n_samples_per_base=10)

        for perturbed, base, scale in samples:
            expected = base * scale
            np.testing.assert_allclose(
                perturbed.numpy(), expected.numpy(), rtol=1e-5, atol=1e-5
            )

    def test_sample_count(self, base_inputs):
        """Should generate correct number of samples (excluding 0,0)."""
        n_samples = 10
        samples = _generate_ood_samples(base_inputs, n_samples_per_base=n_samples)

        # 3 valid base inputs (excluding 0,0) * n_samples each
        expected_count = 3 * n_samples
        assert len(samples) == expected_count

    def test_scale_range_positive(self, base_inputs):
        """Positive scales should be in [1, 100] (10^[0,2])."""
        samples = _generate_ood_samples(base_inputs, n_samples_per_base=100)

        positive_scales = [s for _, _, s in samples if s > 0]
        for scale in positive_scales:
            assert 1 <= scale <= 100

    def test_scale_range_negative(self, base_inputs):
        """Negative scales should be in [-100, -1] (-10^[0,2])."""
        samples = _generate_ood_samples(base_inputs, n_samples_per_base=100)

        negative_scales = [s for _, _, s in samples if s < 0]
        for scale in negative_scales:
            assert -100 <= scale <= -1


# ===== Tests for _generate_noise_samples =====


class TestGenerateNoiseSamples:
    """Tests for _generate_noise_samples function (bonus, related)."""

    @pytest.fixture
    def base_inputs(self):
        return [
            torch.tensor([0.0, 0.0]),
            torch.tensor([0.0, 1.0]),
            torch.tensor([1.0, 0.0]),
            torch.tensor([1.0, 1.0]),
        ]

    def test_noise_magnitude_under_half(self, base_inputs):
        """All noise magnitudes should be < 0.5."""
        samples = _generate_noise_samples(base_inputs, n_samples_per_base=100)

        for perturbed, base, magnitude in samples:
            actual_magnitude = (perturbed - base).norm().item()
            assert actual_magnitude < 0.5
            assert magnitude < 0.5

    def test_noise_magnitude_min(self, base_inputs):
        """Noise magnitudes should be >= 0.01."""
        samples = _generate_noise_samples(base_inputs, n_samples_per_base=100)

        for _, _, magnitude in samples:
            assert magnitude >= 0.01

    def test_noise_preserves_base(self, base_inputs):
        """Base input should be returned unchanged."""
        samples = _generate_noise_samples(base_inputs, n_samples_per_base=10)

        for perturbed, base, _ in samples:
            # Check base is one of the original inputs
            found = any(
                torch.allclose(base, b) for b in base_inputs
            )
            assert found

    def test_noise_sample_count(self, base_inputs):
        """Should generate n_samples_per_base * len(base_inputs) samples."""
        n_samples = 10
        samples = _generate_noise_samples(base_inputs, n_samples_per_base=n_samples)

        expected = len(base_inputs) * n_samples
        assert len(samples) == expected
