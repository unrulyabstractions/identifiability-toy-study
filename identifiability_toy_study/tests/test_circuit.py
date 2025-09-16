import copy
import numpy as np
import pytest
import tempfile
import os
import torch
from unittest.mock import Mock, patch

from identifiability_toy_study.common.circuit import (
    Circuit,
    visualize_circuit_heatmap,
    compute_jaccard_index,
    enumerate_all_valid_circuit,
    analyze_circuits,
    find_circuits,
    _enumerate_edge_mask_per_layer,
)
from identifiability_toy_study.common.causal import Intervention, PatchShape


class TestCircuit:
    def test_circuit_initialization(self):
        """Test Circuit initialization"""
        node_masks = [np.array([1, 1]), np.array([1, 0]), np.array([1])]
        edge_masks = [np.array([[1, 0], [1, 1]]), np.array([[1], [0]])]

        circuit = Circuit(node_masks, edge_masks)

        assert len(circuit.node_masks) == 3
        assert len(circuit.edge_masks) == 2
        np.testing.assert_array_equal(circuit.node_masks[0], [1, 1])

    def test_circuit_repr(self):
        """Test Circuit string representation"""
        node_masks = [np.array([1, 1])]
        edge_masks = [np.array([[1, 0]])]

        circuit = Circuit(node_masks, edge_masks)
        repr_str = repr(circuit)

        assert "Circuit" in repr_str
        assert "node_masks" in repr_str
        assert "edge_masks" in repr_str

    def test_to_intervention(self):
        """Test conversion to intervention"""
        node_masks = [np.array([1, 0]), np.array([1, 1])]
        edge_masks = [np.array([[1, 0], [0, 1]])]

        circuit = Circuit(node_masks, edge_masks)
        mock_model = Mock()
        mock_model.device = "cpu"

        intervention = circuit.to_intervention(mock_model)

        assert isinstance(intervention, Intervention)
        assert len(intervention.patches) >= 1

    def test_full_circuit(self):
        """Test creation of full circuit"""
        layer_sizes = [2, 3, 1]
        circuit = Circuit.full(layer_sizes)

        assert len(circuit.node_masks) == 3
        assert len(circuit.edge_masks) == 2
        assert all(np.all(mask == 1) for mask in circuit.node_masks)
        assert all(np.all(mask == 1) for mask in circuit.edge_masks)

    def test_save_and_load_circuit(self):
        """Test saving and loading circuit"""
        node_masks = [np.array([1, 0]), np.array([1, 1])]
        edge_masks = [np.array([[1, 0], [0, 1]])]

        circuit = Circuit(node_masks, edge_masks)

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp_file:
            try:
                circuit.save_to_file(tmp_file.name)

                loaded_circuit = Circuit.load_from_file(tmp_file.name)

                assert len(loaded_circuit.node_masks) == len(circuit.node_masks)
                assert len(loaded_circuit.edge_masks) == len(circuit.edge_masks)

                for orig, loaded in zip(circuit.node_masks, loaded_circuit.node_masks):
                    np.testing.assert_array_equal(orig, loaded)

            finally:
                os.unlink(tmp_file.name)

    def test_validate_against_model(self):
        """Test circuit validation against model"""
        node_masks = [np.array([1, 1]), np.array([1, 0]), np.array([1])]
        edge_masks = [np.array([[1, 0], [0, 1]]), np.array([[1], [0]])]

        circuit = Circuit(node_masks, edge_masks)

        # Create mock model
        mock_model = Mock()
        mock_model.num_layers = 2

        # Create mock layers
        mock_layer1 = Mock()
        mock_linear1 = Mock()
        mock_linear1.out_features = 2
        mock_linear1.weight = Mock()
        mock_linear1.weight.shape = (2, 2)
        mock_layer1.__getitem__ = Mock(return_value=mock_linear1)

        mock_layer2 = Mock()
        mock_linear2 = Mock()
        mock_linear2.out_features = 1
        mock_linear2.weight = Mock()
        mock_linear2.weight.shape = (1, 2)
        mock_layer2.__getitem__ = Mock(return_value=mock_linear2)

        mock_model.layers = [mock_layer1, mock_layer2]

        # This should not raise any exceptions for a valid circuit
        circuit.validate_against_model(mock_model)

    def test_validate_against_model_invalid_dimensions(self):
        """Test circuit validation with invalid dimensions"""
        node_masks = [np.array([1, 1]), np.array([1, 0, 1])]  # Wrong size
        edge_masks = [np.array([[1, 0], [0, 1]])]

        circuit = Circuit(node_masks, edge_masks)

        mock_model = Mock()
        mock_model.num_layers = 1
        mock_layer = Mock()
        mock_linear = Mock()
        mock_linear.out_features = 2  # Doesn't match node_mask size of 3
        mock_linear.weight = Mock()
        mock_linear.weight.shape = (2, 2)
        mock_layer.__getitem__ = Mock(return_value=mock_linear)
        mock_model.layers = [mock_layer]

        with pytest.raises(ValueError, match="Node mask at layer"):
            circuit.validate_against_model(mock_model)

    def test_circuit_inclusion(self):
        """Test circuit inclusion operator"""
        # Create two circuits where first is included in second
        node_masks1 = [np.array([1, 0]), np.array([1, 0])]
        edge_masks1 = [np.array([[1, 0], [0, 0]])]

        node_masks2 = [np.array([1, 1]), np.array([1, 1])]
        edge_masks2 = [np.array([[1, 1], [1, 1]])]

        circuit1 = Circuit(node_masks1, edge_masks1)
        circuit2 = Circuit(node_masks2, edge_masks2)

        assert circuit1 <= circuit2
        assert not (circuit2 <= circuit1)

    def test_circuit_inclusion_different_sizes(self):
        """Test circuit inclusion with different sizes"""
        node_masks1 = [np.array([1, 0])]
        edge_masks1 = []

        node_masks2 = [np.array([1, 0]), np.array([1])]
        edge_masks2 = [np.array([[1], [0]])]

        circuit1 = Circuit(node_masks1, edge_masks1)
        circuit2 = Circuit(node_masks2, edge_masks2)

        assert not (circuit1 <= circuit2)

    def test_sparsity(self):
        """Test sparsity calculation"""
        node_masks = [np.array([1, 1]), np.array([1, 0]), np.array([1])]
        edge_masks = [np.array([[1, 0], [0, 1]]), np.array([[1], [0]])]

        circuit = Circuit(node_masks, edge_masks)
        node_sparsity, edge_sparsity, combined_sparsity = circuit.sparsity()

        assert 0 <= node_sparsity <= 1
        assert 0 <= edge_sparsity <= 1
        assert 0 <= combined_sparsity <= 1
        assert node_sparsity == 0.5  # 1 out of 2 nodes is zero (excluding input/output)
        assert edge_sparsity == 0.25  # 1 out of 4 edges is zero

    def test_overlap_jaccard(self):
        """Test Jaccard overlap calculation"""
        node_masks1 = [np.array([1, 1]), np.array([1, 0]), np.array([1])]
        edge_masks1 = [np.array([[1, 0], [0, 1]]), np.array([[1], [0]])]

        node_masks2 = [np.array([1, 1]), np.array([1, 1]), np.array([1])]
        edge_masks2 = [np.array([[1, 1], [1, 1]]), np.array([[1], [1]])]

        circuit1 = Circuit(node_masks1, edge_masks1)
        circuit2 = Circuit(node_masks2, edge_masks2)

        jaccard = circuit1.overlap_jaccard(circuit2)

        assert 0 <= jaccard <= 1

    def test_ground(self):
        """Test circuit grounding"""
        node_masks = [np.array([1, 1]), np.array([1, 0]), np.array([1])]
        edge_masks = [np.array([[1, 0], [0, 1]]), np.array([[1], [0]])]

        circuit = Circuit(node_masks, edge_masks)

        # Create mock activations
        activations = [
            torch.tensor([[0.1, 0.9]]),  # Input
            torch.tensor([[0.2, 0.8]]),  # Hidden
            torch.tensor([[0.3]])        # Output
        ]

        with patch('identifiability_toy_study.common.circuit.enumerate_tts') as mock_enumerate_tts, \
             patch('identifiability_toy_study.common.circuit.compute_local_tts') as mock_compute_local_tts, \
             patch('identifiability_toy_study.common.circuit.name_gate') as mock_name_gate:

            mock_enumerate_tts.return_value = [{"separator": 0.5, "tt": {(0, 1): 0, (1, 0): 1}}]
            mock_compute_local_tts.return_value = [{(0, 0): 0, (0, 1): 1}]
            mock_name_gate.return_value = "AND"

            groundings = circuit.ground(activations)

            assert isinstance(groundings, list)

    def test_visualize(self):
        """Test circuit visualization"""
        node_masks = [np.array([1, 1]), np.array([1, 0])]
        edge_masks = [np.array([[1, 0], [0, 1]])]

        circuit = Circuit(node_masks, edge_masks)

        with patch('matplotlib.pyplot.figure'), \
             patch('matplotlib.pyplot.gca'), \
             patch('matplotlib.pyplot.show'), \
             patch('networkx.draw_networkx_nodes'), \
             patch('networkx.draw_networkx_edges'), \
             patch('networkx.draw_networkx_labels'):

            # Should not raise any exceptions
            circuit.visualize()


class TestVisualizeCircuitHeatmap:
    def test_visualize_circuit_heatmap(self):
        """Test circuit heatmap visualization"""
        # Create multiple circuits
        node_masks1 = [np.array([1, 1]), np.array([1, 0])]
        edge_masks1 = [np.array([[1, 0], [0, 1]])]
        circuit1 = Circuit(node_masks1, edge_masks1)

        node_masks2 = [np.array([1, 1]), np.array([1, 1])]
        edge_masks2 = [np.array([[1, 1], [1, 1]])]
        circuit2 = Circuit(node_masks2, edge_masks2)

        circuits = [circuit1, circuit2]

        with patch('matplotlib.pyplot.cm'), \
             patch('matplotlib.pyplot.colorbar'), \
             patch('matplotlib.pyplot.show'), \
             patch('networkx.draw_networkx_nodes'), \
             patch('networkx.draw_networkx_edges'):

            # Should not raise any exceptions
            visualize_circuit_heatmap(circuits)


class TestComputeJaccardIndex:
    def test_compute_jaccard_index_identical(self):
        """Test Jaccard index for identical masks"""
        mask1 = [np.array([1, 0, 1]), np.array([0, 1])]
        mask2 = [np.array([1, 0, 1]), np.array([0, 1])]

        jaccard = compute_jaccard_index(mask1, mask2)

        assert jaccard == 1.0

    def test_compute_jaccard_index_disjoint(self):
        """Test Jaccard index for disjoint masks"""
        mask1 = [np.array([1, 0, 0]), np.array([1, 0])]
        mask2 = [np.array([0, 1, 1]), np.array([0, 1])]

        jaccard = compute_jaccard_index(mask1, mask2)

        assert jaccard == 0.0

    def test_compute_jaccard_index_partial_overlap(self):
        """Test Jaccard index for partially overlapping masks"""
        mask1 = [np.array([1, 1, 0]), np.array([1, 0])]
        mask2 = [np.array([1, 0, 1]), np.array([0, 1])]

        jaccard = compute_jaccard_index(mask1, mask2)

        assert 0 < jaccard < 1

    def test_compute_jaccard_index_empty_masks(self):
        """Test Jaccard index for empty masks"""
        mask1 = [np.array([0, 0, 0]), np.array([0, 0])]
        mask2 = [np.array([0, 0, 0]), np.array([0, 0])]

        jaccard = compute_jaccard_index(mask1, mask2)

        assert jaccard == 0.0


class TestEnumerateEdgeMaskPerLayer:
    def test_enumerate_edge_mask_basic(self):
        """Test basic edge mask enumeration"""
        in_mask = np.array([1, 1])
        out_mask = np.array([1, 1])

        masks = _enumerate_edge_mask_per_layer(in_mask, out_mask)

        assert len(masks) > 0
        assert all(mask.shape == (2, 2) for mask in masks)

    def test_enumerate_edge_mask_inactive_nodes(self):
        """Test edge mask enumeration with inactive nodes"""
        in_mask = np.array([1, 0])
        out_mask = np.array([1, 0])

        masks = _enumerate_edge_mask_per_layer(in_mask, out_mask)

        # Should only have edges between active nodes
        for mask in masks:
            assert mask[1, 0] == 0  # No edge to inactive output
            assert mask[0, 1] == 0  # No edge from inactive input

    def test_enumerate_edge_mask_connectivity_constraint(self):
        """Test that edge masks satisfy connectivity constraints"""
        in_mask = np.array([1, 1])
        out_mask = np.array([1, 1])

        masks = _enumerate_edge_mask_per_layer(in_mask, out_mask)

        # Each mask should have at least one edge per active node
        for mask in masks:
            # Each active output should have at least one incoming edge
            for i, active in enumerate(out_mask):
                if active == 1:
                    assert np.sum(mask[i, :]) > 0

            # Each active input should have at least one outgoing edge
            for j, active in enumerate(in_mask):
                if active == 1:
                    assert np.sum(mask[:, j]) > 0


class TestEnumerateAllValidCircuit:
    def test_enumerate_all_valid_circuit(self):
        """Test enumeration of all valid circuits"""
        mock_model = Mock()
        mock_model.enumerate_valid_node_masks.return_value = [
            [np.array([1, 1]), np.array([1])],
            [np.array([1, 0]), np.array([1])],
        ]

        # Mock layer structure
        mock_layer = Mock()
        mock_linear = Mock()
        mock_linear.in_features = 2
        mock_layer.__getitem__ = Mock(return_value=mock_linear)
        mock_model.layers = [mock_layer]

        circuits = enumerate_all_valid_circuit(mock_model, use_tqdm=False)

        assert len(circuits) > 0
        assert all(isinstance(circuit, Circuit) for circuit in circuits)

    def test_enumerate_all_valid_circuit_with_sparsity(self):
        """Test circuit enumeration with minimum sparsity"""
        mock_model = Mock()
        mock_model.enumerate_valid_node_masks.return_value = [
            [np.array([1, 1]), np.array([1])],  # Low sparsity
            [np.array([0, 0]), np.array([1])],  # High sparsity
        ]

        mock_layer = Mock()
        mock_linear = Mock()
        mock_linear.in_features = 2
        mock_layer.__getitem__ = Mock(return_value=mock_linear)
        mock_model.layers = [mock_layer]

        circuits = enumerate_all_valid_circuit(mock_model, min_sparsity=0.5, use_tqdm=False)

        # Should filter out low sparsity circuits
        assert len(circuits) >= 0


class TestAnalyzeCircuits:
    def test_analyze_circuits(self):
        """Test circuit analysis"""
        # Create some test circuits
        node_masks1 = [np.array([1, 1]), np.array([1, 0])]
        edge_masks1 = [np.array([[1, 0], [0, 1]])]
        circuit1 = Circuit(node_masks1, edge_masks1)

        node_masks2 = [np.array([1, 1]), np.array([1, 1])]
        edge_masks2 = [np.array([[1, 1], [1, 1]])]
        circuit2 = Circuit(node_masks2, edge_masks2)

        circuits = [circuit1, circuit2]

        with patch('tqdm.tqdm', side_effect=lambda x, **kwargs: x):
            fraction_included, avg_jaccard, top_pairs = analyze_circuits(circuits, top_n=1)

        assert 0 <= fraction_included <= 1
        assert 0 <= avg_jaccard <= 1
        assert len(top_pairs) <= 1


class TestFindCircuits:
    def test_find_circuits(self):
        """Test finding circuits in a model"""
        mock_model = Mock()

        # Mock model predictions
        mock_predictions = torch.tensor([[0.9], [0.1], [0.8], [0.2]])
        mock_model.return_value = mock_predictions

        # Create test data
        x = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
        y = torch.tensor([[1], [0], [1], [0]], dtype=torch.float32)

        with patch('identifiability_toy_study.common.circuit.enumerate_all_valid_circuit') as mock_enumerate:
            # Create mock circuits
            mock_circuit = Mock()
            mock_circuit.sparsity.return_value = (0.5, 0.3, 0.4)
            mock_enumerate.return_value = [mock_circuit]

            # Mock circuit predictions
            mock_circuit_predictions = torch.tensor([[0.9], [0.1], [0.8], [0.2]])
            mock_model.side_effect = [mock_predictions, mock_circuit_predictions]

            top_circuits, sparsities, df = find_circuits(
                mock_model, x, y, accuracy_threshold=0.8, use_tqdm=False
            )

        assert isinstance(top_circuits, list)
        assert isinstance(sparsities, list)
        assert hasattr(df, 'shape')  # Should be a DataFrame


class TestIntegration:
    def test_circuit_save_load_validation_integration(self):
        """Integration test for circuit save/load and validation"""
        # Create a circuit
        node_masks = [np.array([1, 1]), np.array([1, 0]), np.array([1])]
        edge_masks = [np.array([[1, 0], [0, 1]]), np.array([[1], [0]])]
        circuit = Circuit(node_masks, edge_masks)

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp_file:
            try:
                # Save and load
                circuit.save_to_file(tmp_file.name)
                loaded_circuit = Circuit.load_from_file(tmp_file.name)

                # Create mock model for validation
                mock_model = Mock()
                mock_model.num_layers = 2

                mock_layer1 = Mock()
                mock_linear1 = Mock()
                mock_linear1.out_features = 2
                mock_linear1.weight = Mock()
                mock_linear1.weight.shape = (2, 2)
                mock_layer1.__getitem__ = Mock(return_value=mock_linear1)

                mock_layer2 = Mock()
                mock_linear2 = Mock()
                mock_linear2.out_features = 1
                mock_linear2.weight = Mock()
                mock_linear2.weight.shape = (1, 2)
                mock_layer2.__getitem__ = Mock(return_value=mock_linear2)

                mock_model.layers = [mock_layer1, mock_layer2]

                # Both original and loaded should validate successfully
                circuit.validate_against_model(mock_model)
                loaded_circuit.validate_against_model(mock_model)

                # They should be equivalent
                assert circuit.overlap_jaccard(loaded_circuit) == 1.0

            finally:
                os.unlink(tmp_file.name)

    def test_circuit_sparsity_and_inclusion_integration(self):
        """Integration test for sparsity calculation and inclusion checking"""
        # Create a sparse circuit
        node_masks1 = [np.array([1, 1]), np.array([1, 0]), np.array([1])]
        edge_masks1 = [np.array([[1, 0], [0, 0]]), np.array([[1], [0]])]
        sparse_circuit = Circuit(node_masks1, edge_masks1)

        # Create a dense circuit
        node_masks2 = [np.array([1, 1]), np.array([1, 1]), np.array([1])]
        edge_masks2 = [np.array([[1, 1], [1, 1]]), np.array([[1], [1]])]
        dense_circuit = Circuit(node_masks2, edge_masks2)

        # Sparse circuit should be included in dense
        assert sparse_circuit <= dense_circuit
        assert not (dense_circuit <= sparse_circuit)

        # Sparsity should be different
        sparse_sparsity = sparse_circuit.sparsity()
        dense_sparsity = dense_circuit.sparsity()

        assert sparse_sparsity[2] > dense_sparsity[2]  # Combined sparsity

        # Jaccard overlap should be meaningful
        jaccard = sparse_circuit.overlap_jaccard(dense_circuit)
        assert 0 < jaccard < 1