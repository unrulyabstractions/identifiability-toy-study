import copy
import itertools
import numpy as np
import pytest
import tempfile
import os
from unittest.mock import Mock, patch

from identifiability_toy_study.common.grounding import (
    Grounding,
    enumerate_tts,
    compute_local_tts,
    load_circuits,
)
from identifiability_toy_study.common.logic_gates import name_gate
from identifiability_toy_study.common.circuit import Circuit


class TestGrounding:
    def test_grounding_initialization(self):
        """Test Grounding class initialization"""
        assignments = [
            {"neuron_info": (1, 0), "gate_name": "AND"},
            {"neuron_info": (1, 1), "gate_name": "OR"},
        ]
        mock_circuit = Mock()

        grounding = Grounding(assignments, mock_circuit)

        assert grounding["(1,0)"] == "AND"
        assert grounding["(1,1)"] == "OR"
        assert grounding.circuit == mock_circuit

    def test_grounding_visualize(self):
        """Test Grounding visualize method"""
        assignments = [{"neuron_info": (1, 0), "gate_name": "AND"}]
        mock_circuit = Mock()
        mock_circuit.visualize = Mock()

        grounding = Grounding(assignments, mock_circuit)
        grounding.visualize(test_param="value")

        mock_circuit.visualize.assert_called_once_with(
            display_idx=True,
            labels=grounding,
            test_param="value"
        )


class TestEnumerateTts:
    def test_enumerate_tts_basic(self):
        """Test basic truth table enumeration"""
        mapping = {"input1": 0.1, "input2": 0.9, "input3": 0.5}

        result = enumerate_tts(mapping)

        assert len(result) == 2  # Two possible separators
        assert all("separator" in tt and "tt" in tt for tt in result)

    def test_enumerate_tts_single_value_raises_exception(self):
        """Test that single value mappings raise exception"""
        mapping = {"input1": 0.5, "input2": 0.5}

        with pytest.raises(Exception, match="Not allowed"):
            enumerate_tts(mapping)

    def test_enumerate_tts_separator_calculation(self):
        """Test separator calculation"""
        mapping = {"a": 0.2, "b": 0.8}
        result = enumerate_tts(mapping)

        assert len(result) == 1
        assert result[0]["separator"] == 0.5
        assert result[0]["tt"]["a"] == 0
        assert result[0]["tt"]["b"] == 1


class TestComputeLocalTts:
    def test_compute_local_tts_consistent(self):
        """Test local truth table computation with consistent data"""
        node_tt = {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 1}
        parents_tts = [
            {(0, 0): 0, (0, 1): 0, (1, 0): 1, (1, 1): 1},  # Parent 1
            {(0, 0): 0, (0, 1): 1, (1, 0): 0, (1, 1): 1},  # Parent 2
        ]
        global_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]

        result = compute_local_tts(node_tt, parents_tts, global_inputs)

        assert result is not None
        assert len(result) >= 1
        assert all(isinstance(tt, dict) for tt in result)

    def test_compute_local_tts_inconsistent(self):
        """Test local truth table computation with inconsistent data"""
        node_tt = {(0, 0): 0, (0, 1): 1, (1, 0): 0, (1, 1): 1}
        parents_tts = [
            {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0},  # All same parent values
        ]
        global_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]

        result = compute_local_tts(node_tt, parents_tts, global_inputs)

        assert result is None

    def test_compute_local_tts_missing_entries(self):
        """Test local truth table computation with missing entries"""
        node_tt = {(0, 0): 0, (1, 1): 1}  # Missing some entries
        parents_tts = [
            {(0, 0): 0, (0, 1): 0, (1, 0): 1, (1, 1): 1},
        ]
        global_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]

        result = compute_local_tts(node_tt, parents_tts, global_inputs)

        assert result is not None
        assert len(result) >= 1

    @patch('identifiability_toy_study.common.grounding.name_gate')
    def test_compute_local_tts_filters_xor_xnor(self, mock_name_gate):
        """Test that XOR and XNOR gates are filtered out"""
        mock_name_gate.side_effect = ["XOR", "XNOR", "AND"]

        node_tt = {(0, 0): 0, (1, 1): 1}
        parents_tts = [{(0, 0): 0, (0, 1): 0, (1, 0): 1, (1, 1): 1}]
        global_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]

        result = compute_local_tts(node_tt, parents_tts, global_inputs)

        # Should filter out XOR and XNOR, keeping only AND
        assert len(result) == 1


class TestLoadCircuits:
    def test_load_circuits_success(self):
        """Test successful circuit loading"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a circuits subdirectory
            circuits_dir = os.path.join(temp_dir, "circuits")
            os.makedirs(circuits_dir)

            # Create mock circuit files
            circuit_file1 = os.path.join(circuits_dir, "circuit-1.npz")
            circuit_file2 = os.path.join(circuits_dir, "circuit-2.npz")

            # Create simple npz files with minimal data
            np.savez(circuit_file1, node_mask_0=np.array([1, 0]), edge_mask_0=np.array([[1, 0]]))
            np.savez(circuit_file2, node_mask_0=np.array([0, 1]), edge_mask_0=np.array([[0, 1]]))

            with patch('identifiability_toy_study.common.grounding.Circuit') as MockCircuit:
                mock_circuit1 = Mock()
                mock_circuit2 = Mock()
                MockCircuit.load_from_file.side_effect = [mock_circuit1, mock_circuit2]

                circuits, ids = load_circuits(temp_dir)

                assert len(circuits) == 2
                assert len(ids) == 2
                assert 1 in ids
                assert 2 in ids
                assert MockCircuit.load_from_file.call_count == 2

    def test_load_circuits_no_circuit_folder(self):
        """Test load_circuits when circuit folder doesn't exist"""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(AssertionError, match="No circuit folder"):
                load_circuits(temp_dir)

    def test_load_circuits_empty_folder(self):
        """Test load_circuits with empty circuits folder"""
        with tempfile.TemporaryDirectory() as temp_dir:
            circuits_dir = os.path.join(temp_dir, "circuits")
            os.makedirs(circuits_dir)

            circuits, ids = load_circuits(temp_dir)

            assert len(circuits) == 0
            assert len(ids) == 0

    def test_load_circuits_ignores_non_npz_files(self):
        """Test that non-npz files are ignored"""
        with tempfile.TemporaryDirectory() as temp_dir:
            circuits_dir = os.path.join(temp_dir, "circuits")
            os.makedirs(circuits_dir)

            # Create a non-npz file
            with open(os.path.join(circuits_dir, "circuit-1.txt"), "w") as f:
                f.write("not a circuit")

            circuits, ids = load_circuits(temp_dir)

            assert len(circuits) == 0
            assert len(ids) == 0


class TestIntegration:
    def test_grounding_with_enumerate_tts(self):
        """Integration test between Grounding and enumerate_tts"""
        mapping = {"input1": 0.1, "input2": 0.9}
        tts = enumerate_tts(mapping)

        assignments = []
        for i, tt in enumerate(tts):
            assignments.append({
                "neuron_info": (1, i),
                "gate_name": f"GATE_{i}"
            })

        mock_circuit = Mock()
        grounding = Grounding(assignments, mock_circuit)

        assert len(grounding) == len(tts)
        for i in range(len(tts)):
            assert grounding[f"(1,{i})"] == f"GATE_{i}"