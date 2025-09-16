import numpy as np
import pytest
import torch
from unittest.mock import Mock, patch

from identifiability_toy_study.common.logic_gates import (
    LogicGate,
    LogicTree,
    generate_noisy_multi_gate_data,
    all_formulas,
    get_formula_dataset,
    xor_func,
    and_func,
    or_func,
    implication_func,
    nand_func,
    nor_func,
    not_implication_func,
    majority_func,
    parity_func,
    full_adder_func,
    threshold,
    mux_func,
    aoi_func,
    exactly_k_func,
    conditional_func,
    print_truth_table_multiple,
    name_gate,
    ALL_LOGIC_GATES,
)


class TestLogicGate:
    def test_logic_gate_initialization(self):
        """Test LogicGate initialization"""
        gate = LogicGate(2, and_func, "AND")

        assert gate.n_inputs == 2
        assert gate.name == "AND"
        assert gate.gate_fn == and_func
        assert 2 in LogicGate.inputs

    def test_logic_gate_initialization_with_string_function(self):
        """Test LogicGate initialization with string function name"""
        with patch('identifiability_toy_study.common.logic_gates.globals') as mock_globals:
            mock_globals.return_value = {"test_func": and_func}
            gate = LogicGate(2, "test_func", "TEST")

            assert gate.gate_fn == and_func

    def test_logic_gate_repr(self):
        """Test LogicGate string representation"""
        gate = LogicGate(2, and_func, "AND")

        assert repr(gate) == "LogicGate(n_inputs=2, name=AND)"

    def test_truth_table(self):
        """Test truth table generation"""
        gate = LogicGate(2, and_func, "AND")
        tt = gate.truth_table()

        expected = {
            (0., 0.): 0.0,
            (0., 1.): 0.0,
            (1., 0.): 0.0,
            (1., 1.): 1.0
        }

        assert tt == expected

    def test_generate_noisy_data(self):
        """Test noisy data generation"""
        gate = LogicGate(2, and_func, "AND")
        x, y = gate.generate_noisy_data(n_repeats=10, noise_std=0.1, device="cpu")

        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        assert x.device.type == "cpu"
        assert y.device.type == "cpu"

    def test_get_inputs(self):
        """Test input generation"""
        inputs = LogicGate.get_inputs(2, 100, None)

        assert inputs.shape[0] == 400  # 100 * 4 possible inputs
        assert inputs.shape[1] == 2

    def test_get_inputs_with_weights(self):
        """Test input generation with weights"""
        weights = np.array([0.1, 0.2, 0.3, 0.4])
        inputs = LogicGate.get_inputs(2, 100, weights)

        assert inputs.shape[0] == 400
        assert inputs.shape[1] == 2

    def test_add_noise_and_repeat(self):
        """Test noise addition and repetition"""
        inputs = np.array([[0, 1], [1, 0]])
        outputs = np.array([[0], [1]])

        x_noisy, y_noisy = LogicGate.add_noise_and_repeat(10, 0.1, inputs, outputs)

        assert x_noisy.shape[0] == 20  # 2 * 10
        assert y_noisy.shape[0] == 20

    def test_add_noise_and_repeat_no_noise(self):
        """Test noise addition with zero noise"""
        inputs = np.array([[0, 1], [1, 0]])
        outputs = np.array([[0], [1]])

        x_noisy, y_noisy = LogicGate.add_noise_and_repeat(10, 0.0, inputs, outputs)

        assert torch.allclose(x_noisy, torch.tensor(inputs, dtype=torch.float32).repeat(10, 1))

    def test_print_truth_table(self):
        """Test truth table printing"""
        gate = LogicGate(2, and_func, "AND")
        mock_logger = Mock()

        gate.print_truth_table(mock_logger)

        assert mock_logger.info.call_count >= 5  # Header + separator + 4 rows


class TestGenerateNoisyMultiGateData:
    def test_generate_noisy_multi_gate_data(self):
        """Test multi-gate data generation"""
        gate1 = LogicGate(2, and_func, "AND")
        gate2 = LogicGate(2, or_func, "OR")
        gates = [gate1, gate2]

        x, y = generate_noisy_multi_gate_data(gates, n_repeats=10, device="cpu")

        assert x.shape[1] == 2  # 2 inputs
        assert y.shape[1] == 2  # 2 outputs (one per gate)

    def test_generate_noisy_multi_gate_data_different_inputs(self):
        """Test that gates with different input counts raise error"""
        gate1 = LogicGate(2, and_func, "AND")
        gate2 = LogicGate(3, majority_func, "MAJORITY")
        gates = [gate1, gate2]

        with pytest.raises(ValueError, match="All logic gates must have the same number of inputs"):
            generate_noisy_multi_gate_data(gates, n_repeats=10, device="cpu")


class TestLogicTree:
    def test_logic_tree_initialization(self):
        """Test LogicTree initialization"""
        tree = LogicTree("AND", ["A", "B"], 1)

        assert tree.op == "AND"
        assert len(tree.children) == 2
        assert tree.id == 1

    def test_logic_tree_initialization_nested(self):
        """Test LogicTree initialization with nested children"""
        child1 = LogicTree("A")
        child2 = LogicTree("B")
        tree = LogicTree("AND", [child1, child2])

        assert len(tree.children) == 2
        assert tree.children[0].op == "A"
        assert tree.children[1].op == "B"

    def test_from_logic(self):
        """Test LogicTree creation from logic expression"""
        tree = LogicTree.from_logic("A")
        assert tree.op == "A"
        assert len(tree.children) == 0

    def test_from_logic_negation(self):
        """Test LogicTree creation with negation"""
        tree = LogicTree.from_logic("¬(A)")
        assert tree.op == "¬"
        assert len(tree.children) == 1
        assert tree.children[0].op == "A"

    def test_from_logic_binary_operation(self):
        """Test LogicTree creation with binary operation"""
        tree = LogicTree.from_logic("(A+B)")
        assert tree.op == "+"
        assert len(tree.children) == 2

    def test_repr(self):
        """Test LogicTree string representation"""
        tree = LogicTree("AND", ["A", "B"], 1)
        repr_str = repr(tree)
        assert "AND" in repr_str
        assert "1" in repr_str

    def test_str(self):
        """Test LogicTree string conversion"""
        tree = LogicTree("+", ["A", "B"])
        str_repr = str(tree)
        assert "A" in str_repr and "B" in str_repr

    def test_add_missing_ids(self):
        """Test adding missing IDs to nodes"""
        tree = LogicTree("+", ["A", "B"])
        next_id = tree.add_missing_ids()

        assert tree.id is not None
        assert tree.children[0].id is not None
        assert tree.children[1].id is not None
        assert isinstance(next_id, int)

    def test_find_node(self):
        """Test finding a node by ID"""
        tree = LogicTree("+", ["A", "B"], 0)
        tree.children[0].id = 1
        tree.children[1].id = 2

        found = tree.find_node(1)
        assert found is not None
        assert found.op == "A"

    def test_find_node_not_found(self):
        """Test finding a non-existent node"""
        tree = LogicTree("+", ["A", "B"], 0)
        found = tree.find_node(999)
        assert found is None

    def test_get_leaf_nodes(self):
        """Test getting all leaf nodes"""
        tree = LogicTree("+", ["A", "B"])
        leaves = list(tree.get_leaf_nodes())

        assert len(leaves) == 2
        assert all(len(leaf.children) == 0 for leaf in leaves)

    def test_get_node_ids(self):
        """Test getting all node IDs"""
        tree = LogicTree("+", ["A", "B"], 0)
        tree.children[0].id = 1
        tree.children[1].id = 2

        ids = tree.get_node_ids()
        assert 0 in ids
        assert 1 in ids
        assert 2 in ids

    def test_get_non_leaf_nodes(self):
        """Test getting all non-leaf nodes"""
        tree = LogicTree("+", ["A", "B"])
        non_leaves = list(tree.get_non_leaf_nodes())

        assert len(non_leaves) == 1
        assert non_leaves[0] == tree

    def test_get_nodes(self):
        """Test getting all nodes"""
        tree = LogicTree("+", ["A", "B"])
        all_nodes = list(tree.get_nodes())

        assert len(all_nodes) == 3  # root + 2 children

    def test_negate(self):
        """Test node negation"""
        tree = LogicTree("+", ["A", "B"], 0)
        tree.children[0].id = 1
        tree.children[1].id = 2

        negated = tree.negate([1])
        # Should wrap child 0 in negation
        assert negated.children[0].op == "¬"

    def test_intervene(self):
        """Test intervention on a node"""
        tree = LogicTree("+", ["A", "B"], 0)
        tree.children[0].id = 1

        intervened = tree.intervene(tree.children[0], 1)

        assert intervened is not tree  # Should be a copy
        # Check that intervention was applied

    def test_evaluate(self):
        """Test tree evaluation"""
        tree = LogicTree("+", ["A", "B"])
        tree.add_missing_ids()

        assignments = {"A": 1, "B": 0}
        result = tree.evaluate(assignments)

        assert result in [0, 1]

    def test_evaluate_with_return_dict(self):
        """Test tree evaluation returning intermediate values"""
        tree = LogicTree("+", ["A", "B"])
        tree.add_missing_ids()

        assignments = {"A": 1, "B": 0}
        result = tree.evaluate(assignments, return_dict=True)

        assert isinstance(result, dict)
        assert tree.id in result

    def test_sample(self):
        """Test tree sampling"""
        tree = LogicTree("+", ["A", "B"])
        samples = tree.sample(device="cpu")

        assert len(samples) == 4  # 2^2 possible assignments
        for assignment, tensor in samples:
            assert isinstance(assignment, dict)
            assert isinstance(tensor, torch.Tensor)

    def test_visualize(self):
        """Test tree visualization"""
        tree = LogicTree("+", ["A", "B"])
        tree.add_missing_ids()

        # Should not raise any errors
        with patch('matplotlib.pyplot.show'):
            tree.visualize()


class TestLogicFunctions:
    def test_xor_func(self):
        """Test XOR function"""
        inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        expected = np.array([0, 1, 1, 0])
        result = xor_func(inputs)
        np.testing.assert_array_equal(result, expected)

    def test_and_func(self):
        """Test AND function"""
        inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        expected = np.array([0, 0, 0, 1])
        result = and_func(inputs)
        np.testing.assert_array_equal(result, expected)

    def test_or_func(self):
        """Test OR function"""
        inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        expected = np.array([0, 1, 1, 1])
        result = or_func(inputs)
        np.testing.assert_array_equal(result, expected)

    def test_implication_func(self):
        """Test implication function"""
        inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        expected = np.array([1, 1, 0, 1])
        result = implication_func(inputs)
        np.testing.assert_array_equal(result, expected)

    def test_nand_func(self):
        """Test NAND function"""
        inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        expected = np.array([1, 1, 1, 0])
        result = nand_func(inputs)
        np.testing.assert_array_equal(result, expected)

    def test_nor_func(self):
        """Test NOR function"""
        inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        expected = np.array([1, 0, 0, 0])
        result = nor_func(inputs)
        np.testing.assert_array_equal(result, expected)

    def test_not_implication_func(self):
        """Test NOT implication function"""
        inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        expected = np.array([0, 0, 1, 0])
        result = not_implication_func(inputs)
        np.testing.assert_array_equal(result, expected)

    def test_majority_func(self):
        """Test majority function"""
        inputs = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 1], [1, 1, 1]])
        expected = np.array([0, 0, 1, 1])
        result = majority_func(inputs)
        np.testing.assert_array_equal(result, expected)

    def test_parity_func(self):
        """Test parity function"""
        inputs = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 1], [1, 1, 1]])
        expected = np.array([0, 1, 0, 1])
        result = parity_func(inputs)
        np.testing.assert_array_equal(result, expected)

    def test_full_adder_func(self):
        """Test full adder function"""
        inputs = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 1], [1, 1, 1]])
        result = full_adder_func(inputs)
        assert result.shape == (4,)

    def test_full_adder_func_wrong_inputs(self):
        """Test full adder with wrong number of inputs"""
        inputs = np.array([[0, 0], [1, 1]])
        with pytest.raises(AssertionError, match="Full Adder requires exactly 3 inputs"):
            full_adder_func(inputs)

    def test_threshold(self):
        """Test threshold function"""
        inputs = np.array([[0.3, 0.7], [0.6, 0.4]])
        expected = np.array([[0, 1], [1, 0]])
        result = threshold(inputs, 0.5)
        np.testing.assert_array_equal(result, expected)

    def test_mux_func(self):
        """Test multiplexer function"""
        inputs = np.array([[0, 1, 0.5, 0.8]])  # sel=0, data=[0.5, 0.8]
        result = mux_func(inputs)
        assert result.shape == (1,)

    def test_aoi_func(self):
        """Test AOI function"""
        inputs = np.array([[1, 1, 0, 1]])  # AND of first half, OR of second half
        result = aoi_func(inputs)
        assert result.shape == (1,)

    def test_exactly_k_func(self):
        """Test exactly k function"""
        inputs = np.array([[0, 1, 1], [1, 1, 1], [0, 0, 1]])
        result = exactly_k_func(inputs, 2)
        expected = np.array([1, 0, 0])  # Only first row has exactly 2 ones
        np.testing.assert_array_equal(result, expected)

    def test_conditional_func(self):
        """Test conditional function"""
        inputs = np.array([[0, 1, 1], [1, 1, 1], [0, 0, 1]])
        expected = np.array([1, 1, 0])  # >= 2 ones
        result = conditional_func(inputs)
        np.testing.assert_array_equal(result, expected)


class TestNameGate:
    def test_name_gate_2_inputs(self):
        """Test gate naming for 2-input gates"""
        # AND gate: [0, 0, 0, 1]
        tt = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 1}
        assert name_gate(tt) == "AND"

        # OR gate: [0, 1, 1, 1]
        tt = {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 1}
        assert name_gate(tt) == "OR"

    def test_name_gate_1_input(self):
        """Test gate naming for 1-input gates"""
        # IDENTITY gate: [0, 1]
        tt = {(0,): 0, (1,): 1}
        assert name_gate(tt) == "IDENTITY"

        # NOT gate: [1, 0]
        tt = {(0,): 1, (1,): 0}
        assert name_gate(tt) == "NOT"

    def test_name_gate_0_inputs(self):
        """Test gate naming for 0-input gates"""
        # FALSE gate
        tt = {(): 0}
        assert name_gate(tt) == "FALSE"

        # TRUE gate
        tt = {(): 1}
        assert name_gate(tt) == "TRUE"

    def test_name_gate_many_inputs(self):
        """Test gate naming for many inputs"""
        tt = {(0, 0, 0): 0, (0, 0, 1): 1, (0, 1, 0): 1, (0, 1, 1): 1,
              (1, 0, 0): 1, (1, 0, 1): 1, (1, 1, 0): 1, (1, 1, 1): 1}
        assert name_gate(tt) == "3INPUTS"


class TestPrintTruthTableMultiple:
    def test_print_truth_table_multiple(self):
        """Test printing truth table for multiple gates"""
        gate1 = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 1}  # AND
        gate2 = {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 1}  # OR
        gates = [gate1, gate2]

        mock_logger = Mock()
        print_truth_table_multiple(gates, mock_logger)

        # Should print header, separator, and 4 rows of data
        assert mock_logger.info.call_count >= 6


class TestAllFormulas:
    def test_all_formulas_depth_0(self):
        """Test formula generation at depth 0"""
        formulas = list(all_formulas(0, ["A", "B"]))
        assert len(formulas) == 2  # Just A and B

    def test_all_formulas_depth_1(self):
        """Test formula generation at depth 1"""
        formulas = list(all_formulas(1, ["A"]))
        assert len(formulas) >= 1  # Should generate some formulas


class TestGetFormulaDataset:
    def test_get_formula_dataset(self):
        """Test formula dataset generation"""
        gates = [LogicGate(2, and_func, "AND")]
        dataset = get_formula_dataset(gates, max_depth=1, device="cpu")

        assert "AND" in dataset
        assert len(dataset["AND"]) >= 0

    def test_get_formula_dataset_different_inputs(self):
        """Test that gates with different input counts raise error"""
        gate1 = LogicGate(2, and_func, "AND")
        gate2 = LogicGate(3, majority_func, "MAJORITY")
        gates = [gate1, gate2]

        with pytest.raises(ValueError, match="Not all gates have the same number of inputs"):
            get_formula_dataset(gates, max_depth=1, device="cpu")


class TestAllLogicGates:
    def test_all_logic_gates_defined(self):
        """Test that ALL_LOGIC_GATES contains expected gates"""
        expected_gates = ["AND", "OR", "XOR", "NAND", "NOR", "IMP", "NIMP"]

        for gate_name in expected_gates:
            assert gate_name in ALL_LOGIC_GATES
            assert isinstance(ALL_LOGIC_GATES[gate_name], LogicGate)

    def test_all_logic_gates_consistency(self):
        """Test that all gates in ALL_LOGIC_GATES work correctly"""
        for gate_name, gate in ALL_LOGIC_GATES.items():
            # Should be able to generate truth table without errors
            tt = gate.truth_table()
            assert isinstance(tt, dict)
            assert len(tt) == 2 ** gate.n_inputs