import numpy as np
import pytest
import tempfile
import os
import torch
import torch.nn as nn
from unittest.mock import Mock, patch

from identifiability_toy_study.common.neural_model import (
    MLP,
    debug_device,
    ACTIVATION_FUNCTIONS,
)
from identifiability_toy_study.common.causal import Intervention, PatchShape
from identifiability_toy_study.common.circuit import Circuit


class TestActivationFunctions:
    def test_activation_functions_defined(self):
        """Test that all activation functions are defined"""
        expected_activations = ["leaky_relu", "relu", "tanh", "sigmoid"]

        for activation in expected_activations:
            assert activation in ACTIVATION_FUNCTIONS
            assert issubclass(ACTIVATION_FUNCTIONS[activation], nn.Module)


class TestDebugDevice:
    def test_debug_device_single_tensor(self):
        """Test debug device function with single tensor"""
        mock_logger = Mock()
        tensor = torch.tensor([1, 2, 3])

        result = debug_device(mock_logger, tensor, "cpu", "test context")

        assert isinstance(result, bool)
        mock_logger.info.assert_called()

    def test_debug_device_multiple_tensors(self):
        """Test debug device function with multiple tensors"""
        mock_logger = Mock()
        tensors = [torch.tensor([1, 2]), torch.tensor([3, 4])]

        result = debug_device(mock_logger, tensors, "cpu", "test context")

        assert isinstance(result, bool)
        assert mock_logger.info.call_count >= 2

    def test_debug_device_none_tensor(self):
        """Test debug device function with None tensor"""
        mock_logger = Mock()

        result = debug_device(mock_logger, [None], "cpu", "test context")

        assert isinstance(result, bool)

    def test_debug_device_wrong_device(self):
        """Test debug device function with wrong device"""
        mock_logger = Mock()
        tensor = torch.tensor([1, 2, 3], device="cpu")

        result = debug_device(mock_logger, tensor, "cuda", "test context")

        # Should return False when device doesn't match
        mock_logger.info.assert_called()


class TestMLP:
    def test_mlp_initialization(self):
        """Test MLP initialization"""
        mlp = MLP(
            hidden_sizes=[4, 3],
            input_size=2,
            output_size=1,
            activation="relu",
            device="cpu"
        )

        assert mlp.input_size == 2
        assert mlp.hidden_sizes == [4, 3]
        assert mlp.output_size == 1
        assert mlp.layer_sizes == [2, 4, 3, 1]
        assert mlp.num_layers == 3
        assert len(mlp.layers) == 3

    def test_mlp_initialization_with_debug(self):
        """Test MLP initialization with debug mode"""
        mock_logger = Mock()

        mlp = MLP(
            hidden_sizes=[2],
            input_size=2,
            output_size=1,
            debug=True,
            logger=mock_logger,
            device="cpu"
        )

        assert mlp.debug is True
        mock_logger.info.assert_called()

    def test_mlp_forward_basic(self):
        """Test basic forward pass"""
        mlp = MLP(hidden_sizes=[2], input_size=2, output_size=1, device="cpu")
        x = torch.tensor([[1.0, 2.0]], dtype=torch.float32)

        output = mlp(x)

        assert output.shape == (1, 1)
        assert output.dtype == torch.float32

    def test_mlp_forward_with_activations(self):
        """Test forward pass returning activations"""
        mlp = MLP(hidden_sizes=[2], input_size=2, output_size=1, device="cpu")
        x = torch.tensor([[1.0, 2.0]], dtype=torch.float32)

        activations = mlp(x, return_activations=True)

        assert isinstance(activations, list)
        assert len(activations) == 3  # Input + hidden + output
        assert all(isinstance(act, torch.Tensor) for act in activations)

    def test_mlp_forward_with_circuit(self):
        """Test forward pass with circuit"""
        mlp = MLP(hidden_sizes=[2], input_size=2, output_size=1, device="cpu")
        x = torch.tensor([[1.0, 2.0]], dtype=torch.float32)

        # Create a simple circuit
        node_masks = [np.array([1, 1]), np.array([1, 0]), np.array([1])]
        edge_masks = [np.array([[1, 0], [0, 1]]), np.array([[1], [0]])]
        circuit = Circuit(node_masks, edge_masks)

        output = mlp(x, circuit=circuit)

        assert output.shape == (1, 1)

    def test_mlp_forward_with_intervention(self):
        """Test forward pass with intervention"""
        mlp = MLP(hidden_sizes=[2], input_size=2, output_size=1, device="cpu")
        x = torch.tensor([[1.0, 2.0]], dtype=torch.float32)

        # Create a simple intervention
        patches = {
            PatchShape(layers=(1,), indices=(0,), axis="neuron"): ("mul", torch.tensor([0.5]))
        }
        intervention = Intervention(patches)

        output = mlp(x, intervention=intervention)

        assert output.shape == (1, 1)

    def test_mlp_out_features(self):
        """Test out_features method"""
        mlp = MLP(hidden_sizes=[4, 3], input_size=2, output_size=1, device="cpu")

        assert mlp.out_features(0) == 4  # First hidden layer
        assert mlp.out_features(1) == 3  # Second hidden layer
        assert mlp.out_features(2) == 1  # Output layer

    def test_mlp_get_states(self):
        """Test get_states method"""
        mlp = MLP(hidden_sizes=[2], input_size=2, output_size=1, device="cpu")
        x = torch.tensor([[1.0, 2.0]], dtype=torch.float32)

        states = {1: [0, 1]}  # Get neurons 0 and 1 from layer 1
        result = mlp.get_states(x, states)

        assert isinstance(result, dict)
        assert (1, (0, 1)) in result
        assert result[(1, (0, 1))].shape[1] == 2  # Two neurons

    def test_mlp_get_patch(self):
        """Test get_patch method"""
        mlp = MLP(hidden_sizes=[2], input_size=2, output_size=1, device="cpu")
        x = torch.tensor([[1.0, 2.0]], dtype=torch.float32)

        shape = PatchShape(layers=(1,), indices=(0, 1), axis="neuron")
        patch = mlp.get_patch(x, shape)

        assert isinstance(patch, Intervention)
        assert len(patch.patches) == 1

    def test_mlp_get_patch_edge_axis_raises(self):
        """Test that get_patch raises error for edge axis"""
        mlp = MLP(hidden_sizes=[2], input_size=2, output_size=1, device="cpu")
        x = torch.tensor([[1.0, 2.0]], dtype=torch.float32)

        shape = PatchShape(layers=(0,), indices=(), axis="edge")

        with pytest.raises(NotImplementedError):
            mlp.get_patch(x, shape)

    def test_mlp_hidden_sizes(self):
        """Test hidden_sizes method"""
        mlp = MLP(hidden_sizes=[4, 3], input_size=2, output_size=1, device="cpu")

        hidden_sizes = mlp.hidden_sizes()

        assert hidden_sizes == [4, 3]

    def test_mlp_save_and_load(self):
        """Test model saving and loading"""
        mlp = MLP(hidden_sizes=[2], input_size=2, output_size=1, device="cpu")

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp_file:
            try:
                mlp.save_to_file(tmp_file.name)

                loaded_mlp = MLP.load_from_file(tmp_file.name)

                assert loaded_mlp.input_size == mlp.input_size
                assert loaded_mlp.hidden_sizes == mlp.hidden_sizes
                assert loaded_mlp.output_size == mlp.output_size
                assert loaded_mlp.activation == mlp.activation

            finally:
                os.unlink(tmp_file.name)

    def test_mlp_visualize(self):
        """Test model visualization"""
        mlp = MLP(hidden_sizes=[2], input_size=2, output_size=1, device="cpu")

        with patch('matplotlib.pyplot.show'), \
             patch('networkx.draw_networkx_nodes'), \
             patch('networkx.draw_networkx_edges'), \
             patch('networkx.draw_networkx_labels'), \
             patch('networkx.draw_networkx_edge_labels'):

            # Should not raise any exceptions
            mlp.visualize()

    def test_mlp_getitem_slice(self):
        """Test MLP slicing"""
        mlp = MLP(hidden_sizes=[4, 3, 2], input_size=2, output_size=1, device="cpu")

        # Get a submodel
        submodel = mlp[1:3]

        assert isinstance(submodel, MLP)
        assert submodel.input_size == 4  # Output of first layer
        assert submodel.output_size == 2  # Output of third layer
        assert len(submodel.layers) == 2

    def test_mlp_getitem_slice_in_place(self):
        """Test MLP slicing with parameter sharing"""
        mlp = MLP(hidden_sizes=[4, 3], input_size=2, output_size=1, device="cpu")

        submodel = mlp[0:2, True]  # in_place=True

        # Parameters should be shared
        assert submodel.layers[0][0].weight is mlp.layers[0][0].weight
        assert submodel.layers[1][0].weight is mlp.layers[1][0].weight

    def test_mlp_getitem_single_index(self):
        """Test MLP single layer access"""
        mlp = MLP(hidden_sizes=[2], input_size=2, output_size=1, device="cpu")

        layer = mlp[0]

        assert isinstance(layer, nn.Sequential)

    def test_mlp_getitem_single_index_copy(self):
        """Test MLP single layer access with copy"""
        mlp = MLP(hidden_sizes=[2], input_size=2, output_size=1, device="cpu")

        layer = mlp[0, False]  # in_place=False (copy)

        # Should be a deep copy
        assert layer is not mlp.layers[0]

    def test_mlp_do_train(self):
        """Test model training"""
        mlp = MLP(hidden_sizes=[2], input_size=2, output_size=1, device="cpu")

        # Create simple training data
        x = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
        y = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
        x_val = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
        y_val = torch.tensor([[0.5]], dtype=torch.float32)

        loss = mlp.do_train(
            x=x,
            y=y,
            x_val=x_val,
            y_val=y_val,
            batch_size=2,
            learning_rate=0.01,
            epochs=5,
            loss_target=1.0,
            val_frequency=5
        )

        assert isinstance(loss, float)
        assert loss >= 0

    def test_mlp_do_train_with_logger(self):
        """Test model training with logger"""
        mlp = MLP(hidden_sizes=[2], input_size=2, output_size=1, device="cpu")
        mock_logger = Mock()

        x = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
        y = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
        x_val = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
        y_val = torch.tensor([[0.5]], dtype=torch.float32)

        loss = mlp.do_train(
            x=x,
            y=y,
            x_val=x_val,
            y_val=y_val,
            batch_size=2,
            learning_rate=0.01,
            epochs=5,
            val_frequency=2,
            logger=mock_logger
        )

        assert isinstance(loss, float)
        mock_logger.info.assert_called()

    def test_mlp_do_eval(self):
        """Test model evaluation"""
        mlp = MLP(hidden_sizes=[2], input_size=2, output_size=1, device="cpu")

        x_test = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
        y_test = torch.tensor([[0.0], [1.0]], dtype=torch.float32)

        accuracy = mlp.do_eval(x_test, y_test)

        assert isinstance(accuracy, float)
        assert 0 <= accuracy <= 1

    def test_mlp_separate_into_k_mlps(self):
        """Test separation into K MLPs"""
        mlp = MLP(hidden_sizes=[2], input_size=2, output_size=3, device="cpu")

        separate_models = mlp.separate_into_k_mlps()

        assert len(separate_models) == 3
        assert all(isinstance(model, MLP) for model in separate_models)
        assert all(model.output_size == 1 for model in separate_models)

    def test_mlp_enumerate_valid_node_masks(self):
        """Test enumeration of valid node masks"""
        mlp = MLP(hidden_sizes=[2], input_size=2, output_size=1, device="cpu")

        masks = mlp.enumerate_valid_node_masks()

        assert len(masks) > 0
        # Should have masks for hidden layer (2 nodes) and output layer (1 node)
        # 2^2 * 2^1 = 8 total combinations
        assert len(masks) == 8

    def test_mlp_apply_neuron_patches_inplace(self):
        """Test neuron patch application"""
        mlp = MLP(hidden_sizes=[2], input_size=2, output_size=1, device="cpu")

        h = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        patches = [("mul", (0,), torch.tensor([0.5]))]

        result = mlp._apply_neuron_patches_inplace(h, patches)

        assert result.shape == h.shape
        assert result[0, 0] == 0.5  # First element should be multiplied by 0.5

    def test_mlp_apply_neuron_patches_set_mode(self):
        """Test neuron patch application with set mode"""
        mlp = MLP(hidden_sizes=[2], input_size=2, output_size=1, device="cpu")

        h = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        patches = [("set", (1,), torch.tensor([0.0]))]

        result = mlp._apply_neuron_patches_inplace(h, patches)

        assert result[0, 1] == 0.0  # Second element should be set to 0

    def test_mlp_apply_neuron_patches_add_mode(self):
        """Test neuron patch application with add mode"""
        mlp = MLP(hidden_sizes=[2], input_size=2, output_size=1, device="cpu")

        h = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        patches = [("add", (0,), torch.tensor([1.0]))]

        result = mlp._apply_neuron_patches_inplace(h, patches)

        assert result[0, 0] == 2.0  # First element should be 1 + 1 = 2

    def test_mlp_apply_neuron_patches_invalid_mode(self):
        """Test neuron patch application with invalid mode"""
        mlp = MLP(hidden_sizes=[2], input_size=2, output_size=1, device="cpu")

        h = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        patches = [("invalid", (0,), torch.tensor([0.5]))]

        with pytest.raises(ValueError, match="Unknown mode"):
            mlp._apply_neuron_patches_inplace(h, patches)

    def test_mlp_apply_edge_patches(self):
        """Test edge patch application"""
        mlp = MLP(hidden_sizes=[2], input_size=2, output_size=1, device="cpu")

        W = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        patches = [("mul", torch.tensor([[0.5, 1.0], [1.0, 0.5]]))]

        result = mlp._apply_edge_patches(W, patches)

        assert result.shape == W.shape
        # Should be element-wise multiplication

    def test_mlp_apply_edge_patches_set_mode(self):
        """Test edge patch application with set mode"""
        mlp = MLP(hidden_sizes=[2], input_size=2, output_size=1, device="cpu")

        W = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        new_values = torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype=torch.float32)
        patches = [("set", new_values)]

        result = mlp._apply_edge_patches(W, patches)

        torch.testing.assert_close(result, new_values)

    def test_mlp_apply_edge_patches_invalid_mode(self):
        """Test edge patch application with invalid mode"""
        mlp = MLP(hidden_sizes=[2], input_size=2, output_size=1, device="cpu")

        W = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        patches = [("invalid", torch.tensor([[0.5, 1.0]]))]

        with pytest.raises(ValueError, match="Unknown mode"):
            mlp._apply_edge_patches(W, patches)


class TestMLPIntegration:
    def test_mlp_training_and_evaluation_integration(self):
        """Integration test for training and evaluation"""
        mlp = MLP(hidden_sizes=[4], input_size=2, output_size=1, device="cpu")

        # Create XOR-like data
        x_train = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
        y_train = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float32)

        # Train the model
        loss = mlp.do_train(
            x=x_train,
            y=y_train,
            x_val=x_train,
            y_val=y_train,
            batch_size=4,
            learning_rate=0.1,
            epochs=50,
            loss_target=0.5
        )

        # Evaluate the model
        accuracy = mlp.do_eval(x_train, y_train)

        assert isinstance(loss, float)
        assert isinstance(accuracy, float)
        assert 0 <= accuracy <= 1

    def test_mlp_save_load_train_integration(self):
        """Integration test for save/load with training"""
        mlp1 = MLP(hidden_sizes=[2], input_size=2, output_size=1, device="cpu")

        # Train the model a bit
        x = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
        y = torch.tensor([[0.0], [1.0]], dtype=torch.float32)

        mlp1.do_train(
            x=x, y=y, x_val=x, y_val=y,
            batch_size=2, learning_rate=0.01, epochs=5
        )

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp_file:
            try:
                # Save and load
                mlp1.save_to_file(tmp_file.name)
                mlp2 = MLP.load_from_file(tmp_file.name)

                # Models should produce same outputs
                output1 = mlp1(x)
                output2 = mlp2(x)

                torch.testing.assert_close(output1, output2, atol=1e-6, rtol=1e-6)

            finally:
                os.unlink(tmp_file.name)

    def test_mlp_circuit_intervention_integration(self):
        """Integration test for circuit and intervention combination"""
        mlp = MLP(hidden_sizes=[3], input_size=2, output_size=1, device="cpu")
        x = torch.tensor([[1.0, 2.0]], dtype=torch.float32)

        # Create circuit
        node_masks = [np.array([1, 1]), np.array([1, 0, 1]), np.array([1])]
        edge_masks = [np.array([[1, 0], [0, 1], [1, 1]]), np.array([[1], [0], [1]])]
        circuit = Circuit(node_masks, edge_masks)

        # Create intervention
        patches = {
            PatchShape(layers=(1,), indices=(0,), axis="neuron"): ("mul", torch.tensor([0.5]))
        }
        intervention = Intervention(patches)

        # Should be able to apply both circuit and intervention
        output = mlp(x, circuit=circuit, intervention=intervention)

        assert output.shape == (1, 1)

    def test_mlp_separate_and_train_integration(self):
        """Integration test for separating into multiple MLPs and training"""
        mlp = MLP(hidden_sizes=[3], input_size=2, output_size=2, device="cpu")

        # Create multi-output data
        x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
        y = torch.tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)

        # Separate into individual models
        separate_models = mlp.separate_into_k_mlps()

        assert len(separate_models) == 2

        # Train each separated model
        for i, model in enumerate(separate_models):
            y_single = y[:, i:i+1]  # Single output
            loss = model.do_train(
                x=x, y=y_single, x_val=x, y_val=y_single,
                batch_size=4, learning_rate=0.01, epochs=5
            )
            assert isinstance(loss, float)

    def test_mlp_patch_application_integration(self):
        """Integration test for various patch applications"""
        mlp = MLP(hidden_sizes=[3], input_size=2, output_size=1, device="cpu")
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)

        # Test neuron patches with different broadcasting
        patches_2d = {
            PatchShape(layers=(1,), indices=(0, 1), axis="neuron"):
            ("set", torch.tensor([[0.5, 0.6], [0.7, 0.8]]))  # [B, k] format
        }
        intervention_2d = Intervention(patches_2d)
        output_2d = mlp(x, intervention=intervention_2d)

        patches_1d = {
            PatchShape(layers=(1,), indices=(0, 1), axis="neuron"):
            ("set", torch.tensor([0.5, 0.6]))  # [k] format
        }
        intervention_1d = Intervention(patches_1d)
        output_1d = mlp(x, intervention=intervention_1d)

        assert output_2d.shape == (2, 1)
        assert output_1d.shape == (2, 1)