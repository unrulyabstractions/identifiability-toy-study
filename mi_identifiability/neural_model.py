import copy
import itertools
from collections import defaultdict
from typing import Dict, Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

ACTIVATION_FUNCTIONS = {
    'leaky_relu': nn.LeakyReLU,
    'relu': nn.ReLU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid
}


class MLP(nn.Module):
    """
    A class implementing a simple multi-layer perceptron model.
    """
    def __init__(self, hidden_sizes: list, input_size=2, output_size=1, activation='leaky_relu', device='cpu'):
        """
        Initialize the MLP model.

        Args:
            hidden_sizes: A list of sizes of the hidden layers (in neurons), not including the input and output layers
            input_size: The size of the input layer
            output_size: The size of the output layer
            activation: The activation function to use
            device: The device to run the model on
        """
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.activation = activation

        self.layers = nn.ModuleList(
            nn.Sequential(
                nn.Linear(in_size, out_size),
                ACTIVATION_FUNCTIONS[activation]() if idx < len(self.layer_sizes) - 1 else nn.Identity()
            ) for idx, (in_size, out_size) in enumerate(zip(self.layer_sizes, self.layer_sizes[1:]))
        )

        self.num_layers = len(self.layers)
        self.device = device
        self.to(device)

    def save_to_file(self, filepath):
        """
        Save the model's configuration and state_dict to a file.
        """
        torch.save({
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'output_size': self.output_size,
            'activation': self.activation,
            'state_dict': self.state_dict()
        }, filepath)

    @classmethod
    def load_from_file(cls, filepath):
        """
        Load a model from a file and return it.
        """
        model_data = torch.load(filepath)
        model = cls(
            model_data['hidden_sizes'],
            input_size=model_data['input_size'],
            output_size=model_data['output_size'],
            activation=model_data['activation']
        )
        model.load_state_dict(model_data['state_dict'])
        return model

    def out_features(self, layer_id):
        """
        Helper function to get the output size of a given layer

        Args:
            layer_id: The index of the layer

        Returns:
            The output size of the layer
        """
        return self.layers[layer_id][0].out_features

    def forward(self, x, circuit=None, interventions=None, return_activations=False):
        """
        Applies a forward pass through the model, optionally using only a subset of the model (circuit) and applying
        interventions.

        Args:
            x: The input tensor
            circuit: The circuit to apply
            interventions: The list of interventions to apply
            return_activations: Whether to return the activations of each layer

        Returns:
            The output of the model if return_activations is False, else the activations of each layer
        """
        activations = [x.detach()] if return_activations else None

        # Prepare the list of interventions, ordering it by layer
        ordered_interventions = defaultdict(list)
        if interventions is not None:
            for (layer_idx, indices), values in interventions.items():
                ordered_interventions[layer_idx].append((indices, values))

        if 0 in ordered_interventions:
            for indices, values in ordered_interventions[0]:
                x[list(indices)] = values

        original_weights = []  # Store original weights if a circuit is applied

        for i, layer in enumerate(self.layers):
            if circuit:
                # Store original weights before applying circuit edge masks
                original_weights.append(layer[0].weight.data.clone())
                layer[0].weight.data *= torch.tensor(
                    circuit.edge_masks[i], dtype=torch.float32, device=self.device
                )
                if circuit.node_masks:
                    x *= torch.tensor(
                        circuit.node_masks[i], dtype=torch.float32, device=self.device
                    )

            # Forward pass
            x = layer(x)

            if return_activations:
                activations.append(x.detach())

            # Apply interventions at the current layer
            if interventions and i + 1 in ordered_interventions:
                for indices, values in ordered_interventions[i + 1]:
                    x[list(indices)] = values

        if circuit:
            # Restore original weights
            for i, layer in enumerate(self.layers):
                layer[0].weight.data = original_weights[i]

        return activations if return_activations else x

    def get_states(self, x, states: Dict[Any, Any]) -> Dict[Any, torch.Tensor]:
        """
        Runs an input through the model and returns selected hidden states.

        Args:
            x: The input tensor
            states: A dictionary of tuples (layer, indices) to return

        Returns:
            A dictionary mapping each tuple (layer, indices) to the corresponding hidden states
        """
        activations = self(x, return_activations=True)
        return {
            (layer, tuple(indices)): activations[layer][indices]
            for layer, indices in states.items()
        }

    def visualize(self, ax=None, circuit=None, activations=None):
        """
        Visualize the model's structure or activations

        Args:
            ax: The axis to plot on
            circuit: The optional circuit to visualize
            activations: The optional activations to visualize
        """

        G = nx.DiGraph()
        pos = {}
        colors = {}
        node_labels = {}
        edge_labels = {}
        use_activation_values = True

        # Set default values
        if circuit is None:
            from .circuit import Circuit

            circuit = Circuit.full(self.layer_sizes)
        node_masks = circuit.node_masks
        edge_masks = circuit.edge_masks

        if activations is None:
            activations = [torch.ones(1, size) for size in self.layer_sizes]
            use_activation_values = False

        # Build the graph
        max_width = max(self.layer_sizes)

        for layer_idx, layer_activations in enumerate(activations):
            num_nodes = layer_activations.shape[-1]
            y_start = -(max_width - num_nodes) / 2  # Center nodes vertically

            for node_idx in range(num_nodes):
                node_id = f"({layer_idx},{node_idx})"
                G.add_node(node_id)
                pos[node_id] = (layer_idx, y_start - node_idx)

                # Apply node mask for visualization
                if node_masks is not None:
                    active = node_masks[layer_idx][node_idx].item() == 1
                else:
                    active = True

                if use_activation_values:
                    activation_value = layer_activations[0, node_idx].item()
                    node_labels[node_id] = f"{activation_value:.2f}"
                    color_intensity = np.clip(activation_value, 0, 1)  # Normalize to [0, 1]
                    node_color = plt.cm.YlOrRd(color_intensity) if active else 'grey'  # Use heatmap for activations
                else:
                    node_color = 'tab:blue' if active else 'grey'
                colors[node_id] = node_color

        # Add edges to the graph and create edge labels
        for layer_idx, edge_mask in enumerate(edge_masks):
            for out_idx, row in enumerate(edge_mask):
                for in_idx, active in enumerate(row):
                    from_node_id = f"({layer_idx},{in_idx})"
                    to_node_id = f"({layer_idx + 1},{out_idx})"
                    G.add_edge(from_node_id, to_node_id, active=active.item())

                    weight = self.layers[layer_idx][0].weight[out_idx, in_idx].item()
                    edge_labels[(from_node_id, to_node_id)] = f"{weight:.2f}"

        # Draw nodes with color corresponding to activation value
        node_colors = [colors[node] for node in G.nodes]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1400, alpha=0.8)

        # Draw edges
        active_edges = [(u, v) for u, v, attr in G.edges(data=True) if attr['active'] == 1]
        inactive_edges = [(u, v) for u, v, attr in G.edges(data=True) if attr['active'] == 0]
        nx.draw_networkx_edges(G, pos, node_size=1400, edgelist=active_edges,
                               edge_color='tab:red' if use_activation_values else 'tab:blue', width=1, alpha=0.6,
                               arrows=True, arrowstyle="-|>", connectionstyle='arc3,rad=0.1', ax=ax)
        nx.draw_networkx_edges(G, pos, node_size=1400, edgelist=inactive_edges, edge_color='grey', width=1, alpha=0.5,
                               style='dashed', arrows=True, arrowstyle="-|>", connectionstyle='arc3,rad=0.1', ax=ax)

        # Draw node labels (activation values)
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=15, font_color='black', alpha=0.8, ax=ax)

        # Draw edge labels (weights)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, alpha=0.7, label_pos=0.6, ax=ax)

        if use_activation_values:
            plt.title("Neural Network Execution Visualization")
        else:
            plt.title("Neural Network Visualization")
        plt.axis('off')
        plt.show()

    def __getitem__(self, idx, in_place=False):
        """
        Overload the indexing operator to create a submodel

        Args:
            idx: The index or slice to use for the submodel
            in_place: Whether to modify the original model weights in-place

        Returns:
            A new submodel, with shared weights if in_place is True, or a copy otherwise
        """
        if isinstance(idx, slice):
            # Handle negative indices in the slice
            start_idx = idx.start if idx.start is not None else 0
            stop_idx = idx.stop if idx.stop is not None else len(self.layers)
            step_idx = idx.step if idx.step is not None else 1

            # Adjust negative indices
            if start_idx < 0:
                start_idx += len(self.layers)
            if stop_idx < 0:
                stop_idx += len(self.layers)

            # Extract the sliced layers
            sliced_layers = self.layers[start_idx:stop_idx:step_idx]
            num_sliced_layers = len(sliced_layers)

            # Calculate the input size for the submodel (first layer in the slice)
            input_size = self.layer_sizes[start_idx]

            # Calculate the hidden sizes and output size for the submodel
            smaller_hidden_sizes = [layer[0].out_features for layer in sliced_layers[:-1]]
            output_size = sliced_layers[-1][0].out_features

            # Create a new submodel with the correct architecture
            submodel = MLP(
                hidden_sizes=smaller_hidden_sizes,
                input_size=input_size,
                output_size=output_size,
                activation=self.activation,
                device=self.device,
            )

            # Copy weights and biases from the original model
            for i, layer in enumerate(sliced_layers):
                layer_data = submodel.layers[i][0]
                layer_data.weight.data = layer[0].weight.data
                layer_data.bias.data = layer[0].bias.data
                if not in_place:
                    layer_data.weight.data = layer_data.weight.data.clone()
                    layer_data.bias.data = layer_data.bias.data.clone()

            return submodel
        else:
            # Handle single index
            if in_place:
                return self.layers[idx]
            return copy.deepcopy(self.layers[idx])

    def do_train(self, x, y, x_val, y_val, batch_size, learning_rate, epochs, loss_target=0.001, val_frequency=10,
                 early_stopping_steps=3, logger=None):
        """
        Train the model using the given data and hyperparameters.

        Args:
            x: The training input tensor
            y: The training target tensor
            x_val: The validation input tensor
            y_val: The validation target tensorearly_stopping_steps
            batch_size: The batch size for training
            learning_rate: The learning rate for training
            epochs: The number of epochs to train
            loss_target: The target loss value to stop training
            val_frequency: The frequency of validation during training
            early_stopping_steps: The number of epochs without improvement to stop training
            logger: The logger to log training progress

        Returns:
            The average loss after training
        """
        # Create a DataLoader for the training dataset
        dataset = TensorDataset(x, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.parameters(), lr=learning_rate)

        best_loss = float('inf')
        bad_epochs = 0

        val_acc = 0

        # Training loop
        for epoch in range(epochs):
            self.train()
            epoch_loss = []
            for inputs, targets in dataloader:
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item())

            avg_loss = np.mean(epoch_loss)

            if avg_loss < best_loss:
                best_loss = avg_loss
                bad_epochs = 0
            else:
                bad_epochs += 1

            # Early stopping
            if avg_loss < loss_target or bad_epochs >= early_stopping_steps:
                break

            # Print training progress
            if (epoch + 1) % val_frequency == 0 and logger is not None:
                self.eval()
                with torch.no_grad():
                    train_outputs = self(x)
                    train_predictions = torch.round(train_outputs)
                    correct_predictions_train = train_predictions.eq(y).all(dim=1)
                    train_acc = correct_predictions_train.sum().item() / y.size(0)

                    val_outputs = self(x_val)
                    val_loss = criterion(val_outputs, y_val).item()
                    val_predictions = torch.round(val_outputs)
                    correct_predictions_val = val_predictions.eq(y_val).all(dim=1)
                    val_acc = correct_predictions_val.sum().item() / y_val.size(0)

                    logger.info(f'Epoch [{epoch + 1}/{epochs}], '
                                f'Train Loss: {avg_loss:.4f}, Train Accuracy: {train_acc:.4f}')
                    logger.info(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}, Bad Epochs: {bad_epochs}')

        return avg_loss

    def do_eval(self, x_test, y_test):
        """
        Performs evaluation on the given test data

        Args:
            x_test: The test input tensor
            y_test: The test target tensor

        Returns:
            The accuracy of the model on the test data
        """
        self.eval()
        with torch.no_grad():
            val_outputs = self(x_test)
            val_predictions = torch.round(val_outputs)
            correct_predictions_val = val_predictions.eq(y_test).all(dim=1)
            acc = correct_predictions_val.sum().item() / y_test.size(0)
        return acc

    def separate_into_k_mlps(self):
        """
        Separates the original MLP into K individual MLPs, each with output size 1.

        Returns:
            A list of K MLP models, each for a single output.
        """
        separate_models = [self[:] for _ in range(self.output_size)]

        last_layer = self.layers[-1]
        last_layer_weights = last_layer[0].weight.data
        last_layer_biases = last_layer[0].bias.data

        for i, model in enumerate(separate_models):
            model.layers[-1][0] = nn.Linear(self.hidden_sizes[-1], 1)
            model.layers[-1][0].weight.data = last_layer_weights[i:i + 1, :].clone()
            model.layers[-1][0].bias.data = last_layer_biases[i:i + 1].clone()
            model.output_size = 1
            model.layer_sizes = model.layer_sizes[:-1] + [1]

        return separate_models

    def enumerate_valid_node_masks(self):
        """
        Generate all valid node masks in the neural network.

        Returns:
            A list of all valid node masks
        """
        # Generate all valid masks for each layer
        all_masks_per_layer = []

        for size in self.layer_sizes[1:]:
            masks = [np.array([int(x) for x in format(i, f'0{size}b')]) for i in range(2 ** size)]
            all_masks_per_layer.append(masks)

        # Generate combinations of masks for all layers
        all_masks = list(itertools.product(*all_masks_per_layer))
        return all_masks
