# neural_model.py
import copy
import itertools
from typing import TYPE_CHECKING, Any, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from .causal import Intervention, PatchShape

if TYPE_CHECKING:
    from .circuit import Circuit

ACTIVATION_FUNCTIONS = {
    "leaky_relu": nn.LeakyReLU,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}


class DecomposedMLP(nn.Module):
    """
    Wrapper holding the result of SPD decomposition on an MLP.
    Stores the component model and provides methods to inspect/visualize components.
    """

    # Default ComponentModel config (used if not provided)
    DEFAULT_CM_CONFIG = {
        "ci_fn_type": "vector_mlp",
        "ci_fn_hidden_dims": [8],
        "sigmoid_type": "leaky_hard",
        "pretrained_model_output_attr": None,
    }

    def __init__(
        self,
        component_model=None,
        target_model: "MLP" = None,
        cm_config: dict = None,
    ):
        super().__init__()
        self.component_model = component_model  # SPD ComponentModel after optimization
        self.target_model = target_model  # Original MLP that was decomposed
        self.cm_config = cm_config or self.DEFAULT_CM_CONFIG.copy()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the decomposed model."""
        assert self.component_model is not None, "No component model available"
        return self.component_model(x)

    def get_n_components(self) -> int:
        """Return number of components per layer."""
        assert self.component_model is not None
        first_component = next(iter(self.component_model.components.values()))
        return first_component.C

    def save(self, path: str):
        """Save decomposed model to file."""
        assert self.component_model is not None
        assert self.target_model is not None

        # Extract module_info for reconstruction
        module_info = [
            {"module_pattern": name, "C": comp.C}
            for name, comp in self.component_model.components.items()
        ]

        torch.save(
            {
                "component_model_state": self.component_model.state_dict(),
                "target_model_state": self.target_model.state_dict(),
                "module_info": module_info,
                "component_model_config": self.cm_config,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, target_model: "MLP", device: str = "cpu"):
        """Load decomposed model from file.

        Args:
            path: Path to saved decomposed model
            target_model: Pre-loaded target model (required)
            device: Device to load model to
        """
        from spd.configs import ModulePatternInfoConfig
        from spd.models.component_model import ComponentModel
        from spd.run_spd import expand_module_patterns

        data = torch.load(path, map_location=device, weights_only=False)

        # Load target model state
        target_model.load_state_dict(data["target_model_state"])
        target_model.to(device)
        target_model.requires_grad_(False)

        # Reconstruct component model
        module_info = [
            ModulePatternInfoConfig(module_pattern=m["module_pattern"], C=m["C"])
            for m in data["module_info"]
        ]
        module_path_info = expand_module_patterns(target_model, module_info)

        cm_config = data["component_model_config"]
        component_model = ComponentModel(
            target_model=target_model,
            module_path_info=module_path_info,
            ci_fn_type=cm_config["ci_fn_type"],
            ci_fn_hidden_dims=cm_config["ci_fn_hidden_dims"],
            sigmoid_type=cm_config["sigmoid_type"],
            pretrained_model_output_attr=cm_config["pretrained_model_output_attr"],
        )
        component_model.load_state_dict(data["component_model_state"])
        component_model.to(device)

        return cls(
            component_model=component_model,
            target_model=target_model,
            cm_config=cm_config,
        )


class MLP(nn.Module):
    """
    A class implementing a simple multi-layer perceptron model.
    """

    def __init__(
        self,
        hidden_sizes: list,
        input_size: int = 2,
        output_size: int = 1,
        activation: str = "relu",
        device: str = "cpu",
        debug: bool = False,
        logger=None,
    ):
        """
        Initialize the MLP model.

        Args:
            hidden_sizes: A list of sizes of the hidden layers (in neurons), not including the input and output layers
            input_size: The size of the input layer
            output_size: The size of the output layer
            activation: The activation function to use
            device: The device to run the model on
            debug: Whether to enable device consistency checking
        """
        super().__init__()
        self.input_size = input_size
        self._hidden_sizes = hidden_sizes  # underscore to avoid conflict with method
        self.output_size = output_size
        self.layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.activation = activation

        # Activation on all layers EXCEPT the last linear (final head outputs logits)
        self.layers = nn.ModuleList(
            nn.Sequential(
                nn.Linear(in_size, out_size),
                ACTIVATION_FUNCTIONS[activation]()
                if idx < (len(self.layer_sizes) - 1)
                else nn.Identity(),
            )
            for idx, (in_size, out_size) in enumerate(
                zip(self.layer_sizes, self.layer_sizes[1:])
            )
        )

        self.num_layers = len(self.layers)
        self.device = device
        self.debug = debug
        self.to(device)

    def save_to_file(self, filepath: str):
        """Save the model's configuration and state_dict to a file."""
        torch.save(
            {
                "hidden_sizes": self._hidden_sizes,
                "input_size": self.input_size,
                "output_size": self.output_size,
                "activation": self.activation,
                "state_dict": self.state_dict(),
            },
            filepath,
        )

    @classmethod
    def load_from_file(cls, filepath, device: str = "cpu"):
        """Load a model from a file and return it."""
        model_data = torch.load(filepath, map_location=device)
        model = cls(
            model_data["hidden_sizes"],
            input_size=model_data["input_size"],
            output_size=model_data["output_size"],
            activation=model_data["activation"],
            device=device,
        )
        model.load_state_dict(model_data["state_dict"])
        return model

    def out_features(self, layer_id: int) -> int:
        """
        Helper function to get the output size of a given layer

        Args:
            layer_id: The index of the layer

        Returns:
            The output size of the layer
        """
        return self.layers[layer_id][0].out_features

    def forward(
        self,
        x: torch.Tensor,
        intervention: Optional[Intervention] = None,
        return_activations: bool = False,
    ):
        activations = [x.detach()] if return_activations else None

        # Fast path: no interventions - use standard module calls (enables forward hooks)
        if intervention is None and not return_activations:
            for layer in self.layers:
                x = layer(x)
            return x

        # Group new Intervention by layer/axis
        neuron_by_layer = (
            intervention.group_neuron_by_layer() if intervention is not None else {}
        )
        edge_by_layer = (
            intervention.group_edge_by_layer() if intervention is not None else {}
        )

        # Apply neuron patches that target current activation h^{(0)} (the input)
        if 0 in neuron_by_layer:
            x = self._apply_neuron_patches_inplace(x, neuron_by_layer[0])

        for i, layer in enumerate(self.layers):
            lin, act = layer[0], layer[1]

            # Effective weights: start from raw, then apply edge patches for this layer i
            Weff = lin.weight
            if i in edge_by_layer:
                Weff = self._apply_edge_patches(Weff, edge_by_layer[i])

            # Also apply any neuron patches that target the current activation h^{(i)}
            if i in neuron_by_layer:
                x = self._apply_neuron_patches_inplace(x, neuron_by_layer[i])

            # Linear + activation
            z = F.linear(x, Weff, lin.bias)
            x = act(z) if i < (self.num_layers - 1) else z

            if return_activations:
                activations.append(x.detach())

            # Apply new Intervention neuron patches at h^{(i+1)}
            if (i + 1) in neuron_by_layer:
                x = self._apply_neuron_patches_inplace(x, neuron_by_layer[i + 1])

        if return_activations:
            return activations
        return x

    def get_states(self, x, states: dict[Any, Any]) -> dict[Any, torch.Tensor]:
        """
        Runs an input through the model and returns selected hidden states.

        Args:
            x: The input tensor
            states: A dictionary whose keys are (layer, indices), values are index lists

        Returns:
            A dictionary mapping (layer, indices) to the corresponding hidden states
        """
        activations = self(x, return_activations=True)
        return {
            (layer, tuple(indices)): activations[layer][indices]
            for layer, indices in states.items()
        }

    def get_patch(self, x: torch.Tensor, shape: PatchShape) -> Intervention:
        """
        τ_U(x) as an Intervention.
        - neuron axis: returns h^{(L)}[:, J] (per L in shape.layers), optionally under `circuit`.
        - edge axis: data-agnostic (use Circuit.to_intervention or make a manual Intervention).
        """
        if shape.axis == "edge":
            raise NotImplementedError(
                "Use Circuit.to_intervention(...) or supply explicit edge values."
            )
        acts = self(x, return_activations=True)
        patches = {}
        for L in shape.single_layers():
            vals = acts[L][:, list(shape.indices)]  # shape [B, k]
            patches[PatchShape(layers=(L,), indices=shape.indices, axis="neuron")] = (
                "set",
                vals,
            )
        return Intervention(patches)

    @property
    def hidden_sizes(self) -> list[int]:
        """Returns the sizes of post-activation hidden layers (1..L-1)."""
        return [self.layers[i][0].out_features for i in range(self.num_layers - 1)]

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
            from .circuit import Circuit  # late import to avoid circular deps

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
                    color_intensity = float(
                        np.clip(activation_value, 0, 1)
                    )  # Normalize to [0, 1]
                    node_color = plt.cm.YlOrRd(color_intensity) if active else "grey"
                else:
                    node_color = "tab:blue" if active else "grey"
                colors[node_id] = node_color

        # Add edges and labels
        for layer_idx, edge_mask in enumerate(edge_masks):
            for out_idx, row in enumerate(edge_mask):
                for in_idx, active in enumerate(row):
                    from_node_id = f"({layer_idx},{in_idx})"
                    to_node_id = f"({layer_idx + 1},{out_idx})"
                    G.add_edge(from_node_id, to_node_id, active=active.item())

                    weight = self.layers[layer_idx][0].weight[out_idx, in_idx].item()
                    edge_labels[(from_node_id, to_node_id)] = f"{weight:.2f}"

        # Draw nodes
        node_colors = [colors[node] for node in G.nodes]
        nx.draw_networkx_nodes(
            G, pos, node_color=node_colors, node_size=1400, alpha=0.8, ax=ax
        )

        # Draw edges
        active_edges = [
            (u, v) for u, v, attr in G.edges(data=True) if attr["active"] == 1
        ]
        inactive_edges = [
            (u, v) for u, v, attr in G.edges(data=True) if attr["active"] == 0
        ]
        nx.draw_networkx_edges(
            G,
            pos,
            node_size=1400,
            edgelist=active_edges,
            edge_color="tab:red" if use_activation_values else "tab:blue",
            width=1,
            alpha=0.6,
            arrows=True,
            arrowstyle="-|>",
            connectionstyle="arc3,rad=0.1",
            ax=ax,
        )
        nx.draw_networkx_edges(
            G,
            pos,
            node_size=1400,
            edgelist=inactive_edges,
            edge_color="grey",
            width=1,
            alpha=0.5,
            style="dashed",
            arrows=True,
            arrowstyle="-|>",
            connectionstyle="arc3,rad=0.1",
            ax=ax,
        )

        # Labels
        nx.draw_networkx_labels(
            G,
            pos,
            labels=node_labels,
            font_size=15,
            font_color="black",
            alpha=0.8,
            ax=ax,
        )
        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels=edge_labels,
            font_size=7,
            alpha=0.7,
            label_pos=0.6,
            ax=ax,
        )

        plt.title(
            "Neural Network Execution Visualization"
            if use_activation_values
            else "Neural Network Visualization"
        )
        plt.axis("off")
        plt.show()

    def _slice(self, idx, in_place: bool):
        # ... your existing logic here ...
        """
        Overload the indexing operator to create a submodel

        Args:
            idx: The index or slice to use for the submodel
            in_place: Whether to share parameters (True) or copy (False)

        Returns:
            A new submodel when slicing. When idx is an int, returns the (copied) layer unless in_place=True.
        """
        if isinstance(idx, slice):
            # Handle negative indices in the slice
            start_idx = idx.start if idx.start is not None else 0
            stop_idx = idx.stop if idx.stop is not None else len(self.layers)
            step_idx = idx.step if idx.step is not None else 1

            if start_idx < 0:
                start_idx += len(self.layers)
            if stop_idx < 0:
                stop_idx += len(self.layers)

            # Extract the sliced layers
            sliced_layers = self.layers[start_idx:stop_idx:step_idx]

            # Sizes for the new submodel
            input_size = self.layer_sizes[start_idx]
            smaller_hidden_sizes = [
                layer[0].out_features for layer in sliced_layers[:-1]
            ]
            output_size = sliced_layers[-1][0].out_features

            # Create the submodel
            submodel = MLP(
                hidden_sizes=smaller_hidden_sizes,
                input_size=input_size,
                output_size=output_size,
                activation=self.activation,
                device=self.device,
                debug=self.debug,
            )

            # Copy or tie parameters
            if in_place:
                # Share Parameters (weight tying)
                for i, layer in enumerate(sliced_layers):
                    src = layer[0]  # original nn.Linear
                    dst = submodel.layers[i][0]  # submodel nn.Linear
                    dst.weight = src.weight
                    dst.bias = src.bias
            else:
                with torch.no_grad():
                    for i, layer in enumerate(sliced_layers):
                        src = layer[0]
                        dst = submodel.layers[i][0]
                        dst.weight.copy_(src.weight.detach())
                        dst.bias.copy_(src.bias.detach())

            return submodel
        else:
            # Single index: return the layer
            if in_place:
                return self.layers[idx]
            return copy.deepcopy(self.layers[idx])

    class _View:
        def __init__(self, model, in_place):
            self._model = model
            self._in_place = in_place

        def __getitem__(self, idx):
            return self._model._slice(idx, in_place=self._in_place)

        def __call__(self, idx=slice(None)):
            return self._model._slice(idx, in_place=self._in_place)

    @property
    def view(self):
        return MLP._View(self, True)

    @property
    def clone(self):
        return MLP._View(self, False)

    def __getitem__(self, idx):
        """Slicing always copies (safe default)."""
        return self.clone(idx)

    def do_train(
        self,
        x,
        y,
        x_val,
        y_val,
        batch_size,
        learning_rate,
        epochs,
        loss_target: float = 0.001,
        val_frequency: int = 1,
        early_stopping_steps: int = 10,
        l1_lambda: float = 1e-5,
        logger=None,
    ):
        """
        Train the model using the given data and hyperparameters.
        """

        # DataLoader
        dataset = TensorDataset(x, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Loss and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=0.0)

        best_loss = float("inf")
        bad_epochs = 0
        val_acc = 0.0

        # Training loop
        for epoch in range(epochs):
            self.train()
            epoch_loss = []
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                optimizer.zero_grad()
                outputs = self(inputs)

                loss = criterion(outputs, targets)

                # Promote sparsity
                if l1_lambda > 0:
                    l1_loss = sum(p.abs().sum() for p in self.parameters())
                    loss = loss + l1_lambda * l1_loss

                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item())

            avg_loss = float(np.mean(epoch_loss))

            if avg_loss < best_loss:
                best_loss = avg_loss
                bad_epochs = 0
            else:
                bad_epochs += 1

            # Progress / validation
            if (epoch + 1) % val_frequency == 0 and logger is not None:
                self.eval()
                with torch.no_grad():
                    train_logits = self(x.to(self.device))
                    train_predictions = (train_logits > 0).float()
                    correct_predictions_train = train_predictions.eq(
                        y.to(self.device)
                    ).all(dim=1)
                    train_acc = correct_predictions_train.sum().item() / y.size(0)

                    val_logits = self(x_val.to(self.device))
                    val_loss = criterion(val_logits, y_val.to(self.device)).item()
                    val_predictions = (val_logits > 0).float()
                    correct_predictions_val = val_predictions.eq(
                        y_val.to(self.device)
                    ).all(dim=1)
                    val_acc = correct_predictions_val.sum().item() / y_val.size(0)

                    logger.info(
                        f"Epoch [{epoch + 1}/{epochs}], "
                        f"Train Loss: {avg_loss:.4f}, Train Accuracy: {train_acc:.4f}"
                    )
                    logger.info(
                        f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}, Bad Epochs: {bad_epochs}"
                    )

            # Early stopping
            if avg_loss < loss_target or bad_epochs >= early_stopping_steps:
                break

        return avg_loss

    def separate_into_k_mlps(self):
        """
        Separates the original MLP into K individual MLPs, each with output size 1.

        Returns:
            A list of K MLP models, each for a single output.
        """
        # Original final linear
        last_linear = self.layers[-1][0]
        in_features = last_linear.in_features
        W, b = last_linear.weight, last_linear.bias

        # Preserve device/dtype in the new heads
        p = next(self.parameters())
        device, dtype = p.device, p.dtype

        separate_models = []
        for i in range(self.output_size):
            # Create a fresh model with output_size=1
            model = MLP(
                hidden_sizes=self.hidden_sizes,
                input_size=self.input_size,
                output_size=1,
                activation=self.activation,
                device=device,
                debug=self.debug,
            )

            # Copy weights from original model (all layers except last)
            with torch.no_grad():
                for layer_idx in range(len(self.layers) - 1):
                    src = self.layers[layer_idx][0]
                    dst = model.layers[layer_idx][0]
                    dst.weight.copy_(src.weight.detach())
                    dst.bias.copy_(src.bias.detach())

                # Copy the i-th output's weights for the last layer
                dst = model.layers[-1][0]
                dst.weight.copy_(W[i : i + 1].detach())  # [1, H]
                dst.bias.copy_(b[i : i + 1].detach())  # [1]

            separate_models.append(model)

        return separate_models

    def separate_subcircuit(self, circuit: "Circuit", gate_idx: int = 0) -> "MLP":
        """
        Create a new MLP with the circuit's masked edges zeroed out.

        Args:
            circuit: Circuit object with edge_masks defining which connections to keep
            gate_idx: Which output gate this subcircuit is for (used to slice last edge_mask)

        Returns:
            New MLP with masked weights set to zero
        """
        # Deep clone the model to avoid modifying original weights
        submodel = self.clone[:]

        # Apply edge masks: zero out weights where mask is 0
        with torch.no_grad():
            for layer_idx, edge_mask in enumerate(circuit.edge_masks):
                linear = submodel.layers[layer_idx][0]
                mask_arr = edge_mask

                # If this is the last layer and the mask has more outputs than the model,
                # slice the mask to match the model's output size
                if layer_idx == len(circuit.edge_masks) - 1:
                    expected_out = linear.weight.shape[0]
                    if mask_arr.shape[0] > expected_out:
                        # Slice to get this gate's mask
                        mask_arr = mask_arr[gate_idx : gate_idx + expected_out, :]

                mask_tensor = torch.tensor(
                    mask_arr, dtype=linear.weight.dtype, device=linear.weight.device
                )
                linear.weight.mul_(mask_tensor)

        return submodel

    def enumerate_valid_node_masks(self, fix_output: bool = True):
        """
        Generate all valid node masks for hidden layers.

        Formula: For d hidden layers each of width w, total masks = 2^(w*d).
        Each hidden node independently can be 0 or 1.

        Args:
            fix_output: If True, output layer is always all-ones.

        Returns:
            A list of all node masks (cartesian product across hidden layers)
        """
        all_masks_per_layer = []

        layer_sizes = self.layer_sizes[1:]  # Skip input layer
        for i, layer_size in enumerate(layer_sizes):
            is_output_layer = i == len(layer_sizes) - 1
            if fix_output and is_output_layer:
                # Output layer: only all-ones mask
                all_masks_per_layer.append([np.ones(layer_size, dtype=int)])
            else:
                # Hidden layers: all 2^w combinations
                layer_masks = [
                    np.array(mask, dtype=int)
                    for mask in itertools.product([0, 1], repeat=layer_size)
                ]
                all_masks_per_layer.append(layer_masks)

        # Combine masks across all layers: product of 2^w for each hidden layer = 2^(w*d)
        return list(itertools.product(*all_masks_per_layer))

    def _apply_neuron_patches_inplace(self, h, patches):  # inside class MLP
        """
        patches: list of (mode, indices, values) for the current activation h (shape [B, n]).
        Broadcast rules: values ∈ {[], [k], [1,k], [B,1], [B,k]} -> expand to [B,k].
        """
        if not patches:
            return h
        B = h.shape[0]
        dev, dt = h.device, h.dtype
        for mode, idxs, vals in patches:
            k = len(idxs)
            vv = vals.to(device=dev, dtype=dt)
            if vv.ndim == 0:
                vv = vv.view(1, 1).repeat(B, k)
            elif vv.ndim == 1:
                if vv.shape[0] == k:
                    vv = vv.view(1, k).repeat(B, 1)
                elif vv.shape[0] == B:
                    vv = vv.view(B, 1).repeat(1, k)
                else:
                    raise ValueError(
                        f"Cannot broadcast {tuple(vv.shape)} to [B={B},k={k}]"
                    )
            elif vv.ndim == 2:
                if vv.shape == (1, k):
                    vv = vv.repeat(B, 1)
                elif vv.shape == (B, 1):
                    vv = vv.repeat(1, k)
                elif vv.shape != (B, k):
                    raise ValueError(f"Expected [B,k], got {tuple(vv.shape)}")
            else:
                raise ValueError(f"Unsupported ndim {vv.ndim} for neuron patch")
            cols = list(idxs)
            if mode == "set":
                h[:, cols] = vv
            elif mode == "mul":
                h[:, cols] = h[:, cols] * vv
            elif mode == "add":
                h[:, cols] = h[:, cols] + vv
            else:
                raise ValueError(f"Unknown mode {mode}")
        return h

    def _apply_edge_patches(self, W, patches):
        """
        patches: list of (mode, values) to apply to W (shape [out,in]).
        Values must be broadcastable to W.
        """
        if not patches:
            return W
        Weff = W
        for mode, V in patches:
            Vv = V.to(device=W.device, dtype=W.dtype)
            if mode == "set":
                Weff = Vv
            elif mode == "mul":
                Weff = Weff * Vv
            elif mode == "add":
                Weff = Weff + Vv
            else:
                raise ValueError(f"Unknown mode {mode}")
        return Weff
