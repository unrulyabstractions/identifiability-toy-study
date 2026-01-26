import copy
import itertools
from collections import defaultdict
from dataclasses import dataclass

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
from matplotlib.cm import ScalarMappable
from torch import nn
from tqdm import tqdm

from .causal import Intervention, PatchShape
from .grounding import Grounding, compute_local_tts, enumerate_tts
from .logic_gates import name_gate
from .utils import get_node_size


@dataclass
class CircuitStructure:
    node_sparsity: float
    edge_sparsity: float
    in_patches: list[PatchShape]
    out_patches: list[PatchShape]
    in_circuit: PatchShape
    out_circuit: PatchShape
    input_size: int
    output_size: int
    width: int
    depth: int

    def to_dict(self) -> dict:
        """Convert CircuitStructure to a serializable dictionary."""
        return {
            "node_sparsity": self.node_sparsity,
            "edge_sparsity": self.edge_sparsity,
            "in_patches": [p.to_dict() for p in self.in_patches],
            "out_patches": [p.to_dict() for p in self.out_patches],
            "in_circuit": self.in_circuit.to_dict() if self.in_circuit else None,
            "out_circuit": self.out_circuit.to_dict() if self.out_circuit else None,
            "input_size": self.input_size,
            "output_size": self.output_size,
            "width": self.width,
            "depth": self.depth,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CircuitStructure":
        """Create from dict."""
        return cls(
            node_sparsity=data["node_sparsity"],
            edge_sparsity=data["edge_sparsity"],
            in_patches=[PatchShape.from_dict(p) for p in data["in_patches"]],
            out_patches=[PatchShape.from_dict(p) for p in data["out_patches"]],
            in_circuit=PatchShape.from_dict(data["in_circuit"])
            if data["in_circuit"]
            else None,
            out_circuit=PatchShape.from_dict(data["out_circuit"])
            if data["out_circuit"]
            else None,
            input_size=data["input_size"],
            output_size=data["output_size"],
            width=data["width"],
            depth=data["depth"],
        )


class Circuit:
    """
    A class representing a circuit of a neural network.

    Attributes:
        node_masks (list of np.ndarray): A list of node masks for each layer.
        edge_masks (list of np.ndarray): A list of edge masks for each layer.
    """

    def __init__(self, node_masks, edge_masks):
        """
        Initializes a Circuit object with node_masks and edge_masks.

        Args:
            node_masks: The node masks for each layer
            edge_masks: The edge masks for each layer
        """
        self.node_masks = node_masks
        self.edge_masks = edge_masks

    def to_dict(self):
        """Convert Circuit to a serializable dictionary."""
        return {
            "node_masks": [mask.tolist() for mask in self.node_masks],
            "edge_masks": [mask.tolist() for mask in self.edge_masks],
        }

    @classmethod
    def from_dict(cls, data):
        """Create a Circuit from a dictionary."""
        node_masks = [np.array(mask) for mask in data["node_masks"]]
        edge_masks = [np.array(mask) for mask in data["edge_masks"]]
        return cls(node_masks, edge_masks)

    def __repr__(self):
        return f"Circuit(node_masks={self.node_masks}, edge_masks={self.edge_masks})"

    def to_intervention(self, device: str = "cpu") -> Intervention:
        """
        Convert Circuit to an Intervention that zeroes masked nodes/edges:
          - Node masks: neuron 'mul' at h^{(L)} with a 0/1 vector (broadcast across batch).
          - Edge masks: edge  'mul' at W^{(L)} with a 0/1 matrix.
        """
        patches = {}

        # Node masks (apply at current activation h^{(L)})
        if getattr(self, "node_masks", None) is not None:
            for L, nm in enumerate(self.node_masks):
                k = len(nm)
                idxs = tuple(range(k))  # all neurons in that activation
                vals = torch.as_tensor(nm, dtype=torch.float32, device=device).view(
                    1, k
                )
                patches[PatchShape(layers=(L,), indices=idxs, axis="neuron")] = (
                    "mul",
                    vals,
                )

        # Edge masks (apply to W^{(L)})
        if getattr(self, "edge_masks", None) is not None:
            for L, em in enumerate(self.edge_masks):
                vals = torch.as_tensor(em, dtype=torch.float32, device=device)
                patches[PatchShape(layers=(L,), indices=(), axis="edge")] = (
                    "mul",
                    vals,
                )

        return Intervention(patches=patches)

    def get_all_possible_intervention_patches(
        self, max_patch_size: int = -1
    ) -> tuple[list[PatchShape], list[PatchShape], PatchShape, PatchShape]:
        """
        Get all possible positions(PatchShape) for interventions.
        Patches returned should modify at most max_patch_size neurons. (-1 return all possible combinations of patches)
        Returns:
            in_patches: Patches that are interventions within circuit (active neurons)
            out_patches: Patches that are interventions outside circuit (inactive neurons)
            in_circuit: Single PatchShape covering all in-circuit hidden neurons
            out_circuit: Single PatchShape covering all out-circuit hidden neurons
        """
        in_patches = []
        out_patches = []

        # Collect all in-circuit and out-circuit neuron indices per layer
        # Skip layer 0 (input) and last layer (output)
        all_in_indices = []
        all_out_indices = []
        hidden_layers = []

        for layer_idx in range(1, len(self.node_masks) - 1):
            node_mask = self.node_masks[layer_idx]
            in_indices = tuple(i for i, active in enumerate(node_mask) if active == 1)
            out_indices = tuple(i for i, active in enumerate(node_mask) if active == 0)

            hidden_layers.append(layer_idx)
            all_in_indices.append(in_indices)
            all_out_indices.append(out_indices)

            # Create per-neuron patches for in-circuit
            for idx in in_indices:
                in_patches.append(
                    PatchShape(layers=(layer_idx,), indices=(idx,), axis="neuron")
                )

            # Create per-neuron patches for out-circuit
            for idx in out_indices:
                out_patches.append(
                    PatchShape(layers=(layer_idx,), indices=(idx,), axis="neuron")
                )

        # If max_patch_size is specified, filter or generate combinations
        if max_patch_size > 0:
            # Filter to patches of at most max_patch_size
            in_patches = [p for p in in_patches if len(p.indices) <= max_patch_size]
            out_patches = [p for p in out_patches if len(p.indices) <= max_patch_size]

        # Create single PatchShape covering all in-circuit neurons per layer
        # We create one per layer since indices differ per layer
        in_circuit_patches = []
        out_circuit_patches = []
        for layer_idx, in_idx, out_idx in zip(
            hidden_layers, all_in_indices, all_out_indices
        ):
            if in_idx:
                in_circuit_patches.append(
                    PatchShape(layers=(layer_idx,), indices=in_idx, axis="neuron")
                )
            if out_idx:
                out_circuit_patches.append(
                    PatchShape(layers=(layer_idx,), indices=out_idx, axis="neuron")
                )

        # in_circuit and out_circuit as combined PatchShape (using first layer as representative)
        # or None if empty
        in_circuit = in_circuit_patches[0] if in_circuit_patches else None
        out_circuit = out_circuit_patches[0] if out_circuit_patches else None

        return in_patches, out_patches, in_circuit, out_circuit

    def analyze_structure(self) -> CircuitStructure:
        node_sparsity, edge_sparsity, _ = self.sparsity()

        in_patches, out_patches, in_circuit, out_circuit = (
            self.get_all_possible_intervention_patches()
        )

        return CircuitStructure(
            node_sparsity=node_sparsity,
            edge_sparsity=edge_sparsity,
            in_patches=in_patches,
            out_patches=out_patches,
            in_circuit=in_circuit,
            out_circuit=out_circuit,
            input_size=len(self.node_masks[0]),
            output_size=len(self.node_masks[-1]),
            width=len(self.node_masks[1]),
            depth=len(self.node_masks[1:-1]),
        )

    @staticmethod
    def full(layer_sizes):
        """
        Returns a full circuit of the given layer sizes.

        Args:
            layer_sizes: The sizes of each layer

        Returns:
            A full circuit of the given layer sizes
        """
        node_masks = [np.ones(x) for x in layer_sizes]
        edge_masks = [
            np.ones((y, x)) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])
        ]
        return Circuit(node_masks, edge_masks)

    def save_to_file(self, filepath):
        """
        Saves the node_masks and edge_masks to a file in .npz format.

        Args:
            filepath (str): The file path where to save the circuit data.
        """
        # Convert the list of numpy arrays to a dictionary for saving
        data_to_save = {
            f"node_mask_{i}": mask for i, mask in enumerate(self.node_masks)
        }
        data_to_save.update(
            {f"edge_mask_{i}": mask for i, mask in enumerate(self.edge_masks)}
        )

        # Save as .npz file
        np.savez(filepath, **data_to_save)

    @classmethod
    def load_from_file(cls, filepath):
        """
        Loads node_masks and edge_masks from a file and returns an instance of Circuit.

        Args:
            filepath (str): The file path from where to load the circuit data.

        Returns:
            Circuit: A new instance of Circuit with the loaded masks.
        """
        # Load the data from the .npz file
        loaded_data = np.load(filepath)

        # Extract node_masks and edge_masks from the loaded data
        node_masks = [
            loaded_data[f"node_mask_{i}"]
            for i in range(len(loaded_data.files) // 2 + 1)
        ]
        edge_masks = [
            loaded_data[f"edge_mask_{i}"] for i in range(len(loaded_data.files) // 2)
        ]

        # Create a new Circuit instance and return it
        return cls(node_masks, edge_masks)

    def validate_against_model(self, model):
        """
        Checks whether the node_masks and edge_masks in the Circuit are valid for the given model.

        Args:
            model: The model to validate against
        """
        if len(self.node_masks) != model.num_layers + 1:
            raise ValueError(
                "The number of node masks in the circuit does not match the number of layers "
                "plus the input layer in the model."
            )

        if len(self.edge_masks) != model.num_layers:
            raise ValueError(
                "The number of edge masks in the circuit does not match the number of layers in the model."
            )

        for i, layer in enumerate(model.layers):
            linear_layer = layer[0]

            node_mask_size = self.node_masks[i + 1].shape[0]
            if node_mask_size != linear_layer.out_features:
                raise ValueError(
                    f"Node mask at layer {i + 1} does not match the size of the model's layer."
                )

            edge_mask_size = self.edge_masks[i].shape
            weight_size = linear_layer.weight.shape

            if edge_mask_size != weight_size:
                raise ValueError(
                    f"Edge mask at layer {i} does not match the size of the model's weight matrix."
                )

            for out_idx in range(weight_size[0]):  # Rows (output neurons of this layer)
                for in_idx in range(
                    weight_size[1]
                ):  # Columns (input neurons of this layer)
                    if self.edge_masks[i][out_idx, in_idx] == 1:
                        if (
                            self.node_masks[i][in_idx] == 0
                            or self.node_masks[i + 1][out_idx] == 0
                        ):
                            raise ValueError(
                                f"Active edge from node {in_idx} to node {out_idx} "
                                f"at layer {i} connects to an inactive node."
                            )

            self._validate_node_connectivity(i)

        # Check for an active path from inputs to outputs
        active_path_exists = self._check_active_path()
        if not active_path_exists:
            raise ValueError(
                "No active path exists from the input to the output in the circuit."
            )

    def _validate_node_connectivity(self, layer_idx):
        """
        Ensure each active node has at least one incoming edge (except if it is an input)
        and one outgoing edge (except if it is an output).

        Args:
            layer_idx: The index of the layer to validate
        """
        if layer_idx > 0:
            # Check incoming edges for nodes in layer_idx
            incoming_edges = self.edge_masks[layer_idx - 1]
            for node_idx in range(incoming_edges.shape[0]):
                if self.node_masks[layer_idx][node_idx] == 1:  # Active node
                    if not any(incoming_edges[node_idx, :] == 1):
                        raise ValueError(
                            f"Active node {node_idx} in layer {layer_idx} has no incoming edges."
                        )

        if layer_idx < len(self.edge_masks):
            # Check outgoing edges for nodes in layer_idx
            outgoing_edges = self.edge_masks[layer_idx]
            for node_idx in range(outgoing_edges.shape[1]):
                if self.node_masks[layer_idx][node_idx] == 1:  # Active node
                    if not any(outgoing_edges[:, node_idx] == 1):
                        raise ValueError(
                            f"Active node {node_idx} in layer {layer_idx} has no outgoing edges."
                        )

    def _check_active_path(self):
        """
        Checks if there's an active path from input to output.
        """
        # Start with active nodes in the input layer
        active_nodes = set(
            i for i, active in enumerate(self.node_masks[0]) if active == 1
        )

        for i in range(len(self.edge_masks)):
            next_active_nodes = set()
            for j in range(self.edge_masks[i].shape[0]):
                if any(
                    self.edge_masks[i][j, k] == 1 and k in active_nodes
                    for k in range(self.edge_masks[i].shape[1])
                ):
                    next_active_nodes.add(j)
            active_nodes = next_active_nodes

        # Check if any output node is active
        return any(self.node_masks[-1][i] == 1 for i in active_nodes)

    def __le__(self, other):
        """
        Checks if the current circuit is included in another circuit.

        Args:
            other (Circuit): The circuit to check inclusion against.

        Returns:
            bool: True if the current circuit is included in the other, False otherwise.
        """
        if len(self.node_masks) != len(other.node_masks) or len(self.edge_masks) != len(
            other.edge_masks
        ):
            return False

        for i, (node_mask, other_node_mask) in enumerate(
            zip(self.node_masks, other.node_masks)
        ):
            # if a node is included in self but not in other, return False
            if np.any(np.logical_and(node_mask == 1, other_node_mask == 0)):
                return False

        for i, (edge_mask, other_edge_mask) in enumerate(
            zip(self.edge_masks, other.edge_masks)
        ):
            # if an edge is included in self but not in other, return False
            if np.any(np.logical_and(edge_mask == 1, other_edge_mask == 0)):
                return False

        return True

    def sparsity(self):
        """
        Computes the overall sparsity of nodes and edges across all layers.

        Returns:
            tuple: (overall_node_sparsity, overall_edge_sparsity)
                   where each is the fraction of excluded nodes/edges across all layers.
        """
        # Combine all node masks into a single array
        combined_node_mask = np.concatenate(self.node_masks[1:-1], axis=None)

        # Combine all edge masks into a single array
        combined_edge_mask = np.concatenate(self.edge_masks, axis=None)

        # Calculate overall sparsity as the fraction of zero values
        overall_node_sparsity = np.mean(combined_node_mask == 0)
        overall_edge_sparsity = np.mean(combined_edge_mask == 0)

        # Combine node and edge masks into a single array for overall combined sparsity
        combined_mask = np.concatenate([combined_node_mask, combined_edge_mask], axis=0)
        overall_combined_sparsity = np.mean(combined_mask == 0)

        return overall_node_sparsity, overall_edge_sparsity, overall_combined_sparsity

    def overlap_jaccard(self, other):
        """
        Computes the overlap Jaccard index with another circuit.

        Args:
            other: The other circuit

        Returns:
            The overlap Jaccard index
        """
        sk1_extended_mask = list(self.node_masks[1:-1]) + list(self.edge_masks)
        sk2_extended_mask = list(other.node_masks[1:-1]) + list(other.edge_masks)
        return compute_jaccard_index(sk1_extended_mask, sk2_extended_mask)

    def ground(self, activations):
        """
        Grounds a circuit in model activations and returns all valid groundings

        Args:
            activations: The model's activations

        Returns:
            A list of all valid groundings
        """
        activations = [x.cpu().numpy() for x in activations]
        activations[-1] = torch.round(torch.tensor(activations[-1])).numpy().astype(int)

        global_inputs_pt = activations[0].astype(int)
        global_inputs = list(map(tuple, global_inputs_pt))

        global_tts = defaultdict(list)
        for layer_id, node_mask in enumerate(self.node_masks[1:], start=1):
            for node_id in range(len(node_mask)):
                if node_mask[node_id] == 0:
                    continue

                neuron_values = activations[layer_id][:, node_id]
                mapping = dict(zip(global_inputs, neuron_values))

                neuron_info = tuple([layer_id, node_id])
                for tt in enumerate_tts(mapping):
                    tt["neuron_info"] = neuron_info
                    global_tts[neuron_info].append(tt)

        # Create initial groundings
        first_neurons = [(key, val) for key, val in global_tts.items() if key[0] == 1]
        initial_groundings = list(itertools.product(*[a[1] for a in first_neurons]))
        initial_groundings = [list(i_g) for i_g in initial_groundings]
        for i_g in initial_groundings:
            for tt_i in i_g:
                tt_i["local_tt"] = tt_i["tt"]
                tt_i["gate_name"] = name_gate(tt_i["local_tt"])
                tt_i["par_n"] = "inputs"

        partial_groundings = initial_groundings
        for neuron_info, neuron_val in global_tts.items():
            current_layer, node_id = neuron_info
            if current_layer <= 1:
                continue

            edge_mask = self.edge_masks[current_layer - 1]
            prev_nodes = np.where(self.node_masks[current_layer - 1])[0]

            # Parents
            par_n = [
                (current_layer - 1, par)
                for par in prev_nodes
                if edge_mask[node_id, par]
            ]

            # Iterate over all partially constructed groundings so far
            new_partial_groundings = []
            for p_g in partial_groundings:
                parents_tts = [val["tt"] for val in p_g if val["neuron_info"] in par_n]

                for possible_current_tt in neuron_val:
                    node_tt = possible_current_tt["tt"]

                    local_tts = compute_local_tts(node_tt, parents_tts, global_inputs)

                    if local_tts is None:
                        continue

                    for l_tt in local_tts:
                        new_pg = copy.deepcopy(p_g)
                        new_neuron_val = copy.deepcopy(possible_current_tt)
                        new_neuron_val["local_tt"] = l_tt
                        new_neuron_val["gate_name"] = name_gate(l_tt)
                        new_neuron_val["par_n"] = par_n
                        new_pg.append(new_neuron_val)
                        new_partial_groundings.append(new_pg)

            partial_groundings = new_partial_groundings

        return [Grounding(grounding, self) for grounding in partial_groundings]

    def visualize(
        self,
        ax=None,
        display_idx=False,
        node_size="small",
        file_path=None,
        labels=None,
        colors=None,
    ):
        node_size = get_node_size(node_size)
        """
        Visualize the circuit.
        
        Args:
            ax: The axis to plot on
            display_idx: Whether to display the index of each node
            node_size: The size of the nodes
            file_path: The path to save the figure to
            labels: The labels for each node
            colors: The colors for each node
        """

        G = nx.DiGraph()
        pos = {}  # Dictionary to store positions for nodes

        # Calculate max width (number of nodes) for proper alignment
        max_width = max(len(mask) for mask in self.node_masks)

        # Create nodes and edges
        for layer_idx, node_mask in enumerate(self.node_masks):
            num_nodes = len(node_mask)
            y_start = -(max_width - num_nodes) / 2  # Centering the layer vertically

            for node_idx, active in enumerate(node_mask):
                node_name = f"({layer_idx},{node_idx})"
                G.add_node(node_name, layer=layer_idx, active=active)
                pos[node_name] = (layer_idx, y_start - node_idx)  # Centered vertically

        for layer_idx, edge_mask in enumerate(self.edge_masks):
            for out_idx, row in enumerate(edge_mask):
                for in_idx, active in enumerate(row):
                    G.add_edge(
                        f"({layer_idx},{in_idx})",
                        f"({layer_idx + 1},{out_idx})",
                        active=active,
                    )

        # Draw nodes
        active_nodes = [
            node for node, attr in G.nodes(data=True) if attr["active"] == 1
        ]
        inactive_nodes = [
            node for node, attr in G.nodes(data=True) if attr["active"] == 0
        ]
        node_color = "tab:blue" if colors is None else colors
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=active_nodes,
            node_color=node_color,
            node_size=node_size,
            alpha=0.8,
            ax=ax,
        )
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=inactive_nodes,
            node_color="grey",
            node_size=node_size,
            alpha=0.4,
            ax=ax,
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
            node_size=node_size,
            edgelist=active_edges,
            edge_color="tab:blue",
            width=2,
            alpha=0.8,
            arrows=True,
            arrowstyle="-|>",
            connectionstyle="arc3,rad=0.1",
            ax=ax,
        )
        nx.draw_networkx_edges(
            G,
            pos,
            node_size=node_size,
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

        # Draw labels
        if display_idx:
            nx.draw_networkx_labels(
                G,
                pos,
                labels=labels,
                font_size=15,
                font_color="black",
                alpha=0.9,
                ax=ax,
            )

        if ax is None:
            if file_path is None:
                plt.axis("off")
                plt.show()
            else:
                plt.tight_layout()
                plt.savefig(file_path)
                plt.close()
        else:
            ax.axis("off")


def visualize_circuit_heatmap(circuits, ax=None, sizes="small", file_path=None):
    """
    Visualize a list of circuits as a heatmap.

    Args:
        circuits: The list of circuits
        ax: The axis to plot on
        sizes: The size of the nodes
        file_path: The path to save the figure to
    """
    node_size = get_node_size(sizes)

    # Assuming all circuits have the same structure (same number of nodes and edges)
    G = nx.DiGraph()
    pos = {}
    node_activation_count = {}
    edge_activation_count = {}

    # Calculate max width for proper alignment
    max_width = max(len(mask) for mask in circuits[0].node_masks)

    # Create nodes and edges, and count activations
    for layer_idx, node_mask in enumerate(circuits[0].node_masks):
        num_nodes = len(node_mask)
        y_start = -(max_width - num_nodes) / 2  # Centering the layer vertically

        for node_idx in range(num_nodes):
            node_name = f"({layer_idx},{node_idx})"
            G.add_node(node_name, layer=layer_idx)
            pos[node_name] = (layer_idx, y_start - node_idx)

            # Initialize node activation count
            node_activation_count[node_name] = 0

            # Count how often the node is active across all circuits
            for circuit in circuits:
                if circuit.node_masks[layer_idx][node_idx] == 1:
                    node_activation_count[node_name] += 1

    # Create edges and count activations
    for layer_idx, edge_mask in enumerate(circuits[0].edge_masks):
        for out_idx, row in enumerate(edge_mask):
            for in_idx in range(len(row)):
                edge_name = (f"({layer_idx},{in_idx})", f"({layer_idx + 1},{out_idx})")
                G.add_edge(*edge_name)

                # Initialize edge activation count
                edge_activation_count[edge_name] = 0

                # Count how often the edge is active across all circuits
                for circuit in circuits:
                    if circuit.edge_masks[layer_idx][out_idx][in_idx] == 1:
                        edge_activation_count[edge_name] += 1

    # Normalize activation counts to [0, 1] for coloring
    max_node_activation = len(circuits)
    max_edge_activation = len(circuits)

    cmap = plt.cm.inferno

    node_colors = [
        cmap(node_activation_count[node] / max_node_activation) for node in G.nodes()
    ]
    edge_colors = [
        cmap(edge_activation_count[edge] / max_edge_activation) for edge in G.edges()
    ]

    # Draw nodes with heatmap colors
    nx.draw_networkx_nodes(
        G, pos, node_color=node_colors, node_size=node_size, alpha=0.9, ax=ax
    )

    # Draw edges with heatmap colors
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=G.edges(),
        edge_color=edge_colors,
        width=2,
        alpha=0.9,
        arrows=True,
        arrowstyle="-|>",
        connectionstyle="arc3,rad=0.1",
        ax=ax,
    )

    # Draw labels for node indices (optional, but can be useful)
    # nx.draw_networkx_labels(G, pos, font_size=12, font_color='black', alpha=0.9, ax=ax)

    # Add colorbar for the frequency (activation percentage)
    sm = ScalarMappable(cmap=cmap)
    sm.set_array([])  # Required for the colorbar
    cbar = plt.colorbar(sm, ax=ax if ax else plt.gca(), fraction=0.03, pad=0.04)
    cbar.set_label("Activation Frequency", rotation=270, labelpad=15)

    # Show or save the figure
    if file_path:
        plt.savefig(file_path, bbox_inches="tight")
    else:
        if ax is None:
            plt.axis("off")
            plt.show()
        else:
            ax.axis("off")


def compute_jaccard_index(mask1, mask2):
    """
    Computes the Jaccard index between two node masks.

    Args:
        mask1: The first node mask
        mask2: The second node mask

    Returns:
        The Jaccard index
    """
    # Flatten masks and compute the intersection and union
    mask1_flat = np.concatenate(mask1, axis=None).flatten()
    mask2_flat = np.concatenate(mask2, axis=None).flatten()

    intersection = np.sum((mask1_flat == 1) & (mask2_flat == 1))
    union = np.sum((mask1_flat == 1) | (mask2_flat == 1))

    if union == 0:
        return 0.0
    return intersection / union


def _enumerate_edge_mask_per_layer(in_mask, out_mask):
    """
    Generates all valid edge masks for a given input and output node mask.

    Args:
        in_mask: The input node mask
        out_mask: The output node mask

    Returns:
        All valid edge masks
    """
    num_out_nodes = len(out_mask)
    num_in_nodes = len(in_mask)

    edge_mask = np.zeros((num_out_nodes, num_in_nodes), dtype=int)

    # Set entire rows to zero for inactive output nodes
    active_out_nodes = np.where(out_mask == 1)[0]
    # Set entire columns to zero for inactive input nodes
    active_in_nodes = np.where(in_mask == 1)[0]

    edge_mask[np.ix_(active_out_nodes, active_in_nodes)] = 1

    all_masks = []
    # Generate all possible combinations for the remaining active edges
    for mask in itertools.product(
        *[range(2)] * len(active_out_nodes) * len(active_in_nodes)
    ):
        mask_array = np.array(mask).reshape(len(active_out_nodes), len(active_in_nodes))

        # Check that each row and column has at least one `1`
        if not np.all(mask_array.sum(axis=1) > 0):
            continue
        if not np.all(mask_array.sum(axis=0) > 0):
            continue

        edge_mask_tmp = copy.copy(edge_mask)

        # Set masked entries
        edge_mask_tmp[np.ix_(active_out_nodes, active_in_nodes)] = mask_array

        all_masks.append(edge_mask_tmp)
    return all_masks


def enumerate_all_valid_circuit(
    model, min_sparsity: float = 0.0, use_tqdm=True
) -> list[Circuit]:
    """
    Enumerate valid node patterns as circuits with full edge connectivity.

    Formula: (2^w - 1)^d valid patterns for d hidden layers of width w.
    Each pattern has at least 1 active node per hidden layer.

    Args:
        model: The input model
        min_sparsity: Minimum node sparsity (fraction of hidden nodes OFF).
            Default 0.0 includes all valid patterns.
        use_tqdm: Whether to use tqdm to show progress

    Returns:
        List of circuits, one per valid node pattern, with full edges.
    """
    from .subcircuit import enumerate_node_patterns, full_edge_config

    layer_widths = model.layer_sizes

    all_circuits = []
    patterns = enumerate_node_patterns(layer_widths, min_sparsity)

    if use_tqdm:
        from .subcircuit import count_node_patterns

        total = count_node_patterns(layer_widths)
        patterns = tqdm(patterns, total=total, desc="Enumerating node patterns")

    for pattern in patterns:
        node_arrays = pattern.to_arrays()
        edge_masks = full_edge_config(pattern)
        circuit = Circuit(node_masks=node_arrays, edge_masks=edge_masks)
        all_circuits.append(circuit)

    return all_circuits


def enumerate_edge_variants(circuit: Circuit) -> list[Circuit]:
    """
    For a given node pattern, enumerate all valid edge configurations.

    Use after identifying best node patterns via subcircuit metrics.

    Args:
        circuit: A circuit with a specific node pattern

    Returns:
        List of circuits with all valid edge configurations.
    """
    from .subcircuit import NodePattern, enumerate_edge_configs

    # Convert circuit to NodePattern
    layer_masks = tuple(
        sum(int(b) << i for i, b in enumerate(mask)) for mask in circuit.node_masks
    )
    layer_widths = tuple(len(mask) for mask in circuit.node_masks)
    pattern = NodePattern(layer_masks=layer_masks, layer_widths=layer_widths)

    circuits = []
    for edge_masks in enumerate_edge_configs(pattern):
        new_circuit = Circuit(node_masks=circuit.node_masks, edge_masks=edge_masks)
        circuits.append(new_circuit)

    return circuits


def analyze_circuits(circuits, top_n=None):
    """
    Analyzes a list of circuits and returns the fraction of included pairs, average Jaccard similarity, and top pairs.

    Args:
        circuits: The list of circuits
        top_n: The top N pairs to return

    Returns:
        The fraction of included pairs, average Jaccard similarity, and top pairs
    """
    n = len(circuits)
    included_count = 0
    total_count = 0
    jaccard_similarities = []

    pair_indices = list(itertools.combinations(range(n), 2))
    pair_similarities = []

    for i, j in tqdm(
        pair_indices, total=len(pair_indices), desc="Comparing circuits pairs"
    ):
        if i == j:
            continue

        if circuits[i] <= circuits[j] or circuits[j] <= circuits[i]:
            included_count += 1

        total_count += 1

        # Calculate Jaccard similarity
        jaccard_sim = circuits[i].overlap_jaccard(circuits[j])
        jaccard_similarities.append(jaccard_sim)
        pair_similarities.append(((i, j), jaccard_sim))

    fraction_included_pairs = included_count / total_count if total_count > 0 else 0
    average_jaccard_similarity = (
        np.mean(jaccard_similarities) if jaccard_similarities else 0
    )

    # Sort pairs by Jaccard similarity
    top_n_pairs = sorted(pair_similarities, key=lambda x: x[1])
    # Get the top_n with lowest similarity
    if top_n:
        top_n_pairs = top_n_pairs[:top_n]

    return fraction_included_pairs, average_jaccard_similarity, top_n_pairs


def find_circuits(
    model: nn.Module, x, y, accuracy_threshold, min_sparsity=None, use_tqdm=True
):
    """
    Find all valid circuits in a model.

    Args:
        model: The input model
        x: The input data to use for validation
        y: The target data to use for validation
        accuracy_threshold: The minimum accuracy threshold for circuits
        min_sparsity: The minimum sparsity threshold for circuits
        use_tqdm: Whether to use tqdm to show progress

    Returns:
        All valid circuits found in the model.
    """
    # Make predictions with the model
    model_predictions = model(x)
    bit_model_pred = torch.round(model_predictions)

    all_sks = enumerate_all_valid_circuit(
        model, min_sparsity=min_sparsity, use_tqdm=use_tqdm
    )

    # Initialize a list to collect data for DataFrame
    data = []

    max_sparsity, max_sparsity_node, max_sparsity_edge = 0, 0, 0
    top_sks = []
    sparsities = []

    # Iterate over all circuits with progress tracking
    it = enumerate(all_sks)
    if use_tqdm:
        it = tqdm(it, total=len(all_sks), desc="Evaluating circuits")
    for i, circuit in it:
        # Make predictions with the current circuit
        sk_predictions = model(x, intervention=circuit.to_intervention(model.device))
        bit_sk_pred = torch.round(sk_predictions)

        # Compute the accuracy with respect to the task
        correct_predictions = bit_sk_pred.eq(y).all(dim=1)
        accuracy = correct_predictions.sum().item() / y.size(0)

        # Compute similarity with model prediction based on logits
        logit_similarity = 1 - nn.MSELoss()(model_predictions, sk_predictions).item()

        # Compute similarity with model prediction
        same_predictions = torch.sum(bit_model_pred == bit_sk_pred).item()
        total_predictions = bit_model_pred.shape[0]
        similarity_bit_preds = same_predictions / total_predictions

        # Compute circuit sparsity
        sk_sparsity_node_sparsity, sk_edge_sparsity, sk_combined_sparsity = (
            circuit.sparsity()
        )
        if sk_sparsity_node_sparsity > max_sparsity_node:
            max_sparsity_node = sk_sparsity_node_sparsity
        if sk_edge_sparsity > max_sparsity_edge:
            max_sparsity_edge = sk_edge_sparsity
        if sk_combined_sparsity > max_sparsity:
            max_sparsity = sk_combined_sparsity

        if accuracy > accuracy_threshold:
            top_sks.append(circuit)
            sparsities.append(sk_sparsity_node_sparsity)

        # Collect the data
        data.append(
            {
                "circuit_idx": i,
                "accuracy": accuracy,
                "logit_similarity": logit_similarity,
                "similarity_bit_preds": similarity_bit_preds,
                "sk_sparsity": sk_combined_sparsity,
                "sk_sparsity_node_sparsity": sk_sparsity_node_sparsity,
                "sk_edge_sparsity": sk_edge_sparsity,
            }
        )

    return top_sks, sparsities, pd.DataFrame(data)
