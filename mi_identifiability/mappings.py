import itertools
import string
from collections import defaultdict
from itertools import chain, combinations
from typing import Dict, List

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgba, ListedColormap
from tqdm import tqdm

from .logic_gates import LogicTree


class Mapping(dict):
    """
    A mapping from high-level nodes (part of a formula) to sets of low-level neurons
    """
    _cmap = None

    def __init__(self, base_dict: Dict[str, Dict[int, List[int]]], formula: LogicTree, layer_sizes):
        """
        Initialize the Mapping.

        Args:
            base_dict: The dictionary of high-level nodes to low-level nodes (node -> (layer -> [nodes]))
            formula: The formula the mapping applies to
            layer_sizes: The sizes of the layers in the neural network (used for visualization)
        """
        from .circuit import Circuit

        super().__init__(base_dict)
        self.circuit = Circuit.full(layer_sizes)
        self.layer_sizes = layer_sizes
        self.formula = formula

    def __lt__(self, other):
        """
        Checks whether this mapping is strictly contained in another

        Args:
            other: The other mapping

        Returns:
            True if the first mapping is contained within the second, False otherwise
        """
        for key_ in self:
            if not smaller_(self[key_], other.get(key_, {})):
                return False
        return True

    @classmethod
    def colormap(cls):
        """
        Sets the colormap to use for visualizing mappings

        Returns:
            A matplotlib colormap
        """
        if cls._cmap is None:
            base_cmap = plt.get_cmap('tab20')
            colors = base_cmap(np.linspace(0, 1, base_cmap.N))
            new_color = to_rgba('grey')
            colors = np.vstack([colors, new_color])
            cls._cmap = ListedColormap(colors)
        return cls._cmap

    def visualize(self, ax=None):
        """
        Visualize the mapping

        Args:
            ax: The axis to plot on
        """

        # Assign colors based on the high-level node
        n_nodes = sum(self.layer_sizes)
        n_previous_nodes = list(itertools.accumulate([0] + self.layer_sizes[:-1]))
        colors = [20] * n_nodes
        for hl_node, ll_mapping in self.items():
            for layer, nodes in ll_mapping.items():
                for node in nodes:
                    colors[n_previous_nodes[layer] + node] = int(hl_node.split(' ')[-1]) % 20
        colors[n_previous_nodes[-1]] = self.formula.id

        # Assign labels based on the high-level node
        labels = defaultdict(str)
        for i in range(self.layer_sizes[0]):
            labels[f'(0,{i})'] = string.ascii_uppercase[i]
        cm = Mapping.colormap()

        self.circuit.visualize(ax=ax, colors=[cm(i) for i in colors], display_idx=True, labels=labels)


def smaller_(map1, map2):
    """
    Checks whether a given high-level node's mapping is contained in another
    Args:
        map1: The first mapping for the node
        map2: The second mapping for the node

    Returns:
        True if map1 is contained in map2, False otherwise
    """
    return (map1 == map2 or set((k, v2) for k, v in map1.items() for v2 in v)
            <= set((k, v2) for k, v in map2.items() for v2 in v))


def all_subsets_reversed(ss):
    """
    Produces all possible subsets of a set, in reverse order

    Args:
        ss: The input set

    Returns:
        All possible subsets of the set, in reverse order
    """
    return chain(*map(lambda x: combinations(ss, x), range(len(ss), -1, -1)))


def find_mappings(model, formula: LogicTree, nodes, explanations, mappings, sample_pairs, pos, intervened_formulas,
                  use_tqdm=False, classif=False, distrib_layers=False, output_pos=None):
    """
    Recursively finds all possible what-then-where explanations for the given model.

    Args:
        model: The input model to explain
        formula: The input formula to find explanations for
        nodes: The list of high-level nodes to find explanations for
        explanations: The list of explanations found so far
        mappings: The list of low-level nodes that haven't been used yet
        sample_pairs: The list of all pairs of samples, with their input, intermediate and output values
        pos: The depth of the current intervention
        intervened_formulas: The list of formulas obtained when all possible interventions are applied to the base tree
        use_tqdm: Whether to use tqdm to show progress
        classif: Whether the model is a classification model (single output neuron)
        distrib_layers: Whether to allow variable representations that are distributed across layers
        output_pos: The position of the output node to consider (defaults to 0)

    Returns:
        A generator of all possible explanations
    """

    # No node left -> we finished
    if not nodes:
        yield explanations
        return

    if output_pos is None:
        output_pos = 0
    elif classif:
        raise NotImplementedError("Multiple outputs not implemented for classification models")

    # Find the first unexplained high-level node in the formula
    cur_node, *nodes = nodes
    tree_node = formula.find_node(cur_node)
    node_name = f"{tree_node} {tree_node.id}"  # Adding the id to guarantee unique node names

    # Remove input and output low-level nodes from possible mappings
    saved_map = []
    last_l = len(model.layer_sizes) - 1
    if not tree_node.children:
        saved_map = [(0, ord(tree_node.op[-1]) - ord('A'))]
    elif cur_node == formula.id:
        saved_map = [(last_l, 0), (last_l, 1)] if classif else [(last_l, output_pos)]
    mappings2 = mappings
    mappings2 -= {(0, x) for x in range(model.input_size)}
    mappings2 -= {(last_l, x) for x in range(model.output_size)}

    # Get each possible partition of the mapping subset
    subsets = all_subsets_reversed(mappings2)
    if use_tqdm:
        subsets = tqdm(subsets, total=2 ** len(mappings2), position=pos, leave=False, desc=node_name)
    for mapping_subset in subsets:
        with torch.no_grad():
            # Add the input/output_nodes if necessary
            mapping_subset = list(mapping_subset) + saved_map

            # Edge case
            if not mapping_subset:
                continue

            # Ensure that all layers are contiguous
            layers = set(x[0] for x in mapping_subset)
            if set(range(min(layers), max(layers) + 1)) != layers:
                continue
            if not distrib_layers and len(layers) > 1:
                continue

            # Turn the mapping into a dictionary
            subset_dict = defaultdict(list)
            for key, val in mapping_subset:
                subset_dict[key].append(val)

            # Check whether the mapping is valid for all samples
            ok = True
            for (source, s_in), (base, b_in) in sample_pairs:
                o_h = intervened_formulas[(cur_node, source[cur_node])].call(base)

                a_l = model.get_states(s_in, subset_dict)
                o_l = model(b_in, interventions=a_l)

                if classif and o_h != np.argmax(o_l.cpu().detach()) or not classif and o_h != int(
                        o_l[output_pos] > 0.5):
                    ok = False
                    break

            if ok:
                # Valid explanation found, we keep going with the rest
                yield from find_mappings(model, formula, nodes, explanations + [(node_name, dict(subset_dict))],
                                         mappings2 - set(mapping_subset), sample_pairs, pos + 1, intervened_formulas,
                                         use_tqdm, classif, distrib_layers, output_pos)


def find_minimal_mappings(model, formula: LogicTree, sample_pairs, intervened_trees, **kwargs):
    """
    Finds all minimal (w.r.t. inclusion) what-then-where explanations for a given model and formula.

    Args:
        model: The input model to explain
        formula: The input formula to find explanations for
        sample_pairs: The list of all pairs of samples, with their input, intermediate and output values
        intervened_trees: The list of formulas obtained when all possible interventions are applied to the base tree
        **kwargs: Additional arguments to pass to the explain function

    Returns:
        A list of minimal what-then-where explanations (Mapping objects)
    """
    all_mappings = set(reversed(list([(i + 1, y) for i, x in enumerate(model.layer_sizes[1:-1]) for y in range(x)])))
    relevant_nodes = set()
    for node in formula.get_node_ids():
        found_node = formula.find_node(node)
        if found_node.children and formula.id != node:
            relevant_nodes.add(node)

    relevant_nodes -= {node.children[0].id for node in formula.get_nodes() if len(node.children) == 1}

    minimal_mappings = []
    for explanation in find_mappings(model, formula, relevant_nodes, [], all_mappings,
                                     sample_pairs, 0, intervened_trees, **kwargs):

        mapping = Mapping(dict(explanation), formula, model.layer_sizes)

        if any(mapping < map2 for map2 in minimal_mappings):
            minimal_mappings = [map2 for map2 in minimal_mappings if not mapping < map2]
            minimal_mappings.append(mapping)
        elif not any(map2 < mapping for map2 in minimal_mappings):
            minimal_mappings.append(mapping)
    return minimal_mappings
