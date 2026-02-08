import copy
import itertools
import os
from collections import defaultdict

from src.domain import name_gate


class Grounding(dict):
    """
    A circuit grounding assigning a logic gate and separation boundary to each node in the circuit
    """
    def __init__(self, assignments, circuit):
        """
        Initialize the grounding

        Args:
            assignments: The logic gate and separation boundary assignments for each circuit node
            circuit: The input circuit
        """
        super().__init__({str(x['neuron_info']).replace(' ', ''): x['gate_name'] for x in assignments})
        self.circuit = circuit

    def visualize(self, **kwargs):
        """
        Visualize the grounding

        Args:
            **kwargs: Additional keyword arguments to pass to the circuit visualization
        """
        self.circuit.visualize(display_idx=True, labels=self, **kwargs)


def enumerate_tts(mapping):
    """
    Enumerate all possible truth tables for a given mapping

    Args:
        mapping: The input mapping

    Returns:
        All possible truth tables
    """
    all_tts = []
    values = sorted(mapping.values())
    for b_x, b_y in zip(values, values[1:]):
        if b_x < b_y:
            separator = (b_y + b_x) / 2
            new_tt = {k: int(v > separator) for k, v in mapping.items()}
            if len(set(new_tt.values())) == 1:
                raise Exception("Not allowed")
            all_tts.append({'separator': separator, 'tt': new_tt})
    return all_tts


def compute_local_tts(node_tt, parents_tts, global_inputs):
    """
    Computes local truth tables for a given node based on its parents

    Args:
        node_tt: The node's truth table
        parents_tts: The truth tables of the node's parents
        global_inputs: The global inputs of the model

    Returns:
        The computed local truth tables
    """
    num_parents = len(parents_tts)
    local_inputs = list(itertools.product([0, 1], repeat=num_parents))

    # Build inverse mapping from parent tuples to global inputs
    inv_parents = defaultdict(list)
    for g_inp in global_inputs:
        parent_tuple = tuple(par_tt[g_inp] for par_tt in parents_tts)
        inv_parents[parent_tuple].append(g_inp)

    local_tt = {}
    missing_entries = []
    for inp in local_inputs:
        global_anchors = inv_parents[inp]

        if len(global_anchors) > 1:
            unique_values = {node_tt[ga] for ga in global_anchors}
            if len(unique_values) > 1:
                return None  # Inconsistent, cannot compute
            local_tt[inp] = unique_values.pop()
        elif len(global_anchors) == 1:
            local_tt[inp] = node_tt[global_anchors[0]]
        else:
            missing_entries.append(inp)

    # Generate all possible expansions for missing entries
    if missing_entries:
        combinations = itertools.product([0, 1], repeat=len(missing_entries))
        local_tts = []
        for bits in combinations:
            new_tt = local_tt.copy()
            for m_e, bit in zip(missing_entries, bits):
                new_tt[m_e] = bit
            local_tts.append(new_tt)
    else:
        local_tts = [local_tt.copy()]

    # Filter out XOR and XNOR gates
    filtered_tt = [l_tt for l_tt in local_tts
                   if name_gate(l_tt) not in ("XOR", "XNOR")]
    return filtered_tt


def load_circuits(log_dir):
    """
    Loads all circuits from a directory

    Args:
        log_dir: The directory containing the circuits

    Returns:
        The loaded circuits and their ids
    """
    from .circuit import Circuit

    circuit_folder = os.path.join(log_dir, "circuits")
    assert os.path.exists(circuit_folder), f"No circuit folder in {log_dir}"

    circuits = []
    ids = []
    for sk_name in os.listdir(circuit_folder):
        if not sk_name.endswith(".npz"):
            continue
        idx = int(sk_name.split('.')[0].split("-")[1])
        sk_path = os.path.join(circuit_folder, sk_name)
        circuits.append(Circuit.load_from_file(sk_path))
        ids.append(idx)

    return circuits, ids


def ground_subcircuit(circuit, activations):
    """
    Ground a subcircuit in model activations.

    This is an alias for circuit.ground(activations) for easier import.

    Args:
        circuit: The circuit to ground
        activations: The model's activations

    Returns:
        A list of all valid groundings
    """
    return circuit.ground(activations)
