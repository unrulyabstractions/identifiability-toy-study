import copy
import itertools
import string
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from itertools import product

import networkx as nx
import numpy as np
import torch
from matplotlib import pyplot as plt

from .utils import get_node_size, powerset


class LogicGate:
    """
    Class representing a logic gate.

    The class attribute `inputs` stores all valid input combinations for a given number of inputs
    """

    inputs = {}

    def __init__(self, n_inputs, gate_fn, name):
        """
        Initialize the logic gate.

        Args:
            n_inputs: The number of inputs of the logic gate
            gate_fn: The function representing the logic gate
            name: The name of the logic gate
        """
        self.n_inputs = n_inputs
        self.name = name
        if isinstance(gate_fn, str):
            self.gate_fn = globals()[gate_fn]
        else:
            self.gate_fn = gate_fn

        if self.n_inputs not in LogicGate.inputs:
            LogicGate.inputs[self.n_inputs] = np.array(
                list(product([0, 1], repeat=self.n_inputs)), dtype=np.float32
            )

    def __repr__(self):
        return f"LogicGate(n_inputs={self.n_inputs}, name={self.name})"

    def truth_table(self):
        """
        Computes the truth table of the gate

        Returns:
            The gate's truth table
        """
        outputs = self.gate_fn(LogicGate.inputs[self.n_inputs])
        return dict(zip(map(tuple, LogicGate.inputs[self.n_inputs]), outputs))

    def generate_noisy_data(
        self, n_repeats=200, noise_std=0.1, weights=None, device="cpu"
    ):
        """
        Generates noisy input and output data for the gate

        Args:
            n_repeats: The number of times to repeat the inputs
            noise_std: The standard deviation of the Gaussian noise
            weights: The weights used to optionally skew the input distribution
            device: The device to store the data on

        Returns:
            The noisy input and output data
        """
        inputs = self.get_inputs(self.n_inputs, n_repeats, weights)
        outputs = np.expand_dims(self.gate_fn(inputs), 1)

        out_x, out_y = self.add_noise_and_repeat(n_repeats, noise_std, inputs, outputs)
        return out_x.to(device), out_y.to(device)

    @staticmethod
    def get_inputs(n_inputs, n_repeats, weights):
        """
        Returns all possible input combinations for a given number of inputs

        Args:
            n_inputs: The number of inputs
            n_repeats: The number of times to repeat the inputs
            weights: The weights used to optionally skew the input distribution

        Returns:
            All possible input combinations
        """
        if weights is None:
            weights = np.ones(2**n_inputs, dtype=np.float32)

        weights /= np.sum(weights)

        x = np.random.choice(
            len(LogicGate.inputs[n_inputs]),
            size=n_repeats * len(LogicGate.inputs[n_inputs]),
            p=weights,
        )
        return LogicGate.inputs[n_inputs][x]

    @staticmethod
    def add_noise_and_repeat(n_repeats, noise_std, inputs, outputs):
        """
        Adds noise to the inputs and repeats them.

        Args:
            n_repeats: The number of times to repeat the inputs
            noise_std: The standard deviation of the Gaussian noise
            inputs: The inputs
            outputs: The outputs

        Returns:
            The noisy inputs and outputs
        """
        if noise_std == 0:
            return torch.tensor(inputs, dtype=torch.float32), torch.tensor(
                outputs, dtype=torch.float32
            )

        # Generate noisy data by repeating the inputs and adding Gaussian noise
        x_noisy = np.repeat(inputs, n_repeats, axis=0)
        y_noisy = np.repeat(outputs, n_repeats, axis=0)

        # Add Gaussian noise
        noise = np.random.normal(0, noise_std, x_noisy.shape).astype(np.float32)
        x_noisy += noise

        return torch.tensor(x_noisy, dtype=torch.float32), torch.tensor(
            y_noisy, dtype=torch.float32
        )

    def print_truth_table(self, logger):
        """
        Prints the truth table of the gate

        Args:
            logger: The logger to use
        """
        tt = self.truth_table()

        # Log header
        header = " | ".join([f"x{i}" for i in range(self.n_inputs)]) + " | Output"
        logger.info(header)
        logger.info("-" * len(header))

        # Log each row of the truth table
        for input_row, output in tt.items():
            input_str = " | ".join(map(str, map(int, input_row)))
            logger.info(f"{input_str} |   {int(output)}")


def generate_noisy_multi_gate_data(
    logic_gates: list[LogicGate],
    n_repeats=200,
    noise_std=0.1,
    weights=None,
    device="cpu",
):
    """
    Generates noisy data for K logic gates. Each gate is expected to take the same number of inputs.

    Args:
        logic_gates: List of logic gate functions, each accepting n_inputs and returning an output.
        n_repeats: Number of times to repeat the inputs for noise generation.
        noise_std: Standard deviation of the Gaussian noise added to the inputs.
        weights: Weights for sampling inputs.
        device: The storage device for output tensors.

    Returns:
        Noisy inputs and corresponding outputs for the K logic gates.
    """
    if len({lg.n_inputs for lg in logic_gates}) != 1:
        raise ValueError("All logic gates must have the same number of inputs.")
    n_inputs = logic_gates[0].n_inputs

    # Generate all possible input combinations for n_inputs binary inputs
    inputs = LogicGate.get_inputs(n_inputs, n_repeats, weights)

    # Generate outputs for each logic gate and stack them horizontally (axis=1)
    outputs = np.hstack([gate.gate_fn(inputs).reshape(-1, 1) for gate in logic_gates])

    out_x, out_y = LogicGate.add_noise_and_repeat(n_repeats, noise_std, inputs, outputs)
    return out_x.to(device), out_y.to(device)


class LogicTree:
    """
    A tree representing a logic expression (combination of logic gates and variables)

    Attributes:
        op: The operator (if not a leaf) or variable
        children: The children nodes (if not a leaf)
        id: The id of the node
    """

    def __init__(self, op, children=None, id_=None):
        """
        Initializes the tree.

        Args:
            op: The operator
            children: The list of children (operands)
            id_: The id of the node
        """
        self.op = op
        self.children = []
        self.id = id_
        if isinstance(children, Iterable):
            for x in children:
                if isinstance(x, LogicTree):
                    self.children.append(x)
                elif isinstance(x, Iterable):
                    self.children.append(LogicTree(*x))
                else:
                    self.children.append(LogicTree(x))
        elif id_ is None:
            self.id = children

    @staticmethod
    def from_logic(eq_):
        """
        Creates a tree from a logic expression

        Params:
            eq_: The logic expression (with strict parentheses)

        Returns:
            A new tree corresponding to the expression
        """
        if eq_ in string.ascii_uppercase:
            leaf = LogicTree(eq_, [])
            return leaf
        if eq_[0] == "¬":
            child = LogicTree.from_logic(eq_[2:-1])
            node_ = LogicTree("¬", [child])
            return node_
        elif eq_[0] == "(":
            idx = 1
            depth_ = 0
            while True:
                if eq_[idx] == "(":
                    depth_ += 1
                elif eq_[idx] == ")":
                    depth_ -= 1
                # OR, AND, IMP, EQV, XOR
                elif eq_[idx] in "+*>=x" and not depth_:
                    left = LogicTree.from_logic(eq_[1:idx])
                    right = LogicTree.from_logic(eq_[idx + 1 : -1])
                    node_ = LogicTree(eq_[idx], [left, right])
                    return node_
                idx += 1

    def __repr__(self, quotes=False):
        """
        Returns a string representation of the tree

        Args:
            quotes: Whether to include quotes around the operator

        Returns:
            A string representation of the tree
        """
        op = self.op
        if quotes:
            op = f"'{op}'"
        if self.children and self.id is not None:
            return f"Tree({op}, {self.children}, {self.id})"
        elif self.children:
            return f"Tree({op}, {self.children})"
        elif self.id is not None:
            return f"Tree({op}, {self.id})"
        return f"Tree({op})"

    def __str__(self):
        """Returns a string representation of the tree"""
        s = self._str()
        if s.startswith("("):
            return s[1:-1]
        return s

    def _str(self, simplify=True):
        """
        Returns a string representation of the tree

        Args:
            simplify: Whether to simplify the string representation

        Returns:
            A string representation of the tree
        """
        if not self.children:
            return self.op

        if len(self.children) == 1:
            if simplify:
                return self.op + self.children[0]._str()
            return "(" + self.op + self.children[0]._str() + ")"

        if len(self.children) == 2:
            return (
                "(" + self.children[0]._str() + self.op + self.children[1]._str() + ")"
            )

    def add_missing_ids(self, cnt=None):
        """
        Adds ids to all nodes that are missing them

        Args:
            cnt: The optional node id to start from

        Returns:
            The next available id
        """
        if cnt is None:
            all_ids = self.get_node_ids() - {None}
            cnt = max(all_ids) + 1 if all_ids else 0
        if self.id is None:
            self.id = cnt
            cnt += 1
        for child in self.children:
            cnt = child.add_missing_ids(cnt)
        return cnt

    def find_node(self, node_):
        """
        Finds a node in the tree by its id

        Args:
            node_: The id of the node

        Returns:
            The node if found, else None
        """
        if self.id == node_:
            return self

        if not self.children:
            return None

        for child in self.children:
            res_ = child.find_node(node_)
            if res_:
                return res_

    def get_leaf_nodes(self):
        """
        Recursively gets all leaf nodes in the tree

        Returns:
            A generator of all leaf nodes
        """
        if not self.children:
            yield self

        for child in self.children:
            yield from child.get_leaf_nodes()

    def get_node_ids(self):
        """
        Gets the ids of all the nodes of the tree

        Returns:
            A set of all the ids of the nodes
        """
        res_ = {self.id}
        for child in self.children:
            res_.update(child.get_node_ids())
        return res_

    def get_non_leaf_nodes(self):
        """
        Recursively gets all non-leaf nodes in the tree

        Returns:
            A generator of all non-leaf nodes
        """
        if self.children:
            yield self
            for child in self.children:
                yield from child.get_non_leaf_nodes()

    def get_nodes(self):
        """
        Recursively gets all nodes in the tree

        Returns:
            A generator of all nodes
        """
        yield self
        for child in self.children:
            yield from child.get_nodes()

    def get_intervened_trees(self, value_space_dict=None):
        """
        Performs all possible interventions on the tree and returns those in a dictionary

        Args:
            value_space_dict: A dict mapping each node to an iterable of possible values

        Returns:
            A dictionary of (node_id, value) -> tree
        """
        if value_space_dict is None:
            value_space_dict = defaultdict(lambda: [0, 1])
        trees = dict()
        for node_list in self._get_unique_intervention_nodes():
            ref_node = node_list[0]
            intervened_trees = [
                self.intervene(ref_node, value) for value in value_space_dict[ref_node]
            ]
            for node in node_list:
                for value in value_space_dict[node]:
                    trees[(node.id, value)] = intervened_trees[
                        value if node.op == ref_node.op else 1 - value
                    ]
        return trees

    def _get_unique_intervention_nodes(self):
        """
        Recursively gets all nodes in the tree on which a unique intervention can be performed
        (includes all non-leaf nodes and one leaf node for each variable)

        Return:
            A generator of unique intervention nodes
        """
        for node in self.get_non_leaf_nodes():
            yield (node,)

        variables = defaultdict(list)
        for leaf in self.get_leaf_nodes():
            variables[leaf.op[-1]].append(leaf)
        for var, nodes in variables.items():
            yield nodes

    def negate(self, node_ids):
        """
        Replaces the specified nodes with its negation

        Args:
            node_ids: The list of the id of the nodes to negate

        Returns:
            The modified formula
        """
        for i, child in enumerate(self.children):
            self.children[i] = child.negate(node_ids)
        if self.id in node_ids:
            return LogicTree("¬", [self])
        return self

    def intervene(self, node_, value_):
        """
        Returns a copy of the tree with an intervention applied to the given node

        Args:
            node_: The node to intervene on
            value_: The value to set the node to

        Returns:
            A copy of the tree with the intervention applied
        """
        tree_copy = copy.deepcopy(self)
        if (
            not node_.children
        ):  # Leaf variable: intervene on all other leaves for the same variable
            var = node_.op[-1]
            for leaf in tree_copy.get_leaf_nodes():
                if leaf.op[-1] == var:
                    tree_copy.do_intervene(
                        leaf, value_ if leaf.op == node_.op else 1 - value_
                    )
        else:
            tree_copy.do_intervene(node_, value_)
        return tree_copy

    def do_intervene(self, node_, value_):
        """
        Performs an intervention on the tree

        Args:
            node_: The node to intervene on
            value_: The value to set the node to
        """
        if self.id == node_.id:
            self.op = value_
            self.children = []
        else:
            for child in self.children:
                child.do_intervene(node_, value_)

    def sample(self, add_vars=None, device="cuda:"):
        """
        Gets all possible samples from the tree

        Args:
            add_vars: An iterable of additional variables to sample
            device: The device to use for the tensor

        Returns:
            A list of (assignments, input tensor) pairs
        """
        self.add_missing_ids()
        if add_vars is None:
            add_vars = set()
        all_vars = {x.op[-1] for x in self.get_leaf_nodes()} | add_vars
        samples_ = []

        for assignments in itertools.product((0, 1), repeat=len(all_vars)):
            val = dict(zip(all_vars, assignments))
            val.update(
                self.evaluate(dict(zip(all_vars, assignments)), return_dict=True)
            )
            samples_.append(
                (
                    val,
                    torch.tensor(
                        [val[var] for var in sorted(all_vars)],
                        dtype=torch.float32,
                        device=device,
                    ),
                )
            )
        return samples_

    def call(self, assignments):
        """
        Applies the function of the tree to the given variable assignments

        Args:
            assignments: The variable assignments

        Returns:
            The output values
        """
        assignments = {k: v for k, v in assignments.items() if isinstance(k, str)}
        return self.evaluate(assignments, return_dict=False)

    def evaluate(self, assignments, return_dict=False):
        """
        Applies the function of the tree to the given variable assignments

        Args:
            assignments: The variable assignments
            return_dict: Whether to return only the full output or a dictionary of all intermediate values

        Returns:
            The output value, or a dictionary of all intermediate values
        """
        if not self.children:
            if self.op in assignments:  # Unit variable
                val = assignments[self.op]
            elif isinstance(self.op, str):  # Negation of unit variable
                val = 1 - assignments[self.op[1:]]
            else:  # Immediate value
                val = self.op
            if return_dict:
                return {self.id: val}
            return val

        # Recursive call
        node_vals = {}
        for child in self.children:
            node_vals.update(child.evaluate(assignments, return_dict=True))

        val = OP_TO_TRUTH_TABLE[self.op]
        for child in self.children:
            val = val[node_vals[child.id]]
        node_vals[self.id] = val

        return node_vals if return_dict else node_vals[self.id]

    def visualize(
        self, ax=None, display_ids=True, node_size="small", file_path=None, labels=None
    ):
        """
        Visualize the tree

        Args:
            ax: The axis to plot on
            display_ids: Whether to display the id of each node
            node_size: The size of the nodes
            file_path: The path to save the figure to
            labels: The optional labels for each node
        """
        node_size = get_node_size(node_size)

        def calculate_positions(node, horiz_loc, ycenter, width):
            positions = {str(node.id): (horiz_loc, ycenter)}
            children = node.children

            if children:
                child_width = width / len(children)
                next_y = ycenter - width / 2 - child_width / 2
                for child in children:
                    next_y += child_width
                    child_positions = calculate_positions(
                        child, horiz_loc - 1, next_y, child_width
                    )
                    positions.update(child_positions)

                topmost = min(positions[str(child.id)][1] for child in children)
                bottommost = max(positions[str(child.id)][1] for child in children)
                positions[str(node.id)] = (horiz_loc, (topmost + bottommost) / 2)

            return positions

        # Create the graph and calculate positions
        G = nx.DiGraph()
        pos = calculate_positions(self, horiz_loc=0, ycenter=1 / 2, width=1)

        # Add nodes and edges
        queue = deque([self])
        while queue:
            node = queue.popleft()
            node_label = str(node.id)
            G.add_node(node_label, pos=pos[node_label])
            for child in node.children:
                child_label = str(child.id)
                G.add_edge(node_label, child_label)
                queue.append(child)

        if labels is None:
            labels = {
                str(node.id): OP_TO_STR.get(node.op, node.op)
                for node in self.get_nodes()
            }

        if ax is None:
            plt.figure()
            ax = plt.gca()

        colormap = plt.get_cmap("tab20")
        node_color = [colormap(int(i) % 20) for i in G.nodes]
        nx.draw(
            G,
            pos,
            with_labels=display_ids,
            labels=labels,
            node_size=node_size,
            font_size=15,
            node_color=node_color,
            edge_color="tab:blue",
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


def all_formulas(depth, variables):
    """
    Generates all possible formulas (LogicTree) up to a certain depth

    Args:
        depth: The maximum depth of the formula
        variables: The variables allowed in the formula

    Returns:
        A generator of all possible formulas
    """
    if depth == 0:
        for var in variables:
            yield LogicTree(var, [])
        return

    for formula1 in all_formulas(depth - 1, variables):
        for i_ in range(depth):
            for formula2 in all_formulas(i_, variables):
                if str(formula1) < str(formula2):
                    yield LogicTree(
                        "*", [copy.deepcopy(formula1), copy.deepcopy(formula2)]
                    )
                    yield LogicTree(
                        "+", [copy.deepcopy(formula1), copy.deepcopy(formula2)]
                    )

    return


def get_formula_dataset(logic_gates, max_depth, device="cuda:0"):
    """
    Generates a dataset of all possible formulas of a certain depth, precomputing all corresponding samples
    and interventions.

    Args:
        logic_gates: The list of all logic gates to generate formulas for
        max_depth: The maximum depth of the formulas
        device: The device to store the data

    Returns:
        A dictionary mapping each gate's name to a list of (formula, sample_pairs, interventions) tuples.
        sample_pairs is a list of (sample_1, sample_2) tuples, where each sample is a dictionary mapping each node's id
        to the value of that node.
        interventions is a dictionary mapping (node_id, node_value) tuples to the tree obtained after setting the node
        to the given value (i.e. performing a intervention).
    """
    formula_dataset = defaultdict(list)

    if len({gate.n_inputs for gate in logic_gates}) > 1:
        raise ValueError("Not all gates have the same number of inputs")

    n_inputs = logic_gates[0].n_inputs

    for gate in logic_gates:
        truth_table = tuple(int(x[1]) for x in sorted(gate.truth_table().items()))
        cnt = 0

        # Generate all formulas of depth <= max_depth, using AND and OR only (ignore NOT for now)
        variables = tuple(string.ascii_uppercase[:n_inputs])
        formulas = []
        for depth in range(max_depth + 1):
            formulas.extend(all_formulas(depth, variables))
        seen = set()

        for formula in formulas:
            formula.add_missing_ids()

            # Apply NOT to all possible combinations of nodes (elements of the powerset of nodes)
            node_ids = formula.get_node_ids()
            for negation_nodes in powerset(node_ids):
                neg_formula = copy.deepcopy(formula)
                neg_formula = neg_formula.negate(negation_nodes)
                neg_formula.add_missing_ids()

                # Check whether the negated formula is equivalent to the searched gate
                formula_ok = True
                for assign_val in range(2**n_inputs):
                    assign = dict(
                        zip(variables, map(int, bin(assign_val)[2:].zfill(n_inputs)))
                    )
                    val = neg_formula.evaluate(assign, return_dict=True)[neg_formula.id]
                    if val != truth_table[assign_val]:
                        formula_ok = False
                        break
                if formula_ok:
                    mod_formula = str(formula)
                    if mod_formula not in seen:
                        cnt += 1
                        seen.add(mod_formula)

                        samples = neg_formula.sample(
                            add_vars=set(string.ascii_uppercase[:n_inputs]),
                            device=device,
                        )
                        sample_pairs = [
                            (samples[i], samples[j])
                            for i in range(len(samples))
                            for j in range(i + 1, len(samples))
                        ]

                        intervened_trees = neg_formula.get_intervened_trees()

                        formula_dataset[gate.name].append(
                            (neg_formula, sample_pairs, intervened_trees)
                        )
                    break
    return formula_dataset


# Various standard logic gate functions


def xor_func(inputs, epsilon=0.5):
    inputs_bin = threshold(inputs, epsilon)
    return np.sum(inputs_bin, axis=1) % 2


def and_func(inputs, epsilon=0.5):
    inputs_bin = threshold(inputs, epsilon)
    return np.all(inputs_bin, axis=1).astype(np.float32)


def or_func(inputs, epsilon=0.5):
    inputs_bin = threshold(inputs, epsilon)
    return np.any(inputs_bin, axis=1).astype(np.float32)


def implication_func(inputs, epsilon=0.5):
    inputs_bin = threshold(inputs, epsilon)
    A_not = 1 - inputs_bin[:, 0]  # Logical NOT: 1 - A
    B = inputs_bin[:, 1]
    return or_func(
        np.stack([A_not, B], axis=1), epsilon
    )  # A -> B is equivalent to (not A) OR B


def nand_func(inputs, epsilon=0.5):
    inputs_bin = threshold(inputs, epsilon)
    return (1 - np.all(inputs_bin, axis=1)).astype(np.float32)


def nor_func(inputs, epsilon=0.5):
    inputs_bin = threshold(inputs, epsilon)
    return (1 - np.any(inputs_bin, axis=1)).astype(np.float32)


def not_implication_func(inputs, epsilon=0.5):
    inputs_bin = threshold(inputs, epsilon)
    A = inputs_bin[:, 0]
    B = inputs_bin[:, 1]
    return (A * (1 - B)).astype(np.float32)  # NOT Implication: A AND (NOT B)


def majority_func(inputs):
    return (np.sum(inputs, axis=1) > (inputs.shape[1] // 2)).astype(np.float32)


def parity_func(inputs):
    return (np.sum(inputs, axis=1) % 2).astype(np.float32)


def full_adder_func(inputs):
    """
    Full Adder function that computes the sum and carry of three binary inputs.

    Args:
        inputs (np.ndarray): A 2D array with exactly 3 columns representing the inputs
                             (a, b, and carry-in). Shape should be (num_samples, 3).

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The sum output of the Full Adder. Shape will be (num_samples,).
            - np.ndarray: The carry output of the Full Adder. Shape will be (num_samples,).
    """
    assert inputs.shape[1] == 3, "Full Adder requires exactly 3 inputs"
    # Compute sum and carry outputs
    sum_output = (np.sum(inputs, axis=1) % 2).astype(np.float32)
    carry_output = (np.sum(inputs, axis=1) > 1).astype(np.float32)
    return sum_output  # , carry_output


def threshold(inputs, epsilon=0.5):
    # Threshold function: maps values to 0 or 1 based on proximity
    return (inputs >= epsilon).astype(np.float32)


def mux_func(inputs):
    """
    Multiplexer function that selects one of the inputs based on the selection bits.

    Args:
        inputs (np.ndarray): A 2D array where the first part represents the selection bits
                             and the second part represents the data inputs.
                             Shape should be (num_samples, 2 * num_select_bits) where the
                             first half is selection bits and the second half is data inputs.

    Returns:
        np.ndarray: The selected values from the data inputs based on the selection bits.
                    Shape will be (num_samples,).
    """
    n_select_bits = inputs.shape[1] // 2
    # Convert the selection bits to indices
    selected_indices = np.packbits(inputs[:, :n_select_bits], axis=-1)[:, 0] % (
        inputs.shape[1] - n_select_bits
    )
    # Select the corresponding data input values
    return np.take_along_axis(
        inputs[:, n_select_bits:], selected_indices[:, None], axis=1
    ).flatten()


def aoi_func(inputs):
    """
    AOI (AND-OR-Invert) function that performs the operation: NOT((AND(inputs[:half]) OR OR(inputs[half:])))

    Args:
        inputs (np.ndarray): A 2D array where the first half of the columns are used for AND operation
                             and the second half for OR operation.
                             Shape should be (num_samples, num_inputs) where num_inputs is even.

    Returns:
        np.ndarray: The result of the AOI operation.
                    Shape will be (num_samples,).
    """
    half = inputs.shape[1] // 2
    # Compute the AND and OR results
    and_result = np.all(inputs[:, :half], axis=1)
    or_result = np.any(inputs[:, half:], axis=1)
    # Return the negation of the OR result of AND and OR
    return (~(and_result | or_result)).astype(np.float32)


def exactly_k_func(inputs, k):
    """
    Function that checks if exactly k bits are set to 1 in the input.

    Args:
        inputs (np.ndarray): A 2D array where each row is a binary input.
                             Shape should be (num_samples, num_inputs).
        k (int): The exact number of bits that should be set to 1.

    Returns:
        np.ndarray: 1 if exactly k bits are set to 1, otherwise 0.
                    Shape will be (num_samples,).
    """
    return (np.sum(inputs, axis=1) == k).astype(np.float32)


def conditional_func(inputs):
    """
    Conditional function that returns 1 if the number of 1s in the input is greater than or equal to 2.

    Args:
        inputs (np.ndarray): A 2D array where each row is a binary input.
                             Shape should be (num_samples, num_inputs).

    Returns:
        np.ndarray: 1 if the number of 1s in the input is greater than or equal to 2, otherwise 0.
                    Shape will be (num_samples,).
    """
    return (np.sum(inputs, axis=1) >= 2).astype(np.float32)


def print_truth_table_multiple(gates, logger):
    """
    Prints the truth table for multiple logic gates.

    Args:
        gates: The logic gates to print the truth table for.
        logger: The logger to use
    """
    # Determine the number of inputs based on the first logic gate
    n_inputs = len(list(gates[0].keys())[0])

    # Log the header
    header = (
        " | ".join([f"x{i}" for i in range(n_inputs)])
        + " | "
        + " | ".join([f"Gate {i + 1}" for i in range(len(gates))])
    )
    logger.info(header)
    logger.info("-" * len(header))

    # Generate all input combinations based on the number of inputs
    from itertools import product

    input_combinations = list(product([0, 1], repeat=n_inputs))

    # Log each row of the truth table for all gates
    for input_row in input_combinations:
        input_str = " | ".join(
            map(str, map(int, input_row))
        )  # String for the input part
        output_str = " | ".join(
            [f"{int(gate[input_row])}" for gate in gates]
        )  # Outputs for each gate
        logger.info(f"{input_str} |   {output_str}")


def name_gate(truth_table):
    """
    Names a logic gate based on its truth table

    Args:
        truth_table: The truth table of the logic gate

    Returns:
        The name of the logic gate
    """
    n_inputs = len(list(truth_table.keys())[0])  # Determine the number of inputs
    tt_output = [val[1] for val in sorted(truth_table.items())]  # For ex. [0, 1, 1, 0]
    gate_id = int("".join(map(str, tt_output)), 2)  # For ex [0, 1, 1, 0] -> 0b0110 -> 6

    gate_list = [
        ["FALSE", "TRUE"],  # 0 inputs
        ["FALSE", "IDENTITY", "NOT", "TRUE"],  # 1 input
        [
            "FALSE",
            "AND",
            "A_NIMPLY_B",
            "A",
            "B_NIMPLY_A",
            "B",
            "XOR",
            "OR",
            "NOR",
            "XNOR",
            "NOT_B",
            "B_IMPLY_A",
            "NOT_A",
            "A_IMPLY_B",
            "NAND",
            "TRUE",
        ],  # 2 inputs
    ]

    if n_inputs <= 2:
        return gate_list[n_inputs][gate_id]

    return f"{n_inputs}INPUTS"  # For more than 2 inputs, return a generic name


OP_TO_STR = {
    "¬": "NOT",
    "+": "OR",
    "*": "AND",
    "x": "XOR",
    "=": "EQV",
    ">": "IMP",
}

OP_TO_TRUTH_TABLE = {
    "¬": (1, 0),
    "+": ((0, 1), (1, 1)),
    "*": ((0, 0), (0, 1)),
    "x": ((0, 1), (1, 0)),
    "=": ((1, 0), (0, 1)),
    ">": ((1, 1), (0, 1)),
}

ALL_LOGIC_GATES: Mapping[str, LogicGate] = {
    "IMP": LogicGate(n_inputs=2, gate_fn=implication_func, name="IMP"),
    "XOR": LogicGate(n_inputs=2, gate_fn=xor_func, name="XOR"),
    "AND": LogicGate(n_inputs=2, gate_fn=and_func, name="AND"),
    "OR": LogicGate(n_inputs=2, gate_fn=or_func, name="OR"),
    "NAND": LogicGate(n_inputs=2, gate_fn=nand_func, name="NAND"),
    "NOR": LogicGate(n_inputs=2, gate_fn=nor_func, name="NOR"),
    "NIMP": LogicGate(n_inputs=2, gate_fn=not_implication_func, name="NIMP"),
    # "MAJORITY": LogicGate(n_inputs=3, gate_fn=majority_func, name="MAJORITY"),
    # "PARITY": LogicGate(n_inputs=3, gate_fn=parity_func, name="PARITY"),
    # "FULL_ADDER": LogicGate(n_inputs=3, gate_fn=full_adder_func, name="FULL_ADDER"),
    # "EXACT_2": LogicGate(
    #     n_inputs=3, gate_fn=lambda x: exactly_k_func(x, 2), name="EXACT_2"
    # ),
}
