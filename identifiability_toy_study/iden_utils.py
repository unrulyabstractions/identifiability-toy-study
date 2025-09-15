import torch
import torch.nn as nn
from tqdm import tqdm

from identifiability_toy_study.mi_identifiability.circuit import (
    Circuit,
)
from identifiability_toy_study.mi_identifiability.neural_model import MLP
from identifiability_toy_study.study_core import (
    CircuitMetrics,
    Dataset,
    GateMetrics,
    IdentifiabilityConstraints,
)


def calculate_match_rate(y_pred, y_gt):
    y_pred = y_pred.reshape(-1)
    y_gt = y_gt.reshape(-1)
    return y_pred.eq(y_gt).float().mean()  # returns 0-D tensor


def filter_by_constraints(
    constraints: IdentifiabilityConstraints,
    dataset: Dataset,
    model: MLP,
    circuits: list[Circuit],
    device: str,
    logger=None,
    use_tqdm=True,
) -> GateMetrics:
    num_total_circuits = len(circuits)

    x = dataset.x
    y_gt = dataset.y
    bit_gt = torch.round(y_gt)

    y_model = model(x)
    bit_model = torch.round(y_model)
    test_acc = calculate_match_rate(bit_model, bit_gt).item()

    per_circuit = {}
    faithful_circuits_idx = []

    # Iterate over all circuits with progress tracking
    it = enumerate(circuits)
    if use_tqdm:
        it = tqdm(
            it, total=num_total_circuits, desc="Filtering circuits by constraints"
        )
    for i, circuit in it:
        # Make predictions with the current circuit
        y_circuit = model(x, circuit=circuit)
        bit_circuit = torch.round(y_circuit)

        # Compute the accuracy with respect to the task
        accuracy = calculate_match_rate(bit_circuit, bit_gt).item()
        logit_similarity = 1 - nn.MSELoss()(y_model, y_circuit).item()
        bit_similarity = calculate_match_rate(bit_circuit, bit_model).item()

        # Compute circuit sparsity
        node_spar, edge_spar, combined_spar = circuit.sparsity()

        per_circuit[i] = CircuitMetrics(
            circuit_idx=i,
            accuracy=accuracy,
            logit_similarity=logit_similarity,
            bit_similarity=bit_similarity,
            sparsity=circuit.sparsity(),
        )

        if node_spar < constraints.min_sparsity:
            continue

        if bit_similarity < constraints.acc_threshold:
            continue

        faithful_circuits_idx.append(i)

    return GateMetrics(
        test_acc=test_acc,
        num_total_circuits=num_total_circuits,
        per_circuit=per_circuit,
        faithful_circuits_idx=faithful_circuits_idx,
    )
