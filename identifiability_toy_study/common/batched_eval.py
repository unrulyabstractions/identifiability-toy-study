"""
Batched evaluation of subcircuits using GPU parallelism.

Instead of running each subcircuit separately, we:
1. Run the full model once to get activations
2. Apply all circuit masks in parallel using batched tensor ops
"""

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from .circuit import Circuit
    from .neural_model import MLP


def batch_evaluate_subcircuits(
    model: "MLP",
    circuits: list["Circuit"],
    x: torch.Tensor,
    gate_idx: int = 0,
    precomputed_masks: list[torch.Tensor] | None = None,
    eval_device: str | None = None,
) -> torch.Tensor:
    """
    Evaluate all subcircuits in a single batched computation.

    Args:
        model: The full trained model
        circuits: List of Circuit objects to evaluate
        x: Input tensor [batch, input_size]
        gate_idx: Which output gate to evaluate (for multi-gate models)
        precomputed_masks: Optional pre-stacked masks per layer (from precompute_circuit_masks)
        eval_device: Device to run evaluation on (defaults to model.device).
            Useful for running on MPS when model is on CPU.

    Returns:
        Tensor of shape [n_circuits, batch, 1] with predictions for each circuit
    """
    device = eval_device if eval_device is not None else model.device
    n_circuits = len(circuits)
    n_layers = len(model.layers)

    # Ensure input is on correct device
    if x.device.type != device:
        x = x.to(device)

    # Get pre-stacked masks or compute them
    if precomputed_masks is not None:
        layer_masks_list = precomputed_masks
        # Verify masks are on correct device
        if layer_masks_list[0].device.type != device:
            layer_masks_list = [m.to(device) for m in layer_masks_list]
    else:
        layer_masks_list = precompute_circuit_masks(
            circuits, n_layers, gate_idx, device
        )

    # Run batched forward pass with masked weights
    with torch.inference_mode():
        # Get layer weights and move to eval device if needed
        weights = [layer[0].weight.to(device) for layer in model.layers]
        biases = [layer[0].bias.to(device) for layer in model.layers]

        # For the last layer, slice to single output
        if model.output_size > 1:
            weights[-1] = weights[-1][gate_idx : gate_idx + 1, :]
            biases[-1] = biases[-1][gate_idx : gate_idx + 1]

        # Expand input for all circuits: [n_circuits, batch, input_size]
        h = x.unsqueeze(0).expand(n_circuits, -1, -1)

        for layer_idx in range(len(weights)):
            W = weights[layer_idx]  # [out, in]
            b = biases[layer_idx]  # [out]

            # Get pre-stacked masks for this layer: [n_circuits, out, in]
            layer_masks = layer_masks_list[layer_idx]

            # Apply masked weights: W_eff[c] = W * mask[c]
            W_masked = W.unsqueeze(0) * layer_masks  # [n_circuits, out, in]

            # Batched linear: h @ W_masked.T + b
            h = torch.bmm(h, W_masked.transpose(1, 2)) + b

            # Apply activation (except last layer)
            if layer_idx < len(weights) - 1:
                h = torch.nn.functional.leaky_relu(h)

    return h  # [n_circuits, batch, 1]


def precompute_circuit_masks_base(
    circuits: list["Circuit"],
    n_layers: int,
    device: str = "cpu",
) -> list[torch.Tensor]:
    """
    Pre-compute and stack all circuit edge masks WITHOUT output slicing.

    This is the optimized version that computes masks once for all gates.
    Use adapt_masks_for_gate to slice for a specific gate.

    Args:
        circuits: List of Circuit objects
        n_layers: Number of layers in the model
        device: Target device

    Returns:
        List of tensors, one per layer, each of shape [n_circuits, out, in]
    """
    # Build numpy arrays first (fast), then convert to tensor once
    layer_masks_np = [[] for _ in range(n_layers)]

    for circuit in circuits:
        for layer_idx, mask in enumerate(circuit.edge_masks):
            layer_masks_np[layer_idx].append(mask)

    # Stack and convert to tensors on device
    layer_masks_list = []
    for layer_idx in range(n_layers):
        stacked = np.stack(layer_masks_np[layer_idx])
        tensor = torch.tensor(stacked, dtype=torch.float32, device=device)
        layer_masks_list.append(tensor)

    return layer_masks_list


def adapt_masks_for_gate(
    base_masks: list[torch.Tensor],
    gate_idx: int,
    output_size: int,
) -> list[torch.Tensor]:
    """
    Adapt base masks for a specific gate by slicing the last layer.

    This is a cheap operation (just slicing) compared to recomputing all masks.

    Args:
        base_masks: Pre-computed masks from precompute_circuit_masks_base
        gate_idx: Which output gate to slice
        output_size: Total number of output gates

    Returns:
        List of tensors with last layer sliced for gate_idx
    """
    if output_size == 1:
        return base_masks

    # Only need to slice the last layer
    result = base_masks[:-1] + [base_masks[-1][:, gate_idx : gate_idx + 1, :]]
    return result


def precompute_circuit_masks(
    circuits: list["Circuit"],
    n_layers: int,
    gate_idx: int = 0,
    device: str = "cpu",
) -> list[torch.Tensor]:
    """
    Pre-compute and stack all circuit edge masks for efficient batched evaluation.

    This function creates tensors on CPU first, stacks them, then moves to device
    in a single operation - much faster than creating many small device tensors.

    For multi-gate models, consider using precompute_circuit_masks_base once
    and adapt_masks_for_gate for each gate (more efficient).

    Args:
        circuits: List of Circuit objects
        n_layers: Number of layers in the model
        gate_idx: Which output gate (for slicing last layer)
        device: Target device

    Returns:
        List of tensors, one per layer, each of shape [n_circuits, out, in]
    """
    # Build numpy arrays first (fast), then convert to tensor once
    layer_masks_np = [[] for _ in range(n_layers)]

    for circuit in circuits:
        for layer_idx, mask in enumerate(circuit.edge_masks):
            # For last layer, slice to single output if multi-gate
            if layer_idx == n_layers - 1 and mask.shape[0] > 1:
                mask = mask[gate_idx : gate_idx + 1, :]
            layer_masks_np[layer_idx].append(mask)

    # Stack and convert to tensors on device
    layer_masks_list = []
    for layer_idx in range(n_layers):
        stacked = np.stack(layer_masks_np[layer_idx])
        tensor = torch.tensor(stacked, dtype=torch.float32, device=device)
        layer_masks_list.append(tensor)

    return layer_masks_list


def batch_compute_metrics(
    model: "MLP",
    circuits: list["Circuit"],
    x: torch.Tensor,
    y_target: torch.Tensor,
    y_pred: torch.Tensor,
    gate_idx: int = 0,
    precomputed_masks: list[torch.Tensor] | None = None,
    eval_device: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute accuracy, logit_similarity, bit_similarity, best_similarity for all circuits.

    Args:
        model: The full trained model
        circuits: List of circuits to evaluate
        x: Input tensor [batch, input_size]
        y_target: Ground truth [batch, 1]
        y_pred: Model predictions [batch, 1]
        gate_idx: Which output gate
        precomputed_masks: Optional pre-stacked masks (from precompute_circuit_masks)
        eval_device: Device to run evaluation on (defaults to model.device)

    Returns:
        Tuple of (accuracies, logit_similarities, bit_similarities, best_similarities) arrays
    """
    device = eval_device if eval_device is not None else model.device

    # Ensure all inputs are on the same device
    if x.device.type != device:
        x = x.to(device)
    if y_target.device.type != device:
        y_target = y_target.to(device)
    if y_pred.device.type != device:
        y_pred = y_pred.to(device)

    # Batch evaluate all circuits
    y_circuits = batch_evaluate_subcircuits(
        model,
        circuits,
        x,
        gate_idx,
        precomputed_masks=precomputed_masks,
        eval_device=device,
    )
    # y_circuits: [n_circuits, batch, 1]

    bit_target = torch.round(y_target)  # [batch, 1]
    bit_pred = torch.round(y_pred)  # [batch, 1]

    # Compute metrics for all circuits at once
    bit_circuits = torch.round(y_circuits)  # [n_circuits, batch, 1]

    # Best: clamp to binary [0,1] after rounding (handles out-of-range outputs)
    best_circuits = torch.clamp(bit_circuits, 0, 1)  # [n_circuits, batch, 1]
    best_pred = torch.clamp(bit_pred, 0, 1)  # [batch, 1]

    # Accuracy: match with ground truth
    # [n_circuits, batch, 1] == [1, batch, 1] -> [n_circuits, batch, 1]
    correct = bit_circuits.eq(bit_target.unsqueeze(0))
    accuracies = correct.float().mean(dim=(1, 2)).cpu().numpy()

    # Bit similarity: match with model predictions
    same_as_model = bit_circuits.eq(bit_pred.unsqueeze(0))
    bit_similarities = same_as_model.float().mean(dim=(1, 2)).cpu().numpy()

    # Best similarity: match after clamping to [0,1]
    same_as_model_best = best_circuits.eq(best_pred.unsqueeze(0))
    best_similarities = same_as_model_best.float().mean(dim=(1, 2)).cpu().numpy()

    # Logit similarity: 1 - MSE
    # y_circuits: [n_circuits, batch, 1], y_pred: [batch, 1]
    mse = ((y_circuits - y_pred.unsqueeze(0)) ** 2).mean(dim=(1, 2))
    logit_similarities = (1 - mse).detach().cpu().numpy()

    return accuracies, logit_similarities, bit_similarities, best_similarities


def batch_evaluate_edge_variants(
    model: "MLP",
    base_circuits: list["Circuit"],
    x: torch.Tensor,
    y_target: torch.Tensor,
    y_pred: torch.Tensor,
    gate_idx: int = 0,
    eval_device: str | None = None,
) -> list[tuple[int, "Circuit", float, float, float]]:
    """
    For each node pattern, enumerate edge variants and find the best one.

    This function explores edge configurations for promising node patterns.
    It returns the best edge configuration for each input circuit.

    Args:
        model: The full trained model
        base_circuits: List of circuits with good node patterns (from filter_subcircuits)
        x: Input tensor [batch, input_size]
        y_target: Ground truth [batch, 1]
        y_pred: Model predictions [batch, 1]
        gate_idx: Which output gate
        eval_device: Device for evaluation

    Returns:
        List of (original_idx, best_circuit, accuracy, logit_sim, bit_sim)
        for each input circuit, with the best edge configuration found.
    """
    from .circuit import enumerate_edge_variants

    device = eval_device if eval_device is not None else model.device

    results = []

    for orig_idx, base_circuit in enumerate(base_circuits):
        # Enumerate all valid edge configurations for this node pattern
        edge_variants = enumerate_edge_variants(base_circuit)

        if not edge_variants:
            # No valid edge variants (shouldn't happen for valid circuits)
            results.append((orig_idx, base_circuit, 0.0, 0.0, 0.0))
            continue

        # If only one variant (full connectivity), no exploration needed
        if len(edge_variants) == 1:
            accs, logit_sims, bit_sims, _ = batch_compute_metrics(
                model, edge_variants, x, y_target, y_pred,
                gate_idx=gate_idx, eval_device=device
            )
            results.append((
                orig_idx, edge_variants[0],
                float(accs[0]), float(logit_sims[0]), float(bit_sims[0])
            ))
            continue

        # Batch evaluate all edge variants
        accs, logit_sims, bit_sims, _ = batch_compute_metrics(
            model, edge_variants, x, y_target, y_pred,
            gate_idx=gate_idx, eval_device=device
        )

        # Find best by (accuracy DESC, bit_similarity DESC, edge_sparsity DESC)
        best_idx = 0
        best_score = (-accs[0], -bit_sims[0], -edge_variants[0].sparsity()[1])

        for i in range(1, len(edge_variants)):
            score = (-accs[i], -bit_sims[i], -edge_variants[i].sparsity()[1])
            if score < best_score:
                best_score = score
                best_idx = i

        results.append((
            orig_idx, edge_variants[best_idx],
            float(accs[best_idx]), float(logit_sims[best_idx]), float(bit_sims[best_idx])
        ))

    return results
