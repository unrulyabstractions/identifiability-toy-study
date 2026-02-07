"""
Batched evaluation of subcircuits using GPU parallelism.

Instead of running each subcircuit separately, we:
1. Run the full model once to get activations
2. Apply all circuit masks in parallel using batched tensor ops

Large batches are automatically chunked to avoid OOM errors.
"""

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from .circuit import Circuit
    from .neural_model import MLP

# Default max circuits per batch to avoid OOM
# 64 is very conservative for CPU; can be increased for GPU with more memory
DEFAULT_MAX_BATCH_SIZE = 64


def _evaluate_subcircuits_chunk(
    x: torch.Tensor,
    weights: list[torch.Tensor],
    biases: list[torch.Tensor],
    layer_masks_list: list[torch.Tensor],
) -> torch.Tensor:
    """
    Evaluate a chunk of subcircuits (internal helper).

    Args:
        x: Input tensor [batch, input_size]
        weights: List of weight tensors per layer
        biases: List of bias tensors per layer
        layer_masks_list: List of mask tensors [n_circuits_chunk, out, in] per layer

    Returns:
        Tensor [n_circuits_chunk, batch, 1]
    """
    n_circuits = layer_masks_list[0].shape[0]

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


def batch_evaluate_subcircuits(
    model: "MLP",
    circuits: list["Circuit"],
    x: torch.Tensor,
    gate_idx: int = 0,
    precomputed_masks: list[torch.Tensor] | None = None,
    eval_device: str | None = None,
    max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
) -> torch.Tensor:
    """
    Evaluate all subcircuits, chunking if necessary to avoid OOM.

    Args:
        model: The full trained model
        circuits: List of Circuit objects to evaluate
        x: Input tensor [batch, input_size]
        gate_idx: Which output gate to evaluate (for multi-gate models)
        precomputed_masks: Optional pre-stacked masks per layer (from precompute_circuit_masks)
        eval_device: Device to run evaluation on (defaults to model.device).
        max_batch_size: Maximum circuits per batch to avoid OOM errors.

    Returns:
        Tensor of shape [n_circuits, batch, 1] with predictions for each circuit
    """
    device = eval_device if eval_device is not None else model.device
    n_circuits = len(circuits)
    n_layers = len(model.layers)

    # Ensure input is on correct device
    if x.device.type != device:
        x = x.to(device)

    # Run batched forward pass with masked weights
    with torch.inference_mode():
        # Get layer weights and move to eval device if needed
        weights = [layer[0].weight.to(device) for layer in model.layers]
        biases = [layer[0].bias.to(device) for layer in model.layers]

        # For the last layer, slice to single output
        if model.output_size > 1:
            weights[-1] = weights[-1][gate_idx : gate_idx + 1, :]
            biases[-1] = biases[-1][gate_idx : gate_idx + 1]

        # If precomputed masks provided, use them
        if precomputed_masks is not None:
            layer_masks_list = precomputed_masks
            if layer_masks_list[0].device.type != device:
                layer_masks_list = [m.to(device) for m in layer_masks_list]

            # If small enough, process in one batch
            if n_circuits <= max_batch_size:
                return _evaluate_subcircuits_chunk(x, weights, biases, layer_masks_list)

            # Otherwise, chunk the circuits (masks already computed)
            results = []
            for start_idx in range(0, n_circuits, max_batch_size):
                end_idx = min(start_idx + max_batch_size, n_circuits)
                chunk_masks = [m[start_idx:end_idx] for m in layer_masks_list]
                chunk_result = _evaluate_subcircuits_chunk(x, weights, biases, chunk_masks)
                results.append(chunk_result)
            return torch.cat(results, dim=0)

        # No precomputed masks - compute per-chunk to avoid OOM on large circuit lists
        # This is crucial for edge variant evaluation with millions of circuits
        if n_circuits <= max_batch_size:
            # Small enough to compute all masks at once
            layer_masks_list = precompute_circuit_masks(circuits, n_layers, gate_idx, device)
            return _evaluate_subcircuits_chunk(x, weights, biases, layer_masks_list)

        # Large circuit list - compute masks incrementally per chunk
        results = []
        for start_idx in range(0, n_circuits, max_batch_size):
            end_idx = min(start_idx + max_batch_size, n_circuits)
            chunk_circuits = circuits[start_idx:end_idx]

            # Compute masks only for this chunk
            chunk_masks = precompute_circuit_masks(chunk_circuits, n_layers, gate_idx, device)

            # Evaluate chunk
            chunk_result = _evaluate_subcircuits_chunk(x, weights, biases, chunk_masks)
            results.append(chunk_result.cpu())  # Move to CPU to free GPU memory

            # Explicitly delete chunk masks to free memory
            del chunk_masks

        # Concatenate all chunks (on CPU, then move back if needed)
        result = torch.cat(results, dim=0)
        if device != "cpu":
            result = result.to(device)
        return result


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


def _compute_chunk_metrics(
    y_chunk: torch.Tensor,
    bit_target: torch.Tensor,
    bit_pred: torch.Tensor,
    best_pred: torch.Tensor,
    y_pred: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute metrics for a chunk of circuit outputs."""
    bit_chunk = torch.round(y_chunk)
    best_chunk = torch.clamp(bit_chunk, 0, 1)

    # Accuracy
    correct = bit_chunk.eq(bit_target.unsqueeze(0))
    accuracies = correct.float().mean(dim=(1, 2)).cpu().numpy()

    # Bit similarity
    same_as_model = bit_chunk.eq(bit_pred.unsqueeze(0))
    bit_sims = same_as_model.float().mean(dim=(1, 2)).cpu().numpy()

    # Best similarity
    same_best = best_chunk.eq(best_pred.unsqueeze(0))
    best_sims = same_best.float().mean(dim=(1, 2)).cpu().numpy()

    # Logit similarity
    mse = ((y_chunk - y_pred.unsqueeze(0)) ** 2).mean(dim=(1, 2))
    logit_sims = (1 - mse).detach().cpu().numpy()

    return accuracies, logit_sims, bit_sims, best_sims


def batch_compute_metrics(
    model: "MLP",
    circuits: list["Circuit"],
    x: torch.Tensor,
    y_target: torch.Tensor,
    y_pred: torch.Tensor,
    gate_idx: int = 0,
    precomputed_masks: list[torch.Tensor] | None = None,
    eval_device: str | None = None,
    max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute accuracy, logit_similarity, bit_similarity, best_similarity for all circuits.

    Large circuit lists are automatically chunked to avoid OOM errors.
    Metrics are computed incrementally per chunk to avoid storing all outputs.

    Args:
        model: The full trained model
        circuits: List of circuits to evaluate
        x: Input tensor [batch, input_size]
        y_target: Ground truth [batch, 1]
        y_pred: Model predictions [batch, 1]
        gate_idx: Which output gate
        precomputed_masks: Optional pre-stacked masks (from precompute_circuit_masks)
        eval_device: Device to run evaluation on (defaults to model.device)
        max_batch_size: Maximum circuits per batch to avoid OOM errors.

    Returns:
        Tuple of (accuracies, logit_similarities, bit_similarities, best_similarities) arrays
    """
    device = eval_device if eval_device is not None else model.device
    n_circuits = len(circuits)
    n_layers = len(model.layers)

    # Ensure all inputs are on the same device
    if x.device.type != device:
        x = x.to(device)
    if y_target.device.type != device:
        y_target = y_target.to(device)
    if y_pred.device.type != device:
        y_pred = y_pred.to(device)

    # Precompute target values
    bit_target = torch.round(y_target)
    bit_pred = torch.round(y_pred)
    best_pred = torch.clamp(bit_pred, 0, 1)

    # If small enough and precomputed masks available, use fast path
    if n_circuits <= max_batch_size:
        y_circuits = batch_evaluate_subcircuits(
            model, circuits, x, gate_idx,
            precomputed_masks=precomputed_masks,
            eval_device=device,
            max_batch_size=max_batch_size,
        )
        return _compute_chunk_metrics(y_circuits, bit_target, bit_pred, best_pred, y_pred)

    # Large circuit list - compute metrics incrementally per chunk
    # This avoids storing all outputs at once
    all_accs = []
    all_logit_sims = []
    all_bit_sims = []
    all_best_sims = []

    with torch.inference_mode():
        # Get layer weights once
        weights = [layer[0].weight.to(device) for layer in model.layers]
        biases = [layer[0].bias.to(device) for layer in model.layers]
        if model.output_size > 1:
            weights[-1] = weights[-1][gate_idx : gate_idx + 1, :]
            biases[-1] = biases[-1][gate_idx : gate_idx + 1]

        for start_idx in range(0, n_circuits, max_batch_size):
            end_idx = min(start_idx + max_batch_size, n_circuits)
            chunk_circuits = circuits[start_idx:end_idx]

            # Compute masks for this chunk
            chunk_masks = precompute_circuit_masks(chunk_circuits, n_layers, gate_idx, device)

            # Evaluate chunk
            y_chunk = _evaluate_subcircuits_chunk(x, weights, biases, chunk_masks)

            # Compute metrics for chunk
            accs, logit_sims, bit_sims, best_sims = _compute_chunk_metrics(
                y_chunk, bit_target, bit_pred, best_pred, y_pred
            )

            all_accs.append(accs)
            all_logit_sims.append(logit_sims)
            all_bit_sims.append(bit_sims)
            all_best_sims.append(best_sims)

            # Free memory
            del chunk_masks, y_chunk

    return (
        np.concatenate(all_accs),
        np.concatenate(all_logit_sims),
        np.concatenate(all_bit_sims),
        np.concatenate(all_best_sims),
    )


def batch_evaluate_edge_variants(
    model: "MLP",
    base_circuits: list["Circuit"],
    x: torch.Tensor,
    y_target: torch.Tensor,
    y_pred: torch.Tensor,
    gate_idx: int = 0,
    eval_device: str | None = None,
    max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
    max_memory_gb: float = 4.0,
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
        max_batch_size: Maximum circuits per batch to avoid OOM errors.
        max_memory_gb: Target max memory usage in GB. Batch size adapts to stay under this.

    Returns:
        List of (original_idx, best_circuit, accuracy, logit_sim, bit_sim)
        for each input circuit, with the best edge configuration found.
    """
    from .circuit import enumerate_edge_variants

    device = eval_device if eval_device is not None else model.device

    # Collect all edge variants with their source indices
    all_variants = []  # List of (orig_idx, variant_idx_in_orig, circuit)
    variant_ranges = []  # (start, end) for each original circuit

    for orig_idx, base_circuit in enumerate(base_circuits):
        edge_variants = enumerate_edge_variants(base_circuit)
        start = len(all_variants)
        for var_idx, variant in enumerate(edge_variants):
            all_variants.append((orig_idx, var_idx, variant))
        end = len(all_variants)
        variant_ranges.append((start, end, edge_variants))

    if not all_variants:
        # No variants at all
        return [(i, c, 0.0, 0.0, 0.0) for i, c in enumerate(base_circuits)]

    # Extract just the circuits for batch evaluation
    circuits_to_eval = [v[2] for v in all_variants]
    n_variants = len(circuits_to_eval)

    # Estimate memory per circuit and adapt batch size
    # Memory ~ n_circuits * batch_size * hidden_size * 4 bytes * num_layers
    batch_size = x.shape[0]
    hidden_size = max(mask.shape[0] for mask in circuits_to_eval[0].edge_masks)
    n_layers = len(circuits_to_eval[0].edge_masks)
    bytes_per_circuit = batch_size * hidden_size * 4 * n_layers * 2  # factor of 2 for safety
    max_circuits_in_memory = int((max_memory_gb * 1e9) / bytes_per_circuit)
    adaptive_batch_size = max(1, min(max_batch_size, max_circuits_in_memory))

    print(f"    [EdgeVariants] {len(base_circuits)} base -> {n_variants} variants, batch_size={adaptive_batch_size} (est. {bytes_per_circuit/1e6:.1f}MB/circuit)")

    # Batch evaluate all variants at once (with adaptive chunking)
    accs, logit_sims, bit_sims, _ = batch_compute_metrics(
        model, circuits_to_eval, x, y_target, y_pred,
        gate_idx=gate_idx, eval_device=device, max_batch_size=adaptive_batch_size
    )

    # Find best variant for each original circuit
    results = []
    for orig_idx, (start, end, edge_variants) in enumerate(variant_ranges):
        if start == end:
            # No variants (shouldn't happen)
            results.append((orig_idx, base_circuits[orig_idx], 0.0, 0.0, 0.0))
            continue

        # Find best by (accuracy DESC, bit_similarity DESC, edge_sparsity DESC)
        best_idx = start
        best_score = (-accs[start], -bit_sims[start], -edge_variants[0].sparsity()[1])

        for i in range(start + 1, end):
            var_idx = i - start
            score = (-accs[i], -bit_sims[i], -edge_variants[var_idx].sparsity()[1])
            if score < best_score:
                best_score = score
                best_idx = i

        var_idx = best_idx - start
        results.append((
            orig_idx, edge_variants[var_idx],
            float(accs[best_idx]), float(logit_sims[best_idx]), float(bit_sims[best_idx])
        ))

    return results
