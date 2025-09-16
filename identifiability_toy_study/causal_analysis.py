from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .common.causal import Intervention
from .common.circuit import (
    Circuit,
)
from .common.neural_model import MLP
from .common.schemas import (
    CircuitMetrics,
    Dataset,
    GateMetrics,
    IdentifiabilityConstraints,
)


@torch.no_grad()
def calculate_match_rate(y_pred, y_gt):
    y_pred = y_pred.reshape(-1)
    y_gt = y_gt.reshape(-1)
    return y_pred.eq(y_gt).float().mean()  # returns 0-D tensor


@torch.no_grad()
def _discrete_pred(logits: torch.Tensor) -> torch.Tensor:
    """
    Binary: threshold at 0 (or use round if you prefer).
    Multiclass: argmax.
    Returns int tensor of shape [B].
    """
    if logits.ndim == 2 and logits.size(1) > 1:
        return logits.argmax(dim=-1)
    return (logits > 0).long().view(-1)


@torch.no_grad()
def _top_margin(logits: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
    """
    Scalar task metric per sample.
    - Multiclass: y known -> logit[y] - max(other); y unknown -> top1 - top2.
    - Binary (1 logit): use absolute logit magnitude.
    """
    if logits.ndim == 2 and logits.size(1) > 1:
        if y is not None:
            top_y = logits.gather(1, y.view(-1, 1)).squeeze(1)
            masked = logits.clone()
            masked.scatter_(1, y.view(-1, 1), float("-inf"))
            top2 = masked.max(dim=1).values
            return top_y - top2
        else:
            top2_vals, _ = torch.topk(logits, k=2, dim=1)
            return top2_vals[:, 0] - top2_vals[:, 1]
    else:
        return logits.view(-1).abs()


@torch.no_grad()
def _active_vars_from_circuit(circuit: "Circuit", model: "MLP") -> dict[int, list[int]]:
    """
    Returns {layer_index -> [neuron indices]} for hidden layers that are active in the circuit.
    We assume node_masks has length model.num_layers+1 with index 0=input, L=logits.
    We take hidden layers 1..L-1.
    """
    vars_by_layer: dict[int, list[int]] = {}
    L = model.num_layers
    for ℓ in range(1, L):  # hidden only
        mask = circuit.node_masks[ℓ]
        if torch.is_tensor(mask):
            mask = mask.detach().cpu().numpy()
        idxs = [i for i, v in enumerate(mask) if int(v) == 1]
        if idxs:
            vars_by_layer[ℓ] = idxs
    return vars_by_layer


# ------------------ Level-1 ↔ Level-2: commutation & circuit error ------------


@torch.no_grad()
def compute_circuit_error_and_commutation(
    model: "MLP",
    circuit: "Circuit",
    x: torch.Tensor,
    *,
    atol: float = 0.0,
) -> tuple[float, bool, float]:
    """
    Circuit error (discrete mismatch rate) and commutation gap (mean abs logit diff).
    CE(S) = 1 - mean( 1[ pred_L(x) == pred_S(x) ] )
    Commutes if mean |g_L(x) - g_S(x)| <= atol.
    """
    logits_L = model(x)  # g_L(x)
    logits_S = model(x, circuit=circuit)  # g_S(x)
    ce = (
        1.0
        - (_discrete_pred(logits_L) == _discrete_pred(logits_S)).float().mean().item()
    )
    gap = (logits_L - logits_S).abs().mean().item()
    return ce, bool(gap <= atol), gap


# ------------------ Level-2: faithfulness via edge-wise patching --------------


@torch.no_grad()
def _forward_edge_patched(
    model: "MLP",
    x_clean: torch.Tensor,
    x_corr: torch.Tensor,
    circuit: "Circuit",
) -> torch.Tensor:
    """
    Faithfulness forward:
      z = (W◦M) @ h_clean + (W◦(1-M)) @ h_corr + b,
    with node masks applied to the clean stream before (W◦M) @ h_clean.
    """
    device = next(model.parameters()).device
    x_clean, x_corr = x_clean.to(device), x_corr.to(device)

    # Precompute corrupted hidden activations
    h_corr = x_corr
    corr_hiddens: list[torch.Tensor] = []
    for i, layer in enumerate(model.layers):
        lin, act = layer[0], layer[1]
        zc = F.linear(h_corr, lin.weight, lin.bias)
        if i < (model.num_layers - 1):
            h_corr = act(zc)
            corr_hiddens.append(h_corr)
        else:
            h_corr = zc  # logits

    # Patched forward on clean input
    h = x_clean
    for i, layer in enumerate(model.layers):
        lin, act = layer[0], layer[1]
        W, b = lin.weight, lin.bias
        M = torch.as_tensor(circuit.edge_masks[i], dtype=W.dtype, device=W.device)
        invM = 1.0 - M

        # node mask at current activation h^{(i)} (indexing: 0=input, …)
        if getattr(circuit, "node_masks", None) is not None:
            nm = torch.as_tensor(circuit.node_masks[i], dtype=h.dtype, device=h.device)
            h = h * nm

        h_corr_prev = x_corr if i == 0 else corr_hiddens[i - 1]
        z = F.linear(h, W * M, b) + F.linear(h_corr_prev, W * invM, None)
        h = act(z) if i < (model.num_layers - 1) else z
    return h


@torch.no_grad()
def compute_faithfulness(
    model: "MLP",
    circuit: "Circuit",
    x: torch.Tensor,
    *,
    y: torch.Tensor | None = None,
    normalize: bool = True,
) -> float:
    """
    Returns a normalized faithfulness score in ~[0,1], higher is better.
    Metric = top-margin (w.r.t. y if available; otherwise top1-top2 or |logit|).
    """
    device = next(model.parameters()).device
    x = x.to(device)
    B = x.size(0)
    perm = torch.randperm(B, device=device)

    clean_logits = model(x)
    patched_logits = _forward_edge_patched(model, x, x[perm], circuit)

    m_clean = _top_margin(clean_logits, y.to(device) if y is not None else None)
    m_patch = _top_margin(patched_logits, y.to(device) if y is not None else None)

    if normalize:
        eps = 1e-8
        score = (m_patch.abs() + eps) / (m_clean.abs() + eps)
    else:
        score = 1.0 - (m_clean - m_patch).abs()

    return score.mean().item()


# ------------------ Level-2: IIA with subcircuit-as-algorithm -----------------


@torch.no_grad()
def compute_iia_subcircuit_as_algorithm(
    model: "MLP",
    circuit: "Circuit",
    x: torch.Tensor,
    *,
    num_pairs: int = 128,
    max_vars_per_layer: Optional[int] = None,
) -> float:
    """
    IIA where the 'algorithm' is the subcircuit S.
    Variables U = (ℓ, j) for active neurons in S (hidden layers 1..L-1).

      τ_U(x_s) = h_S^{(ℓ)}(x_s)[j]               (pulled via return_activations=True)
      ω(U←u)   = neuron column patch at (ℓ,j)     (applied via Intervention)

    We compare discrete predictions of:
      - low-level L: y_L = model(x_b, intervention=IV)
      - subcircuit S: y_S = model(x_b, circuit=circuit, intervention=IV)

    where IV = Intervention.from_states_dict({(ℓ,(j,)): τ_U(x_s)}).
    """
    device = next(model.parameters()).device
    x = x.to(device)
    B = x.size(0)

    # choose (base, source) pairs
    idx_base = torch.randint(B, (num_pairs,), device=device)
    idx_src = torch.randint(B, (num_pairs,), device=device)
    xb, xs = x[idx_base], x[idx_src]

    # τ: activations under the subcircuit S on the source batch
    # acts_S_src: [0=input, 1..L-1=hidden activations, L=logits]
    acts_S_src = model(xs, circuit=circuit, return_activations=True)

    # collect active variables U = (layer, neuron) from the circuit's node mask
    vars_by_layer: dict[int, list[int]] = {}
    L = model.num_layers
    for ℓ in range(1, L):  # hidden layers only
        mask = circuit.node_masks[ℓ]
        if torch.is_tensor(mask):
            mask = mask.detach().cpu().numpy()
        js = [j for j, v in enumerate(mask) if int(v) == 1]
        if js:
            vars_by_layer[ℓ] = js

    # optional sub-sampling per layer for speed
    if max_vars_per_layer is not None:
        for ℓ in list(vars_by_layer.keys()):
            js = vars_by_layer[ℓ]
            if len(js) > max_vars_per_layer:
                perm = torch.randperm(len(js), device=device)[:max_vars_per_layer]
                vars_by_layer[ℓ] = [js[i.item()] for i in perm]

    def _discrete_pred(logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim == 2 and logits.size(1) > 1:
            return logits.argmax(dim=-1)
        return (logits > 0).long().view(-1)

    matches = []
    for ℓ, js in vars_by_layer.items():
        if not js:
            continue
        for j in js:
            # τ_U(xs): take the neuron column from S-run activations on sources, shape [P,1]
            tau_vals = acts_S_src[ℓ][:, j].unsqueeze(-1)

            # Build a first-class Intervention (neuron patch at (ℓ,(j,)))
            iv = Intervention.from_states_dict(
                {(ℓ, (j,)): tau_vals}, mode="set", axis="neuron"
            )

            # Apply the SAME intervention to low-level and to the subcircuit
            logits_L = model(xb, intervention=iv)  # low-level
            logits_S = model(
                xb, circuit=circuit, intervention=iv
            )  # subcircuit-as-algorithm

            pred_L = _discrete_pred(logits_L)
            pred_S = _discrete_pred(logits_S)
            matches.append((pred_L == pred_S).float().mean())

    return torch.stack(matches).mean().item() if matches else 0.0


# ------------------ MAIN FUNCTION -----------------


def calculate_subcircuit_metrics(
    x: torch.Tensor,
    model: MLP,
    circuit: Circuit,
    y_gt: torch.Tensor,
    y_full: torch.Tensor,
) -> CircuitMetrics:
    bit_gt = torch.round(y_gt)
    bit_full = torch.round(y_full)

    # Make predictions with the current circuit
    y_circuit = model(x, circuit=circuit)
    bit_circuit = torch.round(y_circuit)

    # Compute the accuracy with respect to the task
    accuracy = calculate_match_rate(bit_circuit, bit_gt).item()
    logit_similarity = 1 - nn.MSELoss()(y_full, y_circuit).item()
    bit_similarity = calculate_match_rate(bit_circuit, bit_full).item()

    # Compute circuit sparsity
    node_spar, edge_spar, combined_spar = circuit.sparsity()

    # ---- Level-1 ↔ Level-2 analyses (commutation, faithfulness, IIA) ----
    ce, commutes, comm_gap = compute_circuit_error_and_commutation(
        model, circuit, x, atol=0.0
    )
    faith = compute_faithfulness(
        model,
        circuit,
        x,
        y=y_gt,
        normalize=True,
    )
    iia = compute_iia_subcircuit_as_algorithm(
        model,
        circuit,
        x,
        num_pairs=128,
        max_vars_per_layer=None,
    )

    return CircuitMetrics(
        accuracy=accuracy,
        logit_similarity=logit_similarity,
        bit_similarity=bit_similarity,
        sparsity=(node_spar, edge_spar, combined_spar),
        commutes=commutes,
        comm_gap=comm_gap,
        faithfulness=faith,
        iia=iia,
    )


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

    # Ground Truth
    y_gt = dataset.y
    bit_gt = torch.round(y_gt)

    # Full circuit, Model
    y_full = model(x)
    bit_full = torch.round(y_full)

    test_acc = calculate_match_rate(bit_full, bit_gt).item()

    per_circuit = {}
    faithful_circuits_idx = []

    # Iterate over all circuits with progress tracking
    it = enumerate(circuits)
    if use_tqdm:
        it = tqdm(
            it, total=num_total_circuits, desc="Filtering circuits by constraints"
        )
    for i, subcircuit in it:
        result = calculate_subcircuit_metrics(
            x=x, y_gt=y_gt, y_full=y_full, model=model, circuit=subcircuit
        )
        per_circuit[i] = result

        if result.sparsity[0] < constraints.min_sparsity:
            continue

        if result.bit_similarity < constraints.acc_threshold:
            continue

        print(f"Subcircuit {i} {result}")

        # optional gating
        if constraints.require_commutation and not result.commutes:
            continue
        if result.faithfulness < constraints.faithfulness_min:
            continue
        if result.iia < constraints.iia_min:
            continue

        faithful_circuits_idx.append(i)

    return GateMetrics(
        test_acc=test_acc,
        num_total_circuits=num_total_circuits,
        per_circuit=per_circuit,
        faithful_circuits_idx=faithful_circuits_idx,
    )
