from typing import Any

import torch

from .causal import (
    calculate_faithfulness_metrics,
    calculate_robustness_metrics,
    create_clean_corrupted_data,
    filter_subcircuits,
)
from .common.batched_eval import (
    adapt_masks_for_gate,
    batch_compute_metrics,
    batch_evaluate_edge_variants,
    precompute_circuit_masks_base,
)
from .common.circuit import enumerate_all_valid_circuit
from .common.helpers import calculate_match_rate, train_model, update_status_fx
from .common.logic_gates import ALL_LOGIC_GATES
from .common.parallelization import ParallelTasks, get_eval_device
from .common.profiler import profile, profile_fn
from .common.schemas import (
    GateMetrics,
    ParallelConfig,
    SubcircuitMetrics,
    TrialData,
    TrialResult,
    TrialSetup,
)
from .common.utils import set_seeds
from .spd_internal.decomposition import decompose_mlp
from .spd_internal.subcircuits import estimate_spd_subcircuits




@profile_fn("Train Model")
def _train_model_phase(setup, data, device, logger, debug, input_size, output_size):
    """Train the MLP model."""
    model, avg_loss, val_acc = train_model(
        train_params=setup.train_params,
        model_params=setup.model_params,
        data=data,
        device=device,
        logger=logger,
        debug=debug,
        input_size=input_size,
        output_size=output_size,
    )
    return model, avg_loss, val_acc


@profile_fn("Compute Activations")
def _compute_activations_phase(model, data, device, input_size):
    """Compute and store activations for test data."""
    x = data.test.x.contiguous()
    y_gt = data.test.y.contiguous()

    with torch.inference_mode():
        y_pred = model(data.test.x)
        activations = model(data.test.x, return_activations=True)

        # Pre-compute canonical activations for visualization
        canonical_inputs = {
            "0_0": torch.tensor([[0.0, 0.0]], device=device),
            "0_1": torch.tensor([[0.0, 1.0]], device=device),
            "1_0": torch.tensor([[1.0, 0.0]], device=device),
            "1_1": torch.tensor([[1.0, 1.0]], device=device),
        }
        canonical_activations = {
            label: [a.clone() for a in model(inp, return_activations=True)]
            for label, inp in canonical_inputs.items()
        }

        # Pre-compute mean activations for different input ranges
        n_samples = 100
        input_ranges = {
            "0_1": (0.0, 1.0),
            "-1_0": (-1.0, 0.0),
            "-2_2": (-2.0, 2.0),
            "-100_100": (-100.0, 100.0),
        }
        mean_activations_by_range = {}
        for label, (low, high) in input_ranges.items():
            random_inputs = (
                torch.rand(n_samples, input_size, device=device) * (high - low) + low
            )
            all_activations = model(random_inputs, return_activations=True)
            mean_activations = [
                a.mean(dim=0, keepdim=True).clone() for a in all_activations
            ]
            mean_activations_by_range[label] = mean_activations

    # Store layer weights and biases
    layer_weights = [layer[0].weight.detach().cpu() for layer in model.layers]
    layer_biases = [layer[0].bias.detach().cpu() for layer in model.layers]

    return {
        "x": x,
        "y_gt": y_gt,
        "y_pred": y_pred,
        "activations": [a.clone() for a in activations],
        "canonical_activations": canonical_activations,
        "mean_activations_by_range": mean_activations_by_range,
        "layer_weights": layer_weights,
        "layer_biases": layer_biases,
    }


@profile_fn("SPD Decomposition")
def _spd_phase(
    setup, trial_result, model, x, y_pred, spd_device, input_size, gate_names
):
    """Run SPD decomposition for all configs."""
    spd_configs_to_run = [setup.spd_config]
    if setup.spd_sweep_configs:
        spd_configs_to_run.extend(setup.spd_sweep_configs)

    for config_idx, spd_config in enumerate(spd_configs_to_run):
        config_id = spd_config.get_config_id()
        print(f"    SPD config {config_idx + 1}/{len(spd_configs_to_run)}: {config_id}")

        with profile(f"spd_mlp_{config_id}"):
            decomposed = decompose_mlp(x, y_pred, model, spd_device, spd_config)

        if config_idx == 0:
            trial_result.decomposed_model = decomposed
        trial_result.decomposed_models_sweep[config_id] = decomposed

    # SPD Subcircuit estimation
    for config_idx, spd_config in enumerate(spd_configs_to_run):
        config_id = spd_config.get_config_id()
        decomposed = trial_result.decomposed_models_sweep[config_id]
        print(
            f"    SPD subcircuit {config_idx + 1}/{len(spd_configs_to_run)}: {config_id}"
        )

        with profile(f"spd_mlp_sc_{config_id}"):
            estimate = estimate_spd_subcircuits(
                decomposed_model=decomposed,
                target_model=model,
                n_inputs=input_size,
                gate_names=gate_names,
                device=spd_device,
            )

        if config_idx == 0:
            trial_result.spd_subcircuit_estimate = estimate
        trial_result.spd_subcircuit_estimates_sweep[config_id] = estimate


@profile_fn("Enumerate Circuits")
def _enumerate_circuits_phase(model, parallel_config):
    """Enumerate all valid circuits and analyze structure."""
    with profile("circuit_enum"):
        subcircuits = enumerate_all_valid_circuit(model, use_tqdm=True)

    with profile("structure_analysis"):
        if parallel_config.enable_parallel_structure:
            with ParallelTasks(
                max_workers=parallel_config.max_workers_structure
            ) as tasks:
                futures = [tasks.submit(s.analyze_structure) for s in subcircuits]
            subcircuit_structures = [f.result() for f in futures]
        else:
            subcircuit_structures = [s.analyze_structure() for s in subcircuits]

    return subcircuits, subcircuit_structures


@profile_fn("Precompute Circuit Masks")
def _precompute_masks_phase(subcircuits, model, gate_names, eval_device, output_size):
    """Precompute circuit masks for all gates."""
    precomputed_base_masks = precompute_circuit_masks_base(
        subcircuits, len(model.layers), device=eval_device
    )
    precomputed_masks_per_gate = {}
    for gate_idx in range(len(gate_names)):
        precomputed_masks_per_gate[gate_idx] = adapt_masks_for_gate(
            precomputed_base_masks, gate_idx, output_size
        )
    return precomputed_masks_per_gate


@profile_fn("Gate Metrics (Batched)")
def _gate_metrics_phase(
    model,
    subcircuits,
    x_eval,
    bit_gate_gt_eval,
    y_gate_eval,
    gate_idx,
    precomputed,
    eval_device,
):
    """Compute metrics for all subcircuits for a gate."""
    accuracies, logit_sims, bit_sims, best_sims = batch_compute_metrics(
        model=model,
        circuits=subcircuits,
        x=x_eval,
        y_target=bit_gate_gt_eval,
        y_pred=y_gate_eval,
        gate_idx=gate_idx,
        precomputed_masks=precomputed,
        eval_device=eval_device,
    )
    return accuracies, logit_sims, bit_sims, best_sims


@profile_fn("Edge Variants")
def _edge_variants_phase(
    model,
    best_circuits_for_gate,
    x_eval,
    bit_gate_gt_eval,
    y_gate_eval,
    gate_idx,
    eval_device,
):
    """Evaluate edge variants for best circuits."""
    return batch_evaluate_edge_variants(
        model=model,
        base_circuits=best_circuits_for_gate,
        x=x_eval,
        y_target=bit_gate_gt_eval,
        y_pred=y_gate_eval,
        gate_idx=gate_idx,
        eval_device=eval_device,
    )


@profile_fn("Robustness Analysis")
def _robustness_phase(
    best_indices, best_subcircuit_models, gate_model, device, parallel_config
):
    """Compute robustness metrics for best subcircuits."""

    def compute_single_robustness(subcircuit_idx):
        subcircuit_model = best_subcircuit_models[subcircuit_idx]
        return calculate_robustness_metrics(
            subcircuit=subcircuit_model,
            full_model=gate_model,
            n_samples_per_base=200,
            device=device,
        )

    if parallel_config.enable_parallel_robustness and len(best_indices) > 1:
        with ParallelTasks(max_workers=min(4, len(best_indices))) as tasks:
            futures = [
                tasks.submit(compute_single_robustness, idx) for idx in best_indices
            ]
        return [f.result() for f in futures]
    else:
        results = []
        for subcircuit_idx in best_indices:
            with profile("calc_robustness"):
                results.append(compute_single_robustness(subcircuit_idx))
        return results


@profile_fn("Faithfulness Analysis")
def _faithfulness_phase(
    best_indices,
    best_subcircuit_models,
    gate_model,
    x,
    y_gate,
    activations,
    subcircuit_structures,
    counterfactual_pairs,
    config,
    device,
    parallel_config,
):
    """Compute faithfulness metrics for best subcircuits."""

    def compute_single_faithfulness(subcircuit_idx):
        subcircuit_model = best_subcircuit_models[subcircuit_idx]
        return calculate_faithfulness_metrics(
            x=x,
            y=y_gate,
            model=gate_model,
            activations=activations,
            subcircuit=subcircuit_model,
            structure=subcircuit_structures[subcircuit_idx],
            counterfactual_pairs=counterfactual_pairs,
            config=config,
            device=device,
        )

    if parallel_config.enable_parallel_faithfulness and len(best_indices) > 1:
        with ParallelTasks(max_workers=min(4, len(best_indices))) as tasks:
            futures = [
                tasks.submit(compute_single_faithfulness, idx) for idx in best_indices
            ]
        return [f.result() for f in futures]
    else:
        results = []
        for subcircuit_idx in best_indices:
            with profile("calc_faithfulness"):
                results.append(compute_single_faithfulness(subcircuit_idx))
        return results


def run_trial(
    setup: TrialSetup,
    data: TrialData,
    device: str,
    spd_device: str = "cpu",
    logger: Any = None,
    debug: bool = False,
    run_spd: bool = False,
) -> TrialResult:
    # This also sets deterministic behavior
    set_seeds(setup.seed)

    trial_result = TrialResult(
        setup=setup,
    )

    trial_metrics = trial_result.metrics
    per_gate_metrics = trial_metrics.per_gate_metrics
    per_gate_bests = trial_metrics.per_gate_bests
    per_gate_bests_robust = trial_metrics.per_gate_bests_robust
    per_gate_bests_faith = trial_metrics.per_gate_bests_faith

    update_status = update_status_fx(trial_result, logger, device=device)

    # ===== Train Model =====
    update_status("STARTED_MLP_TRAINING")
    gate_names = setup.model_params.logic_gates
    input_size = ALL_LOGIC_GATES[gate_names[0]].n_inputs
    output_size = len(gate_names)

    model, avg_loss, val_acc = _train_model_phase(
        setup, data, device, logger, debug, input_size, output_size
    )
    if model is None:
        return trial_result
    trial_result.model = model
    gate_models = model.separate_into_k_mlps()
    update_status("ENDED_MLP_TRAINING")

    # ===== Compute Activations =====
    act_data = _compute_activations_phase(model, data, device, input_size)
    x = act_data["x"]
    y_gt = act_data["y_gt"]
    y_pred = act_data["y_pred"]
    activations = act_data["activations"]

    trial_result.test_x = x
    trial_result.test_y = y_gt
    trial_result.test_y_pred = y_pred.detach().clone()
    trial_result.activations = activations
    trial_result.canonical_activations = act_data["canonical_activations"]
    trial_result.mean_activations_by_range = act_data["mean_activations_by_range"]
    trial_result.layer_weights = act_data["layer_weights"]
    trial_result.layer_biases = act_data["layer_biases"]

    bit_gt = torch.round(y_gt)
    bit_pred = torch.round(y_pred)

    trial_metrics.avg_loss = avg_loss
    trial_metrics.val_acc = val_acc
    trial_metrics.test_acc = calculate_match_rate(torch.round(y_pred), y_gt).item()

    # ===== SPD (if enabled) =====
    if run_spd:
        update_status("STARTED_SPD")
        _spd_phase(
            setup, trial_result, model, x, y_pred, spd_device, input_size, gate_names
        )
        update_status("FINISHED_SPD")

    # ===== Circuit Finding =====
    parallel_config = setup.parallel_config
    update_status("STARTED_CIRCUITS")
    subcircuits, subcircuit_structures = _enumerate_circuits_phase(
        model, parallel_config
    )
    update_status("FINISHED_CIRCUITS")

    trial_result.subcircuits = [
        {"idx": idx, **s.to_dict()} for idx, s in enumerate(subcircuits)
    ]
    trial_result.subcircuit_structure_analysis = [
        {"idx": idx, **s.to_dict()} for idx, s in enumerate(subcircuit_structures)
    ]
    update_status("FINISHED_CIRCUIT_FINDING")

    # Determine evaluation device
    eval_device = _get_eval_device(parallel_config, device)

    # Precompute circuit masks
    precomputed_masks_per_gate = {}
    if parallel_config.precompute_masks:
        precomputed_masks_per_gate = _precompute_masks_phase(
            subcircuits, model, gate_names, eval_device, output_size
        )

    # ===== Gate Analysis =====
    update_status("STARTED_GATE_ANALYSIS")
    for gate_idx in range(len(gate_models)):
        gate_name = gate_names[gate_idx]
        gate_model = gate_models[gate_idx]
        y_gate = y_pred[..., [gate_idx]]
        bit_gate_gt = bit_gt[..., [gate_idx]]
        bit_gate = bit_pred[..., [gate_idx]]

        print(f"\n{'~' * 60}")
        print(f"  Gate {gate_idx}: {gate_name}")
        print("~" * 60)

        update_status(f"STARTED_GATE_METRICS:{gate_idx}")
        gate_acc = calculate_match_rate(bit_gate, bit_gate_gt).item()

        # Move data to eval device
        x_eval = x.to(eval_device) if eval_device != device else x
        y_gate_eval = y_gate.to(eval_device) if eval_device != device else y_gate
        bit_gate_gt_eval = (
            bit_gate_gt.to(eval_device) if eval_device != device else bit_gate_gt
        )

        precomputed = (
            precomputed_masks_per_gate.get(gate_idx)
            if parallel_config.precompute_masks
            else None
        )

        accuracies, logit_sims, bit_sims, best_sims = _gate_metrics_phase(
            model,
            subcircuits,
            x_eval,
            bit_gate_gt_eval,
            y_gate_eval,
            gate_idx,
            precomputed,
            eval_device,
        )

        subcircuit_metrics = [
            SubcircuitMetrics(
                idx=idx,
                accuracy=float(accuracies[idx]),
                logit_similarity=float(logit_sims[idx]),
                bit_similarity=float(bit_sims[idx]),
                best_similarity=float(best_sims[idx]),
            )
            for idx in range(len(subcircuits))
        ]

        per_gate_metrics[gate_name] = GateMetrics(
            test_acc=gate_acc,
            subcircuit_metrics=subcircuit_metrics,
        )

        per_gate_bests[gate_name] = filter_subcircuits(
            setup.constraints,
            per_gate_metrics[gate_name].subcircuit_metrics,
            subcircuits,
            subcircuit_structures,
            max_subcircuits=setup.faithfulness_config.max_subcircuits_per_gate,
        )

        best_indices = per_gate_bests[gate_name]
        best_circuits_for_gate = [subcircuits[idx] for idx in best_indices]

        edge_results = _edge_variants_phase(
            model,
            best_circuits_for_gate,
            x_eval,
            bit_gate_gt_eval,
            y_gate_eval,
            gate_idx,
            eval_device,
        )

        optimized_circuits = {
            best_indices[orig_idx]: opt_circuit
            for orig_idx, opt_circuit, _, _, _ in edge_results
        }

        best_subcircuit_models = {
            idx: gate_model.separate_subcircuit(
                optimized_circuits[idx], gate_idx=gate_idx
            )
            for idx in best_indices
        }

        counterfactual_pairs = create_clean_corrupted_data(
            x=x,
            y=y_gate,
            activations=activations,
            n_pairs=setup.faithfulness_config.n_counterfactual_pairs,
        )

        update_status(f"STARTED_ROBUSTNESS:{gate_idx}")
        robustness_results = _robustness_phase(
            best_indices, best_subcircuit_models, gate_model, device, parallel_config
        )
        per_gate_bests_robust[gate_name].extend(robustness_results)
        update_status(f"FINISHED_ROBUSTNESS:{gate_idx}")

        update_status(f"STARTED_FAITH:{gate_idx}")
        faithfulness_results = _faithfulness_phase(
            best_indices,
            best_subcircuit_models,
            gate_model,
            x,
            y_gate,
            activations,
            subcircuit_structures,
            counterfactual_pairs,
            setup.faithfulness_config,
            device,
            parallel_config,
        )
        per_gate_bests_faith[gate_name].extend(faithfulness_results)
        update_status(f"FINISHED_FAITH:{gate_idx}")

    update_status("FINISHED_GATE_ANALYSIS")
    update_status("SUCCESSFUL_TRIAL")
    return trial_result
