from typing import Any

import torch

from .causal_analysis import (
    calculate_faithfulness_metrics,
    calculate_robustness_metrics,
    create_clean_corrupted_data,
    filter_subcircuits,
)
from .common.batched_eval import batch_compute_metrics, precompute_circuit_masks
from .common.circuit import enumerate_all_valid_circuit
from .common.helpers import calculate_match_rate, train_model, update_status_fx
from .common.logic_gates import ALL_LOGIC_GATES
from .common.schemas import (
    GateMetrics,
    ParallelConfig,
    SubcircuitMetrics,
    TrialData,
    TrialResult,
    TrialSetup,
)
from .common.utils import set_seeds
from .parallelization import ParallelTasks
from .parameter_decomposition import decompose_mlp
from .profiler import profile


def _get_eval_device(parallel_config: ParallelConfig, default_device: str) -> str:
    """Determine the device to use for batched circuit evaluation."""
    if parallel_config.use_mps_if_available and parallel_config.eval_device == "mps":
        if torch.backends.mps.is_available():
            return "mps"
    return parallel_config.eval_device if parallel_config.eval_device != "mps" else default_device


def run_trial(
    setup: TrialSetup,
    data: TrialData,
    device: str,
    spd_device: str = "cpu",
    logger: Any = None,
    debug: bool = False,
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
    if model is None:
        return trial_result
    trial_result.model = model
    gate_models = model.separate_into_k_mlps()
    update_status("ENDED_MLP_TRAINING")

    x = data.test.x.contiguous()
    y_gt = data.test.y.contiguous()

    # Use inference_mode for all forward passes (faster than no_grad)
    with torch.inference_mode():
        y_pred = model(data.test.x)
        activations = model(data.test.x, return_activations=True)

        # Store test data and activations for persistence
        trial_result.test_x = x
        trial_result.test_y = y_gt
        trial_result.test_y_pred = y_pred.detach().clone()
        trial_result.activations = [a.clone() for a in activations]

        # Pre-compute canonical activations for visualization (no model runs needed later)
        canonical_inputs = {
            "0_0": torch.tensor([[0.0, 0.0]], device=device),
            "0_1": torch.tensor([[0.0, 1.0]], device=device),
            "1_0": torch.tensor([[1.0, 0.0]], device=device),
            "1_1": torch.tensor([[1.0, 1.0]], device=device),
        }
        trial_result.canonical_activations = {
            label: [a.clone() for a in model(inp, return_activations=True)]
            for label, inp in canonical_inputs.items()
        }

    # Store layer weights for visualization
    trial_result.layer_weights = [
        layer[0].weight.detach().cpu() for layer in model.layers
    ]

    bit_gt = torch.round(y_gt)
    bit_pred = torch.round(y_pred)

    trial_metrics.avg_loss = avg_loss
    trial_metrics.val_acc = val_acc
    trial_metrics.test_acc = calculate_match_rate(torch.round(y_pred), y_gt).item()

    # ===== SPD =====
    spd_config = setup.spd_config
    update_status("STARTED_SPD")
    with profile("spd_mlp"):
        trial_result.decomposed_model = decompose_mlp(x, y_pred, model, spd_device, spd_config)
    update_status("FINISHED_SPD")

    # ===== Circuit Finding =====
    parallel_config = setup.parallel_config
    update_status("STARTED_CIRCUITS")
    with profile("circuit_enum"):
        subcircuits = enumerate_all_valid_circuit(model, use_tqdm=True)

    # Structure analysis is CPU-bound, safe to parallelize
    with profile("structure_analysis"):
        if parallel_config.enable_parallel_structure:
            with ParallelTasks(max_workers=parallel_config.max_workers_structure) as tasks:
                futures = [tasks.submit(s.analyze_structure) for s in subcircuits]
            subcircuit_structures = [f.result() for f in futures]
        else:
            subcircuit_structures = [s.analyze_structure() for s in subcircuits]
    update_status("FINISHED_CIRCUITS")

    trial_result.subcircuits = [
        {"idx": idx, **s.to_dict()} for idx, s in enumerate(subcircuits)
    ]
    trial_result.subcircuit_structure_analysis = [
        {"idx": idx, **s.to_dict()} for idx, s in enumerate(subcircuit_structures)
    ]
    update_status("FINISHED_CIRCUIT_FINDING")

    # Determine evaluation device (may differ from training device)
    eval_device = _get_eval_device(parallel_config, device)

    # Precompute circuit masks once for all gates (significant speedup on GPU)
    # This avoids repeated tensor creation overhead per gate
    precomputed_masks_per_gate = {}
    if parallel_config.precompute_masks:
        for gate_idx in range(len(gate_names)):
            precomputed_masks_per_gate[gate_idx] = precompute_circuit_masks(
                subcircuits, len(model.layers), gate_idx=gate_idx, device=eval_device
            )

    # Gate subcircuit metrics
    def run_gate_trial(gate_idx: int) -> None:
        gate_name = gate_names[gate_idx]
        gate_model = gate_models[gate_idx]
        y_gate = y_pred[..., [gate_idx]]
        bit_gate_gt = bit_gt[..., [gate_idx]]
        bit_gate = bit_pred[..., [gate_idx]]

        # ===== Calculate Gate Metrics (Batched) =====
        # Use batched GPU evaluation for all subcircuits at once
        # This is 6-12x faster than sequential evaluation
        update_status(f"STARTED_GATE_METRICS:{gate_idx}")
        gate_acc = calculate_match_rate(bit_gate, bit_gate_gt).item()

        # Move data to eval device if different from training device
        x_eval = x.to(eval_device) if eval_device != device else x
        y_gate_eval = y_gate.to(eval_device) if eval_device != device else y_gate
        bit_gate_gt_eval = bit_gate_gt.to(eval_device) if eval_device != device else bit_gate_gt

        with profile("gate_metrics"):
            precomputed = precomputed_masks_per_gate.get(gate_idx) if parallel_config.precompute_masks else None
            accuracies, logit_sims, bit_sims = batch_compute_metrics(
                model=model,
                circuits=subcircuits,
                x=x_eval,
                y_target=bit_gate_gt_eval,
                y_pred=y_gate_eval,
                gate_idx=gate_idx,
                precomputed_masks=precomputed,
                eval_device=eval_device,
            )

        subcircuit_metrics = [
            SubcircuitMetrics(
                idx=idx,
                accuracy=float(accuracies[idx]),
                logit_similarity=float(logit_sims[idx]),
                bit_similarity=float(bit_sims[idx]),
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
        )

        best_indices = per_gate_bests[gate_name]

        # Create subcircuit models only for best subcircuits (saves memory)
        best_subcircuit_models = {
            idx: gate_model.separate_subcircuit(subcircuits[idx], gate_idx=gate_idx)
            for idx in best_indices
        }

        # ===== Create Counterfactual Pairs (quick, needed by faithfulness) =====
        counterfactual_pairs = create_clean_corrupted_data(
            x=x,
            y=y_gate,
            activations=activations,
            n_pairs=10,
        )

        # ===== Robustness + Faithfulness Analysis =====
        # NOTE: GPU operations are not thread-safe by default.
        # enable_parallel_* flags allow experimentation but may cause issues.
        # See: https://github.com/pytorch/pytorch/issues/103793
        update_status(f"STARTED_ROBUSTNESS:{gate_idx}")

        def compute_single_robustness(subcircuit_idx):
            subcircuit_model = best_subcircuit_models[subcircuit_idx]
            return calculate_robustness_metrics(
                subcircuit=subcircuit_model,
                full_model=gate_model,
                n_samples_per_base=100,
                device=device,
            )

        if parallel_config.enable_parallel_robustness and len(best_indices) > 1:
            with ParallelTasks(max_workers=min(4, len(best_indices))) as tasks:
                futures = [tasks.submit(compute_single_robustness, idx) for idx in best_indices]
            robustness_results = [f.result() for f in futures]
        else:
            robustness_results = []
            for subcircuit_idx in best_indices:
                with profile("calc_robustness"):
                    robustness_results.append(compute_single_robustness(subcircuit_idx))

        per_gate_bests_robust[gate_name].extend(robustness_results)
        update_status(f"FINISHED_ROBUSTNESS:{gate_idx}")

        update_status(f"STARTED_FAITH:{gate_idx}")

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
                config=setup.faithfulness_config,
                device=device,
            )

        if parallel_config.enable_parallel_faithfulness and len(best_indices) > 1:
            with ParallelTasks(max_workers=min(4, len(best_indices))) as tasks:
                futures = [tasks.submit(compute_single_faithfulness, idx) for idx in best_indices]
            faithfulness_results = [f.result() for f in futures]
        else:
            faithfulness_results = []
            for subcircuit_idx in best_indices:
                with profile("calc_faithfulness"):
                    faithfulness_results.append(compute_single_faithfulness(subcircuit_idx))

        per_gate_bests_faith[gate_name].extend(faithfulness_results)
        update_status(f"FINISHED_FAITH:{gate_idx}")

        # ===== Subcircuit Parameter Decomposition =====
        spd_for_subcircuit = False
        if spd_for_subcircuit:
            update_status(f"STARTED_SPD_TRAINING_GATE:{gate_idx}")

            # Collect indices for decomposition
            decompose_indices = per_gate_bests[gate_name]
            if spd_config.max_subcircuits:
                decompose_indices = decompose_indices[: spd_config.max_subcircuits]

            # Run gate model and all subcircuit decompositions in parallel
            with ParallelTasks(max_workers=len(decompose_indices) + 1) as tasks:
                # Submit gate model decomposition
                gate_future = tasks.submit(
                    decompose_mlp, x, y_gate, gate_model, spd_device, spd_config
                )

                # Submit all subcircuit decompositions
                subcircuit_futures = {
                    idx: tasks.submit(
                        decompose_mlp,
                        x,
                        y_gate,
                        best_subcircuit_models[idx],
                        spd_device,
                        spd_config,
                    )
                    for idx in decompose_indices
                }

            # Collect results
            trial_result.decomposed_gate_models[gate_name] = gate_future.result()
            update_status(f"FINISHED_SPD_TRAINING_GATE:{gate_idx}")

            decomposed_dict = trial_result.decomposed_subcircuits[gate_name]
            for subcircuit_idx, future in subcircuit_futures.items():
                decomposed_dict[subcircuit_idx] = future.result()
                trial_result.decomposed_subcircuit_indices[gate_name].append(
                    subcircuit_idx
                )
                update_status(
                    f"FINISHED_SPD_TRAINING_SUBCIRCUIT:{gate_idx}:{subcircuit_idx}"
                )

    update_status("STARTED_GATE_ANALYSIS")
    for gate_idx in range(len(gate_models)):
        run_gate_trial(gate_idx)
    update_status("FINISHED_GATE_ANALYSIS")

    update_status("SUCCESSFUL_TRIAL")
    return trial_result
