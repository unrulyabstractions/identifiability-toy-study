"""Main visualization entry point.

Contains the main visualize_experiment function that orchestrates all visualizations.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.circuit import Circuit
from src.infra import profile
from src.schemas import ExperimentResult
from .activation_viz import (
    visualize_circuit_activations_from_data,
    visualize_circuit_activations_mean,
)
from .constants import _layout_cache
from .faithfulness_viz import (
    visualize_faithfulness_circuit_samples,
    visualize_faithfulness_intervention_effects,
)
from .export import save_faithfulness_json, save_gate_summary
from .observational_viz import (
    visualize_robustness_circuit_samples,
    visualize_robustness_curves,
)
from .profiling_viz import (
    visualize_profiling_phases,
    visualize_profiling_summary,
    visualize_profiling_timeline,
)
from .spd_viz import visualize_spd_components


def visualize_experiment(result: ExperimentResult, run_dir: str | Path) -> dict:
    """
    Generate all visualizations for experiment using pre-computed data.

    IMPORTANT: This function does NOT run any models. All data comes from:
    - trial.canonical_activations: Pre-computed activations for binary inputs
    - trial.layer_weights: Weight matrices from the trained model
    - trial.metrics: Robustness and faithfulness results

    Returns paths dict.
    """
    os.makedirs(run_dir, exist_ok=True)

    viz_paths = {}

    for trial_id, trial in result.trials.items():
        subcircuits = [Circuit.from_dict(s) for s in trial.subcircuits]
        viz_paths[trial_id] = {}
        trial_dir = os.path.join(run_dir, trial_id)

        # --- profiling/ (timing visualizations) - run in parallel ---
        if trial.profiling and trial.profiling.events:
            profiling_dir = os.path.join(trial_dir, "profiling")
            os.makedirs(profiling_dir, exist_ok=True)
            viz_paths[trial_id]["profiling"] = {}

            with ThreadPoolExecutor(max_workers=3) as executor:
                timeline_future = executor.submit(
                    visualize_profiling_timeline, trial.profiling, profiling_dir
                )
                phases_future = executor.submit(
                    visualize_profiling_phases, trial.profiling, profiling_dir
                )
                summary_future = executor.submit(
                    visualize_profiling_summary, trial.profiling, profiling_dir
                )

            if path := timeline_future.result():
                viz_paths[trial_id]["profiling"]["timeline"] = path
            if path := phases_future.result():
                viz_paths[trial_id]["profiling"]["phases"] = path
            if path := summary_future.result():
                viz_paths[trial_id]["profiling"]["summary"] = path

        # Extract pre-computed data
        canonical_activations = trial.canonical_activations or {}
        mean_activations_by_range = trial.mean_activations_by_range or {}
        layer_weights = trial.layer_weights or []
        layer_biases = trial.layer_biases or []
        gate_names = trial.setup.model_params.logic_gates

        if not canonical_activations or not layer_weights:
            continue

        # Full circuit (all edges/nodes active)
        layer_sizes = [layer_weights[0].shape[1]] + [w.shape[0] for w in layer_weights]
        full_circuit = Circuit.full(layer_sizes)

        # Pre-cache layout for this structure
        _layout_cache.get_positions(tuple(layer_sizes))

        # --- all_gates/ (full multi-gate model) ---
        folder = os.path.join(trial_dir, "all_gates")
        os.makedirs(folder, exist_ok=True)
        viz_paths[trial_id]["all_gates"] = {}
        all_gates_label = "All Gates (" + ", ".join(gate_names) + ")"

        act_path = visualize_circuit_activations_from_data(
            canonical_activations,
            layer_weights,
            full_circuit,
            folder,
            gate_name=all_gates_label,
            layer_biases=layer_biases if layer_biases else None,
        )
        viz_paths[trial_id]["all_gates"]["activations"] = act_path

        if trial.decomposed_model:
            if path := visualize_spd_components(
                trial.decomposed_model, folder, gate_name=all_gates_label
            ):
                viz_paths[trial_id]["all_gates"]["spd"] = path

        # --- Per-gate visualization ---
        for gate_idx, gname in enumerate(gate_names):
            folder = os.path.join(trial_dir, gname, "full")
            os.makedirs(folder, exist_ok=True)
            viz_paths[trial_id].setdefault(gname, {})["full"] = {}

            gate_label = f"{gname} (Full)"

            act_path = visualize_circuit_activations_from_data(
                canonical_activations,
                layer_weights,
                full_circuit,
                folder,
                gate_name=gate_label,
                layer_biases=layer_biases if layer_biases else None,
                gate_idx=gate_idx,
            )
            viz_paths[trial_id][gname]["full"]["activations"] = act_path

            # Mean activations for different input ranges
            if mean_activations_by_range:
                mean_act_path = visualize_circuit_activations_mean(
                    mean_activations_by_range,
                    layer_weights,
                    full_circuit,
                    folder,
                    gate_name=gate_label,
                    layer_biases=layer_biases if layer_biases else None,
                    gate_idx=gate_idx,
                )
                viz_paths[trial_id][gname]["full"]["activations_mean"] = mean_act_path

            if gname in trial.decomposed_gate_models:
                decomposed = trial.decomposed_gate_models[gname]
                if path := visualize_spd_components(
                    decomposed, folder, gate_name=gate_label
                ):
                    viz_paths[trial_id][gname]["full"]["spd"] = path

            # Save gate summary.json
            gate_folder = os.path.join(trial_dir, gname)
            summary_path = save_gate_summary(gname, gate_folder, trial.metrics)
            viz_paths[trial_id][gname]["summary"] = summary_path

        # --- Subcircuit visualization ---
        for gate_idx, gname in enumerate(gate_names):
            best_indices = trial.metrics.per_gate_bests.get(gname, [])
            if not best_indices:
                continue

            print(
                f"[VIZ] Gate {gname}: {len(best_indices)} best subcircuits to visualize"
            )
            bests_robust = trial.metrics.per_gate_bests_robust.get(gname, [])
            decomposed_indices = trial.decomposed_subcircuit_indices.get(gname, [])

            for i, sc_idx in enumerate(best_indices):
                circuit = subcircuits[sc_idx]
                folder = os.path.join(trial_dir, gname, str(sc_idx))
                os.makedirs(folder, exist_ok=True)
                viz_paths[trial_id].setdefault(gname, {})[sc_idx] = {}

                sc_label = f"{gname} (SC #{sc_idx})"

                # Static circuit structure
                path = os.path.join(folder, "circuit.png")
                circuit.visualize(file_path=path, node_size="small")
                viz_paths[trial_id][gname][sc_idx]["circuit"] = path

                # Circuit activations
                act_path = visualize_circuit_activations_from_data(
                    canonical_activations,
                    layer_weights,
                    circuit,
                    folder,
                    gate_name=sc_label,
                    layer_biases=layer_biases if layer_biases else None,
                    gate_idx=gate_idx,
                )
                viz_paths[trial_id][gname][sc_idx]["activations"] = act_path

                # Mean activations for different input ranges
                if mean_activations_by_range:
                    mean_act_path = visualize_circuit_activations_mean(
                        mean_activations_by_range,
                        layer_weights,
                        circuit,
                        folder,
                        gate_name=sc_label,
                        layer_biases=layer_biases if layer_biases else None,
                        gate_idx=gate_idx,
                    )
                    viz_paths[trial_id][gname][sc_idx]["activations_mean"] = mean_act_path

                # Robustness and Faithfulness visualization
                bests_faith = trial.metrics.per_gate_bests_faith.get(gname, [])
                has_robust = i < len(bests_robust)
                has_faith = i < len(bests_faith)

                robustness_data = bests_robust[i] if has_robust else None
                faithfulness_data = bests_faith[i] if has_faith else None

                # Create directories upfront
                # Faithfulness contains: observational/ (robustness), counterfactual/, interventional/
                faithfulness_dir = os.path.join(folder, "faithfulness")
                os.makedirs(faithfulness_dir, exist_ok=True)
                viz_paths[trial_id][gname][sc_idx]["faithfulness"] = {}

                # Observational dir (renamed from robustness) - lives inside faithfulness/
                observational_dir = (
                    os.path.join(faithfulness_dir, "observational") if has_robust else None
                )
                if observational_dir:
                    os.makedirs(observational_dir, exist_ok=True)
                    viz_paths[trial_id][gname][sc_idx]["faithfulness"]["observational"] = {}

                # Run quick visualizations sequentially (matplotlib is not thread-safe)
                if has_robust and observational_dir:
                    with profile("robust_curves"):
                        viz_paths[trial_id][gname][sc_idx]["faithfulness"]["observational"]["stats"] = (
                            visualize_robustness_curves(
                                robustness_data, observational_dir, sc_label
                            )
                        )
                # Run expensive circuit visualizations sequentially (they use ProcessPoolExecutor internally)
                if has_robust and observational_dir:
                    with profile("robust_circuit_viz"):
                        circuit_paths = visualize_robustness_circuit_samples(
                            robustness_data, circuit, layer_weights, observational_dir,
                            canonical_activations=canonical_activations,
                            layer_biases=layer_biases if layer_biases else None,
                        )
                    viz_paths[trial_id][gname][sc_idx]["faithfulness"]["observational"]["circuit_viz"] = (
                        circuit_paths
                    )

                if has_faith:
                    with profile("faith_circuit_viz"):
                        circuit_paths = visualize_faithfulness_circuit_samples(
                            faithfulness_data, circuit, layer_weights, faithfulness_dir,
                            layer_biases=layer_biases if layer_biases else None,
                        )
                    viz_paths[trial_id][gname][sc_idx]["faithfulness"][
                        "circuit_viz"
                    ] = circuit_paths

                    # Add intervention effect plots (like noise_by_input for robustness)
                    with profile("faith_intervention_effects"):
                        intervention_paths = visualize_faithfulness_intervention_effects(
                            faithfulness_data, faithfulness_dir, sc_label
                        )
                    viz_paths[trial_id][gname][sc_idx]["faithfulness"][
                        "intervention_effects"
                    ] = intervention_paths

                # Save result.json files in each subfolder and summary.json in faithfulness/
                interventional_dir = os.path.join(faithfulness_dir, "interventional") if has_faith else None
                counterfactual_dir = os.path.join(faithfulness_dir, "counterfactual") if has_faith else None
                json_paths = save_faithfulness_json(
                    observational_dir=observational_dir,
                    interventional_dir=interventional_dir,
                    counterfactual_dir=counterfactual_dir,
                    faithfulness_dir=faithfulness_dir,
                    robustness=robustness_data,
                    faithfulness=faithfulness_data,
                )
                viz_paths[trial_id][gname][sc_idx]["faithfulness"]["json"] = json_paths

                # SPD
                if gname in trial.decomposed_subcircuits:
                    if sc_idx in trial.decomposed_subcircuits[gname]:
                        decomposed = trial.decomposed_subcircuits[gname][sc_idx]
                        if path := visualize_spd_components(
                            decomposed, folder, gate_name=sc_label
                        ):
                            viz_paths[trial_id][gname][sc_idx]["spd"] = path

    return viz_paths
