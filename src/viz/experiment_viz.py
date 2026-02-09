"""Main visualization entry point.

Contains the main visualize_experiment function that orchestrates all visualizations.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.circuit import Circuit
from src.infra import profile, get_memory_mb, log_memory
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
from .export import (
    save_all_samples,
    save_faithfulness_json,
    save_gate_summary,
    save_node_pattern_summary,
    save_summary,
)
from .observational_viz import (
    visualize_observational_circuits,
    visualize_observational_curves,
)
from .profiling_viz import (
    visualize_profiling_phases,
    visualize_profiling_summary,
    visualize_profiling_timeline,
)


def _print_viz_phase(phase_name: str, mem_before: float, is_start: bool = True):
    """Print visualization phase header or footer."""
    if is_start:
        print(f"\n{'=' * 60}")
        print(f"  [VIZ] {phase_name}")
        print(f"  Memory before: {mem_before:.1f} MB")
        print("=" * 60)


def _print_viz_phase_end(phase_name: str, elapsed_ms: float, mem_before: float):
    """Print visualization phase completion."""
    mem_after = get_memory_mb()
    mem_delta = mem_after - mem_before
    print(f"  -> Completed in {elapsed_ms:.0f}ms")
    print(f"  -> Memory after: {mem_after:.1f} MB (delta: {mem_delta:+.1f} MB)")
    log_memory(f"after_viz_{phase_name.lower().replace(' ', '_')}")


def visualize_experiment(result: ExperimentResult, run_dir: str | Path) -> dict:
    """
    Generate all visualizations for experiment using pre-computed data.

    IMPORTANT: This function does NOT run any models. All data comes from:
    - trial.canonical_activations: Pre-computed activations for binary inputs
    - trial.layer_weights: Weight matrices from the trained model
    - trial.metrics: Robustness and faithfulness results

    Returns paths dict.
    """
    import time

    # Overall viz profiling
    viz_start = time.time()
    viz_mem_start = get_memory_mb()
    print(f"\n{'~' * 60}")
    print(f"  VISUALIZATION PHASE")
    print(f"  Memory before: {viz_mem_start:.1f} MB")
    print(f"{'~' * 60}")

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
        _mem_before = get_memory_mb()
        _t0 = time.time()
        _print_viz_phase("Activation Visualizations", _mem_before)

        folder = os.path.join(trial_dir, "all_gates")
        os.makedirs(folder, exist_ok=True)
        viz_paths[trial_id]["all_gates"] = {}
        all_gates_label = "All Gates (" + ", ".join(gate_names) + ")"

        with profile("viz_activations"):
            act_path = visualize_circuit_activations_from_data(
                canonical_activations,
                layer_weights,
                full_circuit,
                folder,
                gate_name=all_gates_label,
                layer_biases=layer_biases if layer_biases else None,
            )
        viz_paths[trial_id]["all_gates"]["activations"] = act_path

        # --- Per-gate visualization ---
        for gate_idx, gname in enumerate(gate_names):
            folder = os.path.join(trial_dir, gname, "full")
            os.makedirs(folder, exist_ok=True)
            viz_paths[trial_id].setdefault(gname, {})["full"] = {}

            gate_label = f"{gname} (Full)"

            with profile("viz_activations"):
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
                with profile("viz_activations"):
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

            # Save gate summary.json
            gate_folder = os.path.join(trial_dir, gname)
            summary_path = save_gate_summary(gname, gate_folder, trial.metrics)
            viz_paths[trial_id][gname]["summary"] = summary_path

        _print_viz_phase_end("activations", (time.time() - _t0) * 1000, _mem_before)

        # --- Subcircuit visualization ---
        for gate_idx, gname in enumerate(gate_names):
            best_keys = trial.metrics.per_gate_bests.get(gname, [])
            if not best_keys:
                continue

            print(
                f"[VIZ] Gate {gname}: {len(best_keys)} best subcircuits to visualize"
            )
            bests_faith = trial.metrics.per_gate_bests_faith.get(gname, [])

            # Group edge variations by node pattern for summary generation
            node_pattern_edges: dict[int, list[tuple[int, int, "FaithfulnessMetrics"]]] = {}
            for i, sc_key in enumerate(best_keys):
                faith = bests_faith[i] if i < len(bests_faith) else None
                if isinstance(sc_key, tuple):
                    node_idx, edge_var_idx = sc_key
                else:
                    node_idx, edge_var_idx = sc_key, 0
                node_pattern_edges.setdefault(node_idx, []).append((edge_var_idx, i, faith))

            for i, sc_key in enumerate(best_keys):
                # Handle both legacy int keys and new (node_idx, edge_var_idx) tuple keys
                if isinstance(sc_key, tuple):
                    node_idx, edge_var_idx = sc_key
                    circuit = subcircuits[node_idx]  # Use node pattern's circuit structure
                    folder = os.path.join(trial_dir, gname, str(node_idx), str(edge_var_idx))
                    sc_label = f"{gname} (Node#{node_idx}/Edge#{edge_var_idx})"
                else:
                    node_idx = sc_key
                    circuit = subcircuits[sc_key]
                    folder = os.path.join(trial_dir, gname, str(sc_key))
                    sc_label = f"{gname} (SC #{sc_key})"

                os.makedirs(folder, exist_ok=True)
                viz_paths[trial_id].setdefault(gname, {})[sc_key] = {}

                # Static circuit structure
                path = os.path.join(folder, "circuit.png")
                circuit.visualize(file_path=path, node_size="small")
                viz_paths[trial_id][gname][sc_key]["circuit"] = path

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
                viz_paths[trial_id][gname][sc_key]["activations"] = act_path

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
                    viz_paths[trial_id][gname][sc_key]["activations_mean"] = mean_act_path

                # Robustness and Faithfulness visualization
                # Robustness is now inside faithfulness.observational
                has_faith = i < len(bests_faith)
                faithfulness_data = bests_faith[i] if has_faith else None
                observational_data = faithfulness_data.observational if faithfulness_data else None
                has_observational = observational_data is not None

                # Create directories upfront
                # Faithfulness contains: observational/ (robustness), counterfactual/, interventional/
                faithfulness_dir = os.path.join(folder, "faithfulness")
                os.makedirs(faithfulness_dir, exist_ok=True)
                viz_paths[trial_id][gname][sc_key]["faithfulness"] = {}

                # Observational dir (renamed from robustness) - lives inside faithfulness/
                observational_dir = (
                    os.path.join(faithfulness_dir, "observational") if has_observational else None
                )
                if observational_dir:
                    os.makedirs(observational_dir, exist_ok=True)
                    viz_paths[trial_id][gname][sc_key]["faithfulness"]["observational"] = {}

                # Run quick visualizations sequentially (matplotlib is not thread-safe)
                if has_observational and observational_dir:
                    _mem_before = get_memory_mb()
                    _t0 = time.time()
                    _print_viz_phase("Observational Curves", _mem_before)
                    with profile("viz_observational_curves"):
                        viz_paths[trial_id][gname][sc_key]["faithfulness"]["observational"]["stats"] = (
                            visualize_observational_curves(
                                observational_data, observational_dir, sc_label
                            )
                        )
                    _print_viz_phase_end("observational_curves", (time.time() - _t0) * 1000, _mem_before)

                # Run expensive circuit visualizations sequentially (they use ProcessPoolExecutor internally)
                if has_observational and observational_dir:
                    _mem_before = get_memory_mb()
                    _t0 = time.time()
                    _print_viz_phase("Observational Circuit Viz", _mem_before)
                    with profile("viz_observational_circuit"):
                        circuit_paths = visualize_observational_circuits(
                            observational_data, circuit, layer_weights, observational_dir,
                            canonical_activations=canonical_activations,
                            layer_biases=layer_biases if layer_biases else None,
                        )
                    _print_viz_phase_end("observational_circuit", (time.time() - _t0) * 1000, _mem_before)
                    viz_paths[trial_id][gname][sc_key]["faithfulness"]["observational"]["circuit_viz"] = (
                        circuit_paths
                    )

                if has_faith:
                    _mem_before = get_memory_mb()
                    _t0 = time.time()
                    _print_viz_phase("Faithfulness Circuit Viz", _mem_before)
                    with profile("viz_faith_circuit"):
                        circuit_paths = visualize_faithfulness_circuit_samples(
                            faithfulness_data, circuit, layer_weights, faithfulness_dir,
                            layer_biases=layer_biases if layer_biases else None,
                        )
                    _print_viz_phase_end("faith_circuit", (time.time() - _t0) * 1000, _mem_before)
                    viz_paths[trial_id][gname][sc_key]["faithfulness"][
                        "circuit_viz"
                    ] = circuit_paths

                    # Add intervention effect plots (like noise_by_input for robustness)
                    _mem_before = get_memory_mb()
                    _t0 = time.time()
                    _print_viz_phase("Intervention Effects", _mem_before)
                    with profile("viz_intervention_effects"):
                        intervention_paths = visualize_faithfulness_intervention_effects(
                            faithfulness_data, faithfulness_dir, sc_label
                        )
                    _print_viz_phase_end("intervention_effects", (time.time() - _t0) * 1000, _mem_before)
                    viz_paths[trial_id][gname][sc_key]["faithfulness"][
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
                    faithfulness=faithfulness_data,
                )
                viz_paths[trial_id][gname][sc_key]["faithfulness"]["json"] = json_paths

                # Save summary.json and samples.json in this leaf folder
                if has_faith:
                    summary_path = save_summary(folder, faithfulness_data, sc_key)
                    samples_path = save_all_samples(folder, faithfulness_data, sc_key)
                    viz_paths[trial_id][gname][sc_key]["summary"] = summary_path
                    viz_paths[trial_id][gname][sc_key]["samples"] = samples_path

            # After processing all edge variations, save node pattern summaries
            for node_idx, edge_list in node_pattern_edges.items():
                node_dir = os.path.join(trial_dir, gname, str(node_idx))
                edge_variations = [(edge_var_idx, faith) for edge_var_idx, _, faith in edge_list]
                save_node_pattern_summary(node_idx, node_dir, edge_variations)

    # Final viz profiling summary
    viz_elapsed_ms = (time.time() - viz_start) * 1000
    viz_mem_end = get_memory_mb()
    viz_mem_delta = viz_mem_end - viz_mem_start
    log_memory("after_visualization")
    print(f"\n{'~' * 60}")
    print(f"  VISUALIZATION COMPLETE")
    print(f"  Total time: {viz_elapsed_ms:.0f}ms")
    print(f"  Memory after: {viz_mem_end:.1f} MB (delta: {viz_mem_delta:+.1f} MB)")
    print(f"{'~' * 60}")

    return viz_paths
