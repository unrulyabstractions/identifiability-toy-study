"""Main visualization entry point.

Contains the main visualize_experiment function that orchestrates all visualizations.
"""

import os
from pathlib import Path

from src.circuit import Circuit
from src.infra import (
    get_memory_mb,
    log_memory,
    parse_subcircuit_key,
    run_parallel,
    timed_phase,
)
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


def _generate_circuit_diagrams(result: ExperimentResult, run_dir: str | Path) -> None:
    """Generate circuit diagrams for all subcircuits at run level.

    Creates:
        subcircuits/circuit_diagrams/
            node_masks/
                {node_mask_idx}.png - Diagram for each node mask (with full edges)
            edge_masks/
                {edge_mask_idx}.png - Diagram showing edge mask variations
            subcircuit_masks/
                {subcircuit_idx}.png - Diagram for each subcircuit
    """
    # Get circuits from first trial
    first_trial = next(iter(result.trials.values()), None)
    if not first_trial or not first_trial.subcircuits:
        return

    diagrams_dir = Path(run_dir) / "subcircuits" / "circuit_diagrams"

    # Create all three diagram folders
    node_masks_dir = diagrams_dir / "node_masks"
    edge_masks_dir = diagrams_dir / "edge_masks"
    subcircuit_masks_dir = diagrams_dir / "subcircuit_masks"

    node_masks_dir.mkdir(parents=True, exist_ok=True)
    edge_masks_dir.mkdir(parents=True, exist_ok=True)
    subcircuit_masks_dir.mkdir(parents=True, exist_ok=True)

    # Track unique node patterns (by their serialized form)
    seen_node_patterns = {}  # node_pattern_tuple -> node_mask_idx
    node_mask_idx_counter = 0

    max_diagrams = min(48, len(first_trial.subcircuits))
    print(f"[VIZ] Generating circuit diagrams for {max_diagrams} subcircuits...")

    for sc_data in first_trial.subcircuits[:max_diagrams]:
        subcircuit_idx = sc_data.get("idx", 0)
        circuit = Circuit.from_dict(sc_data)

        # Generate subcircuit_masks/{subcircuit_idx}.png
        sc_path = subcircuit_masks_dir / f"{subcircuit_idx}.png"
        try:
            circuit.visualize(file_path=str(sc_path), node_size="small")
        except Exception as e:
            print(f"  Warning: Failed to generate subcircuit diagram {subcircuit_idx}: {e}")

        # Track unique node patterns for node_masks/
        node_masks = sc_data.get("node_masks", [])
        # Serialize node pattern (exclude input/output which are always full)
        hidden_masks = tuple(tuple(m) for m in node_masks[1:-1]) if len(node_masks) > 2 else ()
        if hidden_masks not in seen_node_patterns:
            seen_node_patterns[hidden_masks] = node_mask_idx_counter
            # Generate node_masks/{node_mask_idx}.png
            nm_path = node_masks_dir / f"{node_mask_idx_counter}.png"
            try:
                circuit.visualize(file_path=str(nm_path), node_size="small")
            except Exception as e:
                print(f"  Warning: Failed to generate node_mask diagram {node_mask_idx_counter}: {e}")
            node_mask_idx_counter += 1

    # For edge_masks, generate diagrams for a few representative edge configurations
    # Since we use full_edges_only=True, each node pattern has one edge config
    # Generate edge_masks as copies of node_masks (they're equivalent with full edges)
    for node_mask_idx in range(min(node_mask_idx_counter, 20)):
        src = node_masks_dir / f"{node_mask_idx}.png"
        dst = edge_masks_dir / f"{node_mask_idx}.png"
        if src.exists() and not dst.exists():
            import shutil
            try:
                shutil.copy(src, dst)
            except Exception:
                pass

    print(f"  Generated {len(list(subcircuit_masks_dir.glob('*.png')))} subcircuit diagrams")
    print(f"  Generated {len(list(node_masks_dir.glob('*.png')))} node_mask diagrams")
    print(f"  Generated {len(list(edge_masks_dir.glob('*.png')))} edge_mask diagrams")


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

    # Generate circuit diagrams at run level
    with timed_phase("Circuit Diagrams"):
        _generate_circuit_diagrams(result, run_dir)

    viz_paths = {}

    for trial_id, trial in result.trials.items():
        subcircuits = [Circuit.from_dict(s) for s in trial.subcircuits]
        viz_paths[trial_id] = {}
        trial_dir = os.path.join(run_dir, "trials", trial_id)

        # --- profiling/ (timing visualizations) - run in parallel ---
        if trial.profiling and trial.profiling.events:
            profiling_dir = os.path.join(trial_dir, "profiling")
            os.makedirs(profiling_dir, exist_ok=True)
            viz_paths[trial_id]["profiling"] = {}

            timeline_path, phases_path, summary_path = run_parallel(
                lambda: visualize_profiling_timeline(trial.profiling, profiling_dir),
                lambda: visualize_profiling_phases(trial.profiling, profiling_dir),
                lambda: visualize_profiling_summary(trial.profiling, profiling_dir),
            )

            if timeline_path:
                viz_paths[trial_id]["profiling"]["timeline"] = timeline_path
            if phases_path:
                viz_paths[trial_id]["profiling"]["phases"] = phases_path
            if summary_path:
                viz_paths[trial_id]["profiling"]["summary"] = summary_path

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
        with timed_phase("Activation Visualizations"):
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

                # Save gate summary.json
                gate_folder = os.path.join(trial_dir, gname)
                summary_path = save_gate_summary(gname, gate_folder, trial.metrics)
                viz_paths[trial_id][gname]["summary"] = summary_path

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
                node_idx, edge_var_idx = parse_subcircuit_key(sc_key)
                node_pattern_edges.setdefault(node_idx, []).append((edge_var_idx, i, faith))

            for i, sc_key in enumerate(best_keys):
                # Convert list to tuple for hashability (JSON loads as list)
                if isinstance(sc_key, list):
                    sc_key = tuple(sc_key)
                node_idx, edge_var_idx = parse_subcircuit_key(sc_key)
                circuit = subcircuits[node_idx]

                # Build folder path and label based on key type
                if isinstance(sc_key, (tuple, list)):
                    folder = os.path.join(trial_dir, gname, str(node_idx), str(edge_var_idx))
                    sc_label = f"{gname} (Node#{node_idx}/Edge#{edge_var_idx})"
                else:
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
                    with timed_phase("Observational Curves"):
                        viz_paths[trial_id][gname][sc_key]["faithfulness"]["observational"]["stats"] = (
                            visualize_observational_curves(
                                observational_data, observational_dir, sc_label
                            )
                        )

                # Run expensive circuit visualizations sequentially (they use ProcessPoolExecutor internally)
                if has_observational and observational_dir:
                    with timed_phase("Observational Circuit Viz"):
                        circuit_paths = visualize_observational_circuits(
                            observational_data, circuit, layer_weights, observational_dir,
                            canonical_activations=canonical_activations,
                            layer_biases=layer_biases if layer_biases else None,
                        )
                    viz_paths[trial_id][gname][sc_key]["faithfulness"]["observational"]["circuit_viz"] = (
                        circuit_paths
                    )

                if has_faith:
                    with timed_phase("Faithfulness Circuit Viz"):
                        circuit_paths = visualize_faithfulness_circuit_samples(
                            faithfulness_data, circuit, layer_weights, faithfulness_dir,
                            layer_biases=layer_biases if layer_biases else None,
                        )
                    viz_paths[trial_id][gname][sc_key]["faithfulness"][
                        "circuit_viz"
                    ] = circuit_paths

                    # Add intervention effect plots (like noise_by_input for robustness)
                    with timed_phase("Intervention Effects"):
                        intervention_paths = visualize_faithfulness_intervention_effects(
                            faithfulness_data, faithfulness_dir, sc_label
                        )
                    viz_paths[trial_id][gname][sc_key]["faithfulness"][
                        "intervention_effects"
                    ] = intervention_paths

                # Save result.json files in each subfolder and summary.json in faithfulness/
                interventional_dir = os.path.join(faithfulness_dir, "interventional") if has_faith else None
                counterfactual_dir = os.path.join(faithfulness_dir, "counterfactual") if has_faith else None
                # Create directories if they don't exist
                if interventional_dir:
                    os.makedirs(interventional_dir, exist_ok=True)
                if counterfactual_dir:
                    os.makedirs(counterfactual_dir, exist_ok=True)
                json_paths = save_faithfulness_json(
                    observational_dir=observational_dir,
                    interventional_dir=interventional_dir,
                    counterfactual_dir=counterfactual_dir,
                    faithfulness_dir=faithfulness_dir,
                    faithfulness=faithfulness_data,
                )
                viz_paths[trial_id][gname][sc_key]["faithfulness"]["json"] = json_paths

                # Save summary.json and structured samples in nested folders
                if has_faith:
                    summary_path = save_summary(folder, faithfulness_data, sc_key)
                    samples_paths = save_all_samples(folder, faithfulness_data, sc_key)
                    viz_paths[trial_id][gname][sc_key]["summary"] = summary_path
                    viz_paths[trial_id][gname][sc_key]["samples"] = samples_paths

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
