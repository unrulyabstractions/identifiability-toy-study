"""Main visualization entry point.

Contains the main visualize_experiment function that orchestrates all visualizations.

IMPORTANT: NO PyTorch model inference should happen during visualization.
All data must be pre-computed during trial execution (in run_trial/decision_boundary_phase).
The @no_pytorch decorator enforces this by blocking torch operations.
"""

import functools
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from src.circuit import Circuit
from src.circuit.enumeration import (
    count_valid_subcircuits,
    get_edge_list,
    get_num_edges,
)
from src.domain import resolve_gate

if TYPE_CHECKING:
    from src.schemas import FaithfulnessMetrics


def no_pytorch_inference(func):
    """Decorator that detects PyTorch model inference during function execution.

    This ensures visualization functions don't accidentally run model inference.
    All data should be pre-computed in trial execution phases.

    Note: This allows read-only torch operations (type checks, tensor inspection)
    but blocks nn.Module forward passes.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        import torch.nn as nn

        # Store original forward methods
        original_module_call = nn.Module.__call__

        def blocked_forward(self, *args, **kwargs):
            raise RuntimeError(
                f"PyTorch model inference detected: {type(self).__name__}.__call__(). "
                "Visualization must use pre-computed data only. "
                "Move model inference to trial execution phase."
            )

        # Block nn.Module forward passes
        nn.Module.__call__ = blocked_forward
        try:
            return func(*args, **kwargs)
        finally:
            nn.Module.__call__ = original_module_call

    return wrapper


from src.infra import (
    get_memory_mb,
    log_memory,
    parse_subcircuit_key,
    run_parallel,
    timed_phase,
)
from src.schemas import ExperimentResult
from src.visualization.decision_boundary import plot_decision_boundary_from_data

from .activation_viz import (
    visualize_circuit_activations_from_data,
    visualize_circuit_activations_mean,
)
from .constants import _layout_cache
from .export import (
    save_all_samples,
    save_faithfulness_json,
    save_gate_summary,
    save_node_pattern_summary,
    save_summary,
)
from .faithfulness_viz import (
    visualize_faithfulness_circuit_samples,
    visualize_faithfulness_intervention_effects,
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
from .slice_viz import visualize_slice
from .viz_config import VizConfig, VizLevel

from src.analysis.slices import (
    ObservationalSlice,
    InterventionalSlice,
    CounterfactualSlice,
)


def _plot_theoretical_decision_boundary(gate, output_path: str) -> None:
    """Plot the theoretical (ground truth) decision boundary for a gate.

    Uses the same colormap and styling as plot_decision_boundary_2d_from_data
    for visual consistency.

    Args:
        gate: LogicGate object with n_inputs and gate_fn
        output_path: Path to save the PNG
    """
    import matplotlib.pyplot as plt
    import numpy as np

    n_inputs = gate.n_inputs
    if n_inputs > 2:
        return  # Only support 1D and 2D for now

    # Match styling from decision_boundary.py
    low, high = -3.0, 3.0
    resolution = 100

    if n_inputs == 1:
        fig, ax = plt.subplots(figsize=(8, 5))
        # 1D: plot gate output vs input
        x = np.linspace(low, high, resolution).astype(np.float32)
        # gate_fn expects 2D array (n_samples, n_inputs)
        inputs = x.reshape(-1, 1)
        y = gate.gate_fn(inputs).flatten().astype(np.float32)
        ax.plot(x, y, color="steelblue", linewidth=2)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Decision boundary")
        ax.axvline(0.5, color="red", linestyle=":", alpha=0.5, label="Input threshold")

        # Mark binary corners
        corners = np.array([[0], [1]], dtype=np.float32)
        corner_preds = gate.gate_fn(corners).flatten()
        for cx, cy in zip([0, 1], corner_preds):
            color = "blue" if cy < 0.5 else "red"
            ax.scatter([cx], [cy], c=color, s=150, marker="s", edgecolors="black", zorder=5)
            ax.annotate(
                f"({int(cx)})\n{cy:.2f}",
                (cx, cy),
                textcoords="offset points",
                xytext=(10, 10),
                fontsize=9,
            )

        ax.set_xlabel("Input")
        ax.set_ylabel("P(output=1)")
        ax.set_title(f"{gate.name}: Theoretical Decision Boundary (1D)")
        ax.set_xlim(low, high)
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        fig, ax = plt.subplots(figsize=(8, 7))
        # 2D: plot decision regions with same colormap as decision_boundary.py
        x = np.linspace(low, high, resolution).astype(np.float32)
        y = np.linspace(low, high, resolution).astype(np.float32)
        xx, yy = np.meshgrid(x, y, indexing="ij")

        # Evaluate gate at each point
        # gate_fn expects shape (n_samples, n_inputs) - each row is a sample
        inputs = np.stack([xx.flatten(), yy.flatten()], axis=1)
        zz = gate.gate_fn(inputs).reshape(xx.shape).astype(np.float32)

        # Plot filled contours - match decision_boundary.py styling
        cf = ax.contourf(xx, yy, zz, levels=50, cmap="RdYlBu_r", alpha=0.9)

        # Mark binary corner points
        corners = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
        corner_preds = gate.gate_fn(corners).flatten()
        for i, (cx, cy) in enumerate(corners):
            pred = corner_preds[i]
            color = "blue" if pred < 0.5 else "red"
            ax.scatter([cx], [cy], c=color, s=200, marker="s", edgecolors="black", zorder=5)
            ax.annotate(
                f"({int(cx)},{int(cy)})\n{pred:.2f}",
                (cx, cy),
                textcoords="offset points",
                xytext=(10, 10),
                fontsize=8,
            )

        plt.colorbar(cf, ax=ax, label="P(output=1)")
        ax.set_xlabel("Input 1")
        ax.set_ylabel("Input 2")
        ax.set_title(f"{gate.name}: Theoretical Decision Boundary (2D)")
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _generate_circuit_diagrams(result: ExperimentResult, run_dir: str | Path) -> None:
    """Generate circuit diagrams showing individual edges and node masks.

    Creates:
        subcircuits/circuit_diagrams/
            edges/
                {edge_idx}.png - Diagram highlighting single edge
            node_masks/
                {node_mask_idx}.png - Diagram for each node mask pattern
            full_circuit.png - Full circuit with all edges
            README.txt - Explanation of subcircuit indexing

    A subcircuit index is the decimal interpretation of a binary edge mask.
    Edge i is active iff bit i of the subcircuit index is 1.

    Example: subcircuit_idx=13 (binary 1101) means edges 0, 2, 3 are active.
    """
    import numpy as np

    # Get architecture from config
    cfg = result.config
    width = cfg.base_trial.model_params.width
    depth = cfg.base_trial.model_params.depth
    gate_names = cfg.base_trial.model_params.logic_gates
    input_size = max(resolve_gate(g).n_inputs for g in gate_names)

    layer_widths = [input_size] + [width] * depth + [1]

    diagrams_dir = Path(run_dir) / "subcircuits" / "circuit_diagrams"
    edges_dir = diagrams_dir / "edges"
    node_masks_dir = diagrams_dir / "node_masks"
    edges_dir.mkdir(parents=True, exist_ok=True)
    node_masks_dir.mkdir(parents=True, exist_ok=True)

    # Get edge list and count valid subcircuits
    edges = get_edge_list(input_size, width, depth)
    num_edges = len(edges)
    num_valid = count_valid_subcircuits(input_size, width, depth)

    print(f"[VIZ] Generating edge diagrams for architecture {layer_widths}...")
    print(f"  {num_edges} edges, {num_valid:,} valid subcircuits")

    # Generate full circuit diagram (all edges active)
    full_node_masks = [np.ones(w, dtype=np.int8) for w in layer_widths]
    full_edge_masks = [
        np.ones((layer_widths[i + 1], layer_widths[i]), dtype=np.int8)
        for i in range(len(layer_widths) - 1)
    ]
    full_circuit = Circuit(node_masks=full_node_masks, edge_masks=full_edge_masks)
    full_path = diagrams_dir / "full_circuit.png"
    try:
        full_circuit.visualize(file_path=str(full_path), node_size="small")
    except Exception as e:
        print(f"  Warning: Failed to generate full_circuit.png: {e}")

    # Generate individual edge diagrams
    for edge_idx, (src_layer, src_idx, dst_layer, dst_idx) in enumerate(edges):
        # Create circuit with only this edge active
        node_masks = [np.ones(w, dtype=np.int8) for w in layer_widths]
        edge_masks = [
            np.zeros((layer_widths[i + 1], layer_widths[i]), dtype=np.int8)
            for i in range(len(layer_widths) - 1)
        ]
        # Activate just this edge
        layer_transition = src_layer  # edge_masks index = source layer
        edge_masks[layer_transition][dst_idx, src_idx] = 1

        circuit = Circuit(node_masks=node_masks, edge_masks=edge_masks)
        edge_path = edges_dir / f"{edge_idx}.png"
        try:
            circuit.visualize(file_path=str(edge_path), node_size="small")
        except Exception as e:
            print(f"  Warning: Failed to generate edge {edge_idx}.png: {e}")

    # Generate node mask diagrams
    # Enumerate circuits to get node patterns
    from src.circuit import enumerate_circuits_for_architecture
    node_mask_circuits = enumerate_circuits_for_architecture(layer_widths, use_tqdm=False)
    num_node_masks = len(node_mask_circuits)
    print(f"  Generating {num_node_masks} node mask diagrams...")

    for node_mask_idx, circuit in enumerate(node_mask_circuits):
        node_mask_path = node_masks_dir / f"{node_mask_idx}.png"
        try:
            circuit.visualize(file_path=str(node_mask_path), node_size="small")
        except Exception as e:
            print(f"  Warning: Failed to generate node_mask {node_mask_idx}.png: {e}")

    # Write README explaining the indexing
    readme_path = diagrams_dir / "README.txt"
    readme_content = f"""Subcircuit Indexing
===================

Architecture: {layer_widths}
Total edges: {num_edges}
Node masks: {num_node_masks}
Valid subcircuits: {num_valid:,}

EDGE INDEXING
-------------
Each edge has an index from 0 to {num_edges - 1}.
See edges/{{idx}}.png for visualization of each edge.

Edge list (src_layer, src_node, dst_layer, dst_node):
"""
    for idx, (sl, si, dl, di) in enumerate(edges):
        readme_content += f"  Edge {idx:2d}: L{sl}[{si}] -> L{dl}[{di}]\n"

    readme_content += f"""
NODE MASK INDEXING
------------------
Each node mask pattern has an index from 0 to {num_node_masks - 1}.
See node_masks/{{idx}}.png for visualization of each pattern.

A node mask determines which hidden nodes are active (included in the subcircuit).
Different edge variants can share the same node mask.

SUBCIRCUIT INDEX
----------------
A subcircuit is identified by a decimal index.
The binary representation encodes which edges are active:
  - Bit i = 1 means edge i is active
  - Bit i = 0 means edge i is inactive

Example: subcircuit_idx = 13
  Binary: 1101 (reading right-to-left: bits 0,2,3 are set)
  Active edges: 0, 2, 3

The {num_valid:,} valid subcircuits are those where:
  - All input nodes have >= 1 outgoing edge
  - The output node has >= 1 incoming edge
  - Each hidden node is either absent (no edges) or valid (>= 1 in AND >= 1 out)
"""
    readme_path.write_text(readme_content)

    print(f"  Generated {num_edges} edge diagrams + {num_node_masks} node mask diagrams + README")


def save_per_gate_data(
    result: ExperimentResult,
    run_dir: str | Path,
) -> dict:
    """Save per-gate JSON data (always runs, independent of viz level).

    Creates per-gate folders and saves:
    - {gate}/summary.json - Gate-level summary
    - {gate}/node{X}_edge{Y}/summary.json - Subcircuit summary
    - {gate}/node{X}_edge{Y}/faithfulness/*.json - Faithfulness data
    - {gate}/node{X}/summary.json - Node pattern summary

    Args:
        result: ExperimentResult with metrics
        run_dir: Output directory

    Returns:
        Dict of saved paths
    """
    print("[DATA] Saving per-gate JSON data...")
    os.makedirs(run_dir, exist_ok=True)

    saved_paths = {}

    for trial_id, trial in result.trials.items():
        saved_paths[trial_id] = {}
        trial_dir = os.path.join(run_dir, "trials", trial_id)

        gate_names = trial.setup.model_params.logic_gates
        width = trial.setup.model_params.width
        depth = trial.setup.model_params.depth

        for gate_idx, gname in enumerate(gate_names):
            gate_folder = os.path.join(trial_dir, gname)
            os.makedirs(gate_folder, exist_ok=True)
            saved_paths[trial_id].setdefault(gname, {})

            # Save gate summary.json
            summary_path = save_gate_summary(gname, gate_folder, trial.metrics, width, depth)
            saved_paths[trial_id][gname]["summary"] = summary_path

            # Process subcircuits
            best_keys = trial.metrics.per_gate_bests.get(gname, [])
            if not best_keys:
                continue

            bests_faith = trial.metrics.per_gate_bests_faith.get(gname, [])

            # Group edge variations by node pattern for summary generation
            node_pattern_edges: dict[
                int, list[tuple[int, int, "FaithfulnessMetrics"]]
            ] = {}
            for i, sc_key in enumerate(best_keys):
                faith = bests_faith[i] if i < len(bests_faith) else None
                node_mask_idx, edge_variant_rank = parse_subcircuit_key(sc_key, width, depth)
                node_pattern_edges.setdefault(node_mask_idx, []).append(
                    (edge_variant_rank, i, faith)
                )

            # Save per-subcircuit data
            for i, sc_key in enumerate(best_keys):
                node_mask_idx, edge_variant_rank = parse_subcircuit_key(sc_key, width, depth)

                # Use node{X}/{rank}_{sc_key}/ structure
                folder = os.path.join(
                    trial_dir, gname, f"node{node_mask_idx}", f"{i}_{sc_key}"
                )
                os.makedirs(folder, exist_ok=True)
                saved_paths[trial_id][gname][sc_key] = {}

                has_faith = i < len(bests_faith)
                faithfulness_data = bests_faith[i] if has_faith else None
                observational_data = (
                    faithfulness_data.observational if faithfulness_data else None
                )
                has_observational = observational_data is not None

                # Create faithfulness directory structure
                faithfulness_dir = os.path.join(folder, "faithfulness")
                os.makedirs(faithfulness_dir, exist_ok=True)
                saved_paths[trial_id][gname][sc_key]["faithfulness"] = {}

                observational_dir = (
                    os.path.join(faithfulness_dir, "observational")
                    if has_observational
                    else None
                )
                if observational_dir:
                    os.makedirs(observational_dir, exist_ok=True)

                interventional_dir = (
                    os.path.join(faithfulness_dir, "interventional")
                    if has_faith
                    else None
                )
                counterfactual_dir = (
                    os.path.join(faithfulness_dir, "counterfactual")
                    if has_faith
                    else None
                )
                if interventional_dir:
                    os.makedirs(interventional_dir, exist_ok=True)
                if counterfactual_dir:
                    os.makedirs(counterfactual_dir, exist_ok=True)

                # Save faithfulness JSON
                json_paths = save_faithfulness_json(
                    observational_dir=observational_dir,
                    interventional_dir=interventional_dir,
                    counterfactual_dir=counterfactual_dir,
                    faithfulness_dir=faithfulness_dir,
                    faithfulness=faithfulness_data,
                )
                saved_paths[trial_id][gname][sc_key]["faithfulness"]["json"] = json_paths

                # Save summary.json and samples
                if has_faith:
                    summary_path = save_summary(folder, faithfulness_data, sc_key)
                    samples_paths = save_all_samples(faithfulness_dir, faithfulness_data, sc_key)
                    saved_paths[trial_id][gname][sc_key]["summary"] = summary_path
                    saved_paths[trial_id][gname][sc_key]["samples"] = samples_paths

            # Save node pattern summaries
            for node_mask_idx, edge_list in node_pattern_edges.items():
                node_dir = os.path.join(trial_dir, gname, f"node{node_mask_idx}")
                edge_variations = [
                    (edge_variant_rank, faith) for edge_variant_rank, _, faith in edge_list
                ]
                save_node_pattern_summary(node_mask_idx, node_dir, edge_variations)

    print("[DATA] Per-gate JSON data saved.")
    return saved_paths


@no_pytorch_inference
def visualize_experiment(
    result: ExperimentResult,
    run_dir: str | Path,
    viz_config: VizConfig | None = None,
) -> dict:
    """
    Generate all visualizations for experiment using pre-computed data.

    IMPORTANT: This function does NOT run any models. All data comes from:
    - trial.canonical_activations: Pre-computed activations for binary inputs
    - trial.layer_weights: Weight matrices from the trained model
    - trial.metrics: Robustness and faithfulness results
    - trial.decision_boundary_data: Pre-computed decision boundary grid data
    - trial.subcircuit_decision_boundary_data: Pre-computed subcircuit boundary data

    The @no_pytorch decorator enforces that no model inference happens here.

    Args:
        result: ExperimentResult to visualize
        run_dir: Output directory
        viz_config: Visualization configuration (controls level of detail)

    Returns paths dict.
    """
    import time

    if viz_config is None:
        viz_config = VizConfig()

    os.makedirs(run_dir, exist_ok=True)

    # Skip PNG visualization if level is NONE
    # Note: save_per_gate_data is called separately from pipeline
    if viz_config.skip_all_viz:
        print("[VIZ] Skipping PNG visualization (--viz 0)")
        return {}

    saved_paths = {}

    # Overall viz profiling
    viz_start = time.time()
    viz_mem_start = get_memory_mb()
    print(f"\n{'~' * 60}")
    print(f"  VISUALIZATION PHASE (level: {viz_config.level.name})")
    print(f"  Memory before: {viz_mem_start:.1f} MB")
    print(f"{'~' * 60}")

    # Generate circuit diagrams at run level
    with timed_phase("Circuit Diagrams"):
        _generate_circuit_diagrams(result, run_dir)

    viz_paths = saved_paths  # Merge with saved paths

    for trial_id, trial in result.trials.items():
        # Load subcircuits from trial, or regenerate from architecture if empty
        if trial.subcircuits:
            subcircuits = [Circuit.from_dict(s) for s in trial.subcircuits]
        else:
            # Regenerate subcircuits from architecture (for iterative mode)
            layer_weights = trial.layer_weights or []
            if layer_weights:
                layer_sizes = [layer_weights[0].shape[1]] + [w.shape[0] for w in layer_weights]
                from src.circuit import enumerate_circuits_for_architecture
                subcircuits = enumerate_circuits_for_architecture(layer_sizes, use_tqdm=False)
                print(f"[VIZ] Regenerated {len(subcircuits)} subcircuits from architecture")
            else:
                subcircuits = []
        viz_paths.setdefault(trial_id, {})
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
        width = trial.setup.model_params.width
        depth = trial.setup.model_params.depth

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

                # Static circuit structure (full model)
                circuit_path = os.path.join(folder, "circuit.png")
                full_circuit.visualize(file_path=circuit_path, node_size="small")
                viz_paths[trial_id][gname]["full"]["circuit"] = circuit_path

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
                    viz_paths[trial_id][gname]["full"]["activations_mean"] = (
                        mean_act_path
                    )

                # Full gate decision boundary visualization
                gate = resolve_gate(gname)
                gate_n_inputs = gate.n_inputs
                viz_folder = os.path.join(folder, "viz")
                os.makedirs(viz_folder, exist_ok=True)

                # Model decision boundary (requires pre-computed data)
                if trial.decision_boundary_data:
                    db_data = trial.decision_boundary_data.get(gname)
                    if db_data:
                        try:
                            # Decision boundary for full model
                            db_output = os.path.join(viz_folder, "decision_boundary")
                            if gate_n_inputs <= 2:
                                db_output += ".png"
                            db_paths = plot_decision_boundary_from_data(
                                data=db_data,
                                gate_name=f"{gname} (Full Model)",
                                output_path=db_output,
                            )
                            viz_paths[trial_id][gname]["full"]["decision_boundary"] = db_paths
                        except Exception as e:
                            print(f"[VIZ] Warning: Failed to save decision boundary for {gname} full: {e}")

                # Theoretical decision boundary (always generated - doesn't need model data)
                try:
                    if gate_n_inputs <= 2:
                        theoretical_path = os.path.join(viz_folder, "theoretical.png")
                        _plot_theoretical_decision_boundary(
                            gate=gate,
                            output_path=theoretical_path,
                        )
                        viz_paths[trial_id][gname]["full"]["theoretical"] = theoretical_path
                except Exception as e:
                    print(f"[VIZ] Warning: Failed to save theoretical boundary for {gname}: {e}")

        # --- Slice analysis (always generates JSON, conditional viz) ---
        with timed_phase("Slice Analysis"):
            for gate_idx, gname in enumerate(gate_names):
                best_keys = trial.metrics.per_gate_bests.get(gname, [])
                if not best_keys:
                    continue

                gate_dir = os.path.join(trial_dir, gname)

                # Get input size from layer sizes
                input_size = layer_sizes[0] if layer_sizes else 2

                # Generate all three slices
                obs_slice = ObservationalSlice.from_trial(trial, gname, width, depth, subcircuits, input_size)
                int_slice = InterventionalSlice.from_trial(trial, gname, width, depth, subcircuits, input_size)
                cf_slice = CounterfactualSlice.from_trial(trial, gname, width, depth, subcircuits, input_size)

                # Save JSON files (always)
                obs_paths = obs_slice.save(gate_dir)
                int_paths = int_slice.save(gate_dir)
                cf_paths = cf_slice.save(gate_dir)

                viz_paths[trial_id].setdefault(gname, {})["slice_analysis"] = {
                    "observational": obs_paths,
                    "interventional": int_paths,
                    "counterfactual": cf_paths,
                }

                # Generate slice visualizations (only if not skipping circuit figures)
                if not viz_config.skip_circuit_figures:
                    slice_viz_paths = {}
                    for slice_obj, slice_name in [
                        (obs_slice, "observational"),
                        (int_slice, "interventional"),
                        (cf_slice, "counterfactual"),
                    ]:
                        slice_viz = visualize_slice(slice_obj, gate_dir, gate_name=gname, trial=trial)
                        if slice_viz:
                            slice_viz_paths[slice_name] = slice_viz

                    viz_paths[trial_id][gname]["slice_viz"] = slice_viz_paths

                # Generate node-level circuit.png (e.g., XOR/node222/circuit.png)
                # Get unique node patterns from best subcircuits
                node_patterns_seen = set()
                for sc_key in best_keys:
                    node_mask_idx, _ = parse_subcircuit_key(sc_key, width, depth)
                    node_patterns_seen.add(node_mask_idx)

                for node_mask_idx in node_patterns_seen:
                    node_dir = os.path.join(trial_dir, gname, f"node{node_mask_idx}")
                    os.makedirs(node_dir, exist_ok=True)

                    circuit = subcircuits[node_mask_idx]
                    circuit_path = os.path.join(node_dir, "circuit.png")
                    try:
                        circuit.visualize(file_path=circuit_path, node_size="small")
                        viz_paths[trial_id].setdefault(gname, {}).setdefault(f"node{node_mask_idx}", {})["circuit"] = circuit_path
                    except Exception as e:
                        print(f"[VIZ] Warning: Failed to generate node{node_mask_idx}/circuit.png: {e}")

        # --- Subcircuit visualization ---
        # Skip if level doesn't include subcircuit visualization
        if viz_config.skip_circuit_figures:
            continue

        for gate_idx, gname in enumerate(gate_names):
            best_keys = trial.metrics.per_gate_bests.get(gname, [])
            if not best_keys:
                continue

            bests_faith = trial.metrics.per_gate_bests_faith.get(gname, [])

            # Sort by faithfulness (not accuracy) for visualization
            # This ensures we visualize the most faithful subcircuits
            key_faith_pairs = []
            for i, key in enumerate(best_keys):
                faith = bests_faith[i] if i < len(bests_faith) else None
                faith_score = faith.overall_faithfulness if faith else 0
                key_faith_pairs.append((key, faith_score, i))

            # Sort by faithfulness descending
            key_faith_pairs.sort(key=lambda x: x[1], reverse=True)

            # Limit subcircuits based on viz level
            max_sc = viz_config.max_subcircuits_per_gate
            selected_pairs = key_faith_pairs[:max_sc]

            print(f"[VIZ] Gate {gname}: {len(selected_pairs)} best subcircuits to visualize (max: {max_sc}, ranked by faithfulness)")

            # Get the actual edge variant circuits (not just base node patterns)
            per_gate_circuits = trial.metrics.per_gate_circuits.get(gname, {})

            for rank_idx, (sc_key, faith_score, orig_idx) in enumerate(selected_pairs):
                node_mask_idx, edge_variant_rank = parse_subcircuit_key(sc_key, width, depth)

                # Use the actual edge variant circuit if available, otherwise fall back to base
                if sc_key in per_gate_circuits:
                    circuit_dict = per_gate_circuits[sc_key]
                    circuit = Circuit.from_dict(circuit_dict)
                else:
                    # Fallback to base circuit (all edges active for node pattern)
                    circuit = subcircuits[node_mask_idx]

                # Build folder path using node{X}/{rank}_{sc_key}/ structure
                folder = os.path.join(
                    trial_dir, gname, f"node{node_mask_idx}", f"{rank_idx}_{sc_key}"
                )
                sc_label = f"{gname} (Node#{node_mask_idx}/SC#{sc_key})"

                os.makedirs(folder, exist_ok=True)
                viz_paths[trial_id].setdefault(gname, {})[sc_key] = viz_paths[trial_id].get(gname, {}).get(sc_key, {})

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
                    viz_paths[trial_id][gname][sc_key]["activations_mean"] = (
                        mean_act_path
                    )

                # Subcircuit decision boundary visualization from pre-computed data
                # NO model inference here - uses trial.subcircuit_decision_boundary_data
                if trial.subcircuit_decision_boundary_data:
                    gate_db_data = trial.subcircuit_decision_boundary_data.get(
                        gname, {}
                    )
                    # Lookup using flat subcircuit index
                    db_data = gate_db_data.get(sc_key)
                    if db_data:
                        try:
                            gate_n_inputs = resolve_gate(gname).n_inputs
                            # Create viz subfolder and save
                            viz_folder = os.path.join(folder, "viz")
                            os.makedirs(viz_folder, exist_ok=True)
                            db_output = os.path.join(viz_folder, "decision_boundary")
                            if gate_n_inputs <= 2:
                                db_output += ".png"
                            db_paths = plot_decision_boundary_from_data(
                                data=db_data,
                                gate_name=sc_label,
                                output_path=db_output,
                            )
                            viz_paths[trial_id][gname][sc_key]["decision_boundary"] = (
                                db_paths
                            )
                        except Exception as e:
                            print(
                                f"[VIZ] Warning: Failed to save decision boundary for {sc_label}: {e}"
                            )

                # Robustness and Faithfulness visualization
                # Robustness is now inside faithfulness.observational
                has_faith = orig_idx < len(bests_faith)
                faithfulness_data = bests_faith[orig_idx] if has_faith else None
                observational_data = (
                    faithfulness_data.observational if faithfulness_data else None
                )
                has_observational = observational_data is not None

                # Faithfulness directory already created by save_per_gate_data
                faithfulness_dir = os.path.join(folder, "faithfulness")
                os.makedirs(faithfulness_dir, exist_ok=True)
                viz_paths[trial_id][gname][sc_key].setdefault("faithfulness", {})

                # Observational dir (renamed from robustness) - lives inside faithfulness/
                observational_dir = (
                    os.path.join(faithfulness_dir, "observational")
                    if has_observational
                    else None
                )
                if observational_dir:
                    os.makedirs(observational_dir, exist_ok=True)
                    viz_paths[trial_id][gname][sc_key]["faithfulness"].setdefault(
                        "observational", {}
                    )

                # Run quick visualizations sequentially (matplotlib is not thread-safe)
                if has_observational and observational_dir:
                    with timed_phase("Observational Curves"):
                        viz_paths[trial_id][gname][sc_key]["faithfulness"][
                            "observational"
                        ]["stats"] = visualize_observational_curves(
                            observational_data, observational_dir, sc_label
                        )

                # Run expensive circuit visualizations sequentially (they use ProcessPoolExecutor internally)
                if has_observational and observational_dir:
                    with timed_phase("Observational Circuit Viz"):
                        circuit_paths = visualize_observational_circuits(
                            observational_data,
                            circuit,
                            layer_weights,
                            observational_dir,
                            canonical_activations=canonical_activations,
                            layer_biases=layer_biases if layer_biases else None,
                        )
                    viz_paths[trial_id][gname][sc_key]["faithfulness"]["observational"][
                        "circuit_viz"
                    ] = circuit_paths

                if has_faith:
                    with timed_phase("Faithfulness Circuit Viz"):
                        circuit_paths = visualize_faithfulness_circuit_samples(
                            faithfulness_data,
                            circuit,
                            layer_weights,
                            faithfulness_dir,
                            layer_biases=layer_biases if layer_biases else None,
                        )
                    viz_paths[trial_id][gname][sc_key]["faithfulness"][
                        "circuit_viz"
                    ] = circuit_paths

                    # Add intervention effect plots (like noise_by_input for robustness)
                    with timed_phase("Intervention Effects"):
                        intervention_paths = (
                            visualize_faithfulness_intervention_effects(
                                faithfulness_data, faithfulness_dir, sc_label
                            )
                        )
                    viz_paths[trial_id][gname][sc_key]["faithfulness"][
                        "intervention_effects"
                    ] = intervention_paths

    # Final viz profiling summary
    viz_elapsed_ms = (time.time() - viz_start) * 1000
    viz_mem_end = get_memory_mb()
    viz_mem_delta = viz_mem_end - viz_mem_start
    log_memory("after_visualization")
    print(f"\n{'~' * 60}")
    print("  VISUALIZATION COMPLETE")
    print(f"  Total time: {viz_elapsed_ms:.0f}ms")
    print(f"  Memory after: {viz_mem_end:.1f} MB (delta: {viz_mem_delta:+.1f} MB)")
    print(f"{'~' * 60}")

    return viz_paths
