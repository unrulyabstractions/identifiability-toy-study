"""
Save functions for experiment results.

Changes to this module should be reflected in README.md.

Structure:
    runs/run_{timestamp}/
        summary.json          - Ranked results across all trials and gates
        explanation.md        - How to read this folder
        config.json           - ExperimentConfig only
        trial_org.json        - Maps sweep parameters (width, depth, etc.) to trial IDs
        circuits.json         - Subcircuit masks and structure analysis (run-level)
        training_data/
            train.pt          - Training data tensors (x, y)
            val.pt            - Validation data tensors (x, y)
            test.pt           - Test data tensors (x, y)
            metadata.json     - Data shapes and gate names
        profiling/
            profiling.json    - Timing data (events, phase durations)
        trials/
            {trial_id}/
                summary.json      - Trial-level ranked results
                explanation.md    - How to read this folder
                config.json       - Full effective config (device, debug, + all trial params)
                setup.json        - TrialSetup (legacy, subset of config.json)
                metrics.json      - Metrics (training, per-gate, robustness, faithfulness)
                tensors.pt        - Test data, activations, weights
                all_gates/
                    model.pt
                gates/
                    {gate_name}/
                        summary.json      - Gate-level ranked subcircuits
                        explanation.md    - How to read this folder
                        full/
                            decomposed_model.pt
                        {subcircuit_idx}/
                            decomposed_model.pt
                spd/                    - SPD decomposition outputs
                    config.json         - SPD configuration used
                    decomposed_model.pt - Trained decomposed model
                    estimate.json       - Subcircuit clustering results
                    component_importance.npy - Per-component importance scores
                    coactivation_matrix.npy  - Component coactivation matrix
                    clustering/
                        assignments.json    - Full clustering data
                        importance_matrix.npy
                        coactivation_matrix.npy
                    visualizations/
                        importance_heatmap.png
                        coactivation_matrix.png
                        components_as_circuits.png
                        uv_matrices.png
                        summary.png
                    clusters/
                        {cluster_idx}/
                            analysis.json
"""

import json
import math
import os
from dataclasses import asdict
from pathlib import Path

import torch

from src.analysis.subcircuit_ranking import (
    ALL_SUBCIRCUIT_METRICS,
    DEFAULT_SUBCIRCUIT_METRICS,
    SUBCIRCUIT_METRICS_RANKING,
    extract_all_metrics,
    filter_and_rank_subcircuit_metrics,
    get_ranking_dict,
    get_ranking_tuple,
)
from src.schemas import ExperimentResult
from src.serialization import filter_non_serializable

# =============================================================================
# Explanation Templates
# =============================================================================


def _save_json(data: dict, path: Path):
    """Save dict as formatted JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def _save_gates_registry(result, run_dir: Path):
    """Save gates.json with gate name to base gate function mapping.

    This documents which gate function each gate name uses, supporting
    repeated gates like ["XOR", "XOR"] -> ["XOR", "XOR_2"].
    """
    from src.domain import build_gate_registry, get_max_n_inputs, resolve_gate

    # Get gate names from first trial
    first_trial = next(iter(result.trials.values()), None)
    if first_trial is None:
        return

    gate_names = first_trial.setup.model_params.logic_gates

    # Build registry
    registry = build_gate_registry(gate_names)
    max_n_inputs = get_max_n_inputs(gate_names)

    gates_data = {
        "description": "Mapping from gate name to base gate function. Supports repeated gates and mixed input sizes.",
        "max_n_inputs": max_n_inputs,
        "gates": [
            {
                "name": name,
                "base_gate": base,
                "n_inputs": resolve_gate(name).n_inputs,
                "index": idx,
            }
            for idx, (name, base) in enumerate(registry.items())
        ],
        "registry": registry,
    }
    _save_json(gates_data, run_dir / "gates.json")


def _save_summary_by_gate(result, run_dir: Path):
    """Save summary_by_gate.json with per-gate aggregated metrics across all trials."""
    from src.domain import get_base_gate_name

    gate_summaries = {}

    for trial_id, trial in result.trials.items():
        for gate_name, gate_metrics in trial.metrics.per_gate_metrics.items():
            if gate_name not in gate_summaries:
                gate_summaries[gate_name] = {
                    "gate_name": gate_name,
                    "base_gate": get_base_gate_name(gate_name),
                    "n_trials": 0,
                    "test_accuracies": [],
                    "best_subcircuit_accuracies": [],
                    "trial_results": [],
                }

            summary = gate_summaries[gate_name]
            summary["n_trials"] += 1

            # Get test accuracy for this gate
            gate_test_acc = getattr(gate_metrics, "test_acc", None)
            if gate_test_acc is not None:
                summary["test_accuracies"].append(gate_test_acc)

            # Get best subcircuit metrics
            sc_metrics = getattr(gate_metrics, "subcircuit_metrics", [])
            sorted_by_acc = sorted(
                sc_metrics, key=lambda x: getattr(x, "accuracy", 0) or 0, reverse=True
            )

            if sorted_by_acc:
                best = sorted_by_acc[0]
                summary["best_subcircuit_accuracies"].append(best.accuracy or 0)
                summary["trial_results"].append(
                    {
                        "trial_id": trial_id,
                        "test_acc": gate_test_acc,
                        "best_subcircuit_idx": best.idx,
                        "best_subcircuit_accuracy": best.accuracy,
                        "best_subcircuit_bit_similarity": best.bit_similarity,
                    }
                )

    # Compute aggregate statistics
    for gate_name, summary in gate_summaries.items():
        accs = summary["test_accuracies"]
        sc_accs = summary["best_subcircuit_accuracies"]
        summary["avg_test_accuracy"] = sum(accs) / len(accs) if accs else None
        summary["avg_best_subcircuit_accuracy"] = (
            sum(sc_accs) / len(sc_accs) if sc_accs else None
        )
        summary["min_best_subcircuit_accuracy"] = min(sc_accs) if sc_accs else None
        summary["max_best_subcircuit_accuracy"] = max(sc_accs) if sc_accs else None

    _save_json(
        {
            "description": "Per-gate summary aggregated across all trials",
            "gates": gate_summaries,
        },
        run_dir / "summary_by_gate.json",
    )


def _save_trial_organization(result: ExperimentResult, run_dir: Path):
    """Save trial_org.json mapping sweep parameters to trial IDs.

    Creates a file that shows which trials used which sweep values, making it
    easy to find trials with specific configurations.

    Structure:
        sweep_axes: Lists all unique values for each sweep parameter
        trials: List of trial configs with their IDs
        by_axis: Grouped trial IDs by each axis value
    """
    from collections import defaultdict

    # Collect all unique sweep values and trial configs
    widths = set()
    depths = set()
    activations = set()
    learning_rates = set()
    gate_combos = set()
    seeds = set()

    trials_list = []
    by_width = defaultdict(list)
    by_depth = defaultdict(list)
    by_activation = defaultdict(list)
    by_lr = defaultdict(list)
    by_gates = defaultdict(list)
    by_seed = defaultdict(list)

    for trial_id, trial in result.trials.items():
        setup = trial.setup
        mp = setup.model_params
        tp = setup.train_params

        # Extract sweep values
        width = mp.width
        depth = mp.depth
        activation = mp.activation
        lr = tp.learning_rate
        gates = tuple(mp.logic_gates)
        seed = setup.seed

        # Track unique values
        widths.add(width)
        depths.add(depth)
        activations.add(activation)
        learning_rates.add(lr)
        gate_combos.add(gates)
        seeds.add(seed)

        # Build trial entry
        trial_entry = {
            "trial_id": trial_id,
            "width": width,
            "depth": depth,
            "activation": activation,
            "learning_rate": lr,
            "gates": list(gates),
            "seed": seed,
        }
        trials_list.append(trial_entry)

        # Group by axis
        by_width[width].append(trial_id)
        by_depth[depth].append(trial_id)
        by_activation[activation].append(trial_id)
        by_lr[lr].append(trial_id)
        by_gates[",".join(gates)].append(trial_id)
        by_seed[seed].append(trial_id)

    # Build output structure
    trial_org = {
        "description": "Maps sweep parameters to trial IDs for easy lookup",
        "n_trials": len(result.trials),
        "sweep_axes": {
            "width": sorted(widths),
            "depth": sorted(depths),
            "activation": sorted(activations),
            "learning_rate": sorted(learning_rates),
            "gates": [list(g) for g in sorted(gate_combos, key=lambda x: (len(x), x))],
            "seed": sorted(seeds),
        },
        "trials": sorted(
            trials_list,
            key=lambda t: (
                t["width"],
                t["depth"],
                t["activation"],
                t["learning_rate"],
                len(t["gates"]),
                tuple(t["gates"]),
                t["seed"],
            ),
        ),
        "by_axis": {
            "width": {str(k): v for k, v in sorted(by_width.items())},
            "depth": {str(k): v for k, v in sorted(by_depth.items())},
            "activation": dict(sorted(by_activation.items())),
            "learning_rate": {str(k): v for k, v in sorted(by_lr.items())},
            "gates": dict(sorted(by_gates.items())),
            "seed": {str(k): v for k, v in sorted(by_seed.items())},
        },
    }

    _save_json(trial_org, run_dir / "trial_org.json")


def _save_decision_boundary_visualizations(
    result: ExperimentResult, run_dir: Path, logger=None
):
    """Save decision boundary visualizations for each gate using Monte Carlo sampling.

    Data generation happens here (calling generate_* functions), then plotting reads the data.

    Creates:
        visualizations/
            decision_boundaries/
                {gate_name}_decision_boundary.png (for 1D/2D)
                {gate_name}_decision_boundary_3d.png (for 3D)
                {gate_name}_decision_boundary_proj_*.png (2D projections for 3D+)
    """
    from src.domain import get_max_n_inputs
    from src.visualization.decision_boundary import (
        generate_grid_data,
        generate_monte_carlo_data,
        visualize_all_gates_from_data,
    )

    # Get first trial with a model
    first_trial = next((t for t in result.trials.values() if t.model is not None), None)
    if first_trial is None:
        return

    gate_names = first_trial.setup.model_params.logic_gates
    n_inputs = get_max_n_inputs(gate_names)

    viz_dir = run_dir / "visualizations" / "decision_boundaries"
    viz_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Generate data for each gate
        gate_data = {}
        for gate_idx, gate_name in enumerate(gate_names):
            if n_inputs <= 2:
                # Use grid data for clean contour plots
                data = generate_grid_data(
                    model=first_trial.model,
                    n_inputs=n_inputs,
                    gate_idx=gate_idx,
                    resolution=100,
                )
            else:
                # Use Monte Carlo for higher dimensions
                data = generate_monte_carlo_data(
                    model=first_trial.model,
                    n_inputs=n_inputs,
                    gate_idx=gate_idx,
                    n_samples=5000,
                )
            gate_data[gate_name] = data

        # Save visualizations from data
        paths = visualize_all_gates_from_data(
            gate_data=gate_data,
            output_dir=str(viz_dir),
        )
        logger and logger.info(f"Saved decision boundary visualizations to {viz_dir}")
    except Exception as e:
        logger and logger.warning(
            f"Failed to save decision boundary visualizations: {e}"
        )


def _save_subcircuit_decision_boundaries(
    result: ExperimentResult, run_dir: Path, logger=None
):
    """Save decision boundary visualizations for each analyzed subcircuit.

    For each gate's best subcircuits (after filtering), generates Monte Carlo
    decision boundary data and visualizations.

    Creates:
        visualizations/
            subcircuit_boundaries/
                {gate_name}/
                    subcircuit_{node_mask_idx}_{edge_idx}_decision_boundary.png
    """
    import torch

    from src.domain import get_max_n_inputs, resolve_gate
    from src.visualization.decision_boundary import (
        generate_grid_data,
        generate_monte_carlo_data,
        plot_decision_boundary_from_data,
    )

    # Get first trial with metrics
    first_trial = next(
        (
            t
            for t in result.trials.values()
            if t.model is not None and t.metrics.per_gate_bests
        ),
        None,
    )
    if first_trial is None:
        return

    gate_names = first_trial.setup.model_params.logic_gates
    model = first_trial.model
    per_gate_bests = first_trial.metrics.per_gate_bests
    max_n_inputs = get_max_n_inputs(gate_names)

    viz_dir = run_dir / "visualizations" / "subcircuit_boundaries"

    # Create a wrapper that pads inputs for gates with fewer inputs than max
    class _PaddedGateModel:
        def __init__(self, gate_model, gate_n_inputs, max_n_inputs, device):
            self.gate_model = gate_model
            self.gate_n_inputs = gate_n_inputs
            self.max_n_inputs = max_n_inputs
            self.device = device

        def __call__(self, x):
            # Pad input to max_n_inputs if needed
            if x.shape[1] < self.max_n_inputs:
                padding = torch.zeros(
                    x.shape[0], self.max_n_inputs - x.shape[1], device=x.device
                )
                x = torch.cat([x, padding], dim=1)
            return self.gate_model(x)

        def eval(self):
            self.gate_model.eval()
            return self

        def parameters(self):
            return self.gate_model.parameters()

    try:
        for gate_idx, gate_name in enumerate(gate_names):
            gate_dir = viz_dir / gate_name
            gate_dir.mkdir(parents=True, exist_ok=True)

            # Get the best subcircuit keys for this gate
            best_keys = per_gate_bests.get(gate_name, [])
            if not best_keys:
                continue

            gate_n_inputs = resolve_gate(gate_name).n_inputs

            # Get the gate model with input padding wrapper
            gate_model = model.separate_into_k_mlps()[gate_idx]
            device = str(next(gate_model.parameters()).device)
            wrapped_model = _PaddedGateModel(
                gate_model, gate_n_inputs, max_n_inputs, device
            )

            for key in best_keys[:5]:  # Limit to top 5 subcircuits per gate
                # Key format is (node_mask_idx, edge_mask_idx)
                if isinstance(key, (tuple, list)):
                    node_mask_idx, edge_mask_idx = key
                    subcircuit_name = f"subcircuit_n{node_mask_idx}_e{edge_mask_idx}"
                else:
                    subcircuit_name = f"subcircuit_{key}"

                # Generate decision boundary data using the gate's actual input size
                try:
                    if gate_n_inputs <= 2:
                        data = generate_grid_data(
                            model=wrapped_model,
                            n_inputs=gate_n_inputs,
                            gate_idx=0,  # Single gate model has output index 0
                            resolution=100,
                            device=device,
                        )
                    else:
                        data = generate_monte_carlo_data(
                            model=wrapped_model,
                            n_inputs=gate_n_inputs,
                            gate_idx=0,
                            n_samples=2000,
                            device=device,
                        )

                    # Save visualization
                    output_path = str(gate_dir / f"{subcircuit_name}_decision_boundary")
                    if gate_n_inputs <= 2:
                        output_path += ".png"
                    plot_decision_boundary_from_data(
                        data=data,
                        gate_name=f"{gate_name} ({subcircuit_name})",
                        output_path=output_path,
                    )
                except Exception as e:
                    logger and logger.debug(
                        f"Failed to generate subcircuit boundary for {gate_name}/{subcircuit_name}: {e}"
                    )

        logger and logger.info(
            f"Saved subcircuit decision boundary visualizations to {viz_dir}"
        )
    except Exception as e:
        logger and logger.warning(
            f"Failed to save subcircuit decision boundary visualizations: {e}"
        )


def save_results(result: ExperimentResult, run_dir: str | Path, logger=None):
    """
    Save experiment results to disk with clean folder structure.

    Creates:
        config.json           - Experiment configuration
        circuits.json         - Circuit masks and structures (run-level)
        profiling/profiling.json - Profiling data (run-level)
        trials/
            {trial_id}/
                config.json       - Full effective config (device, debug, + trial params)
                setup.json        - Trial setup parameters (legacy)
                metrics.json      - All metrics and analysis results
                tensors.pt        - Tensor data (test samples, activations, weights)
                all_gates/model.pt
                gates/{gate}/full/decomposed_model.pt
                gates/{gate}/{sc_idx}/decomposed_model.pt
    """
    os.makedirs(run_dir, exist_ok=True)
    run_dir = Path(run_dir)

    # Save experiment config
    config_data = filter_non_serializable(asdict(result.config))
    _save_json(config_data, run_dir / "config.json")

    # Save run-level summary and explanation
    run_summary = _generate_run_summary(result)
    _save_json(run_summary, run_dir / "summary.json")
    _save_explanation(run_dir / "explanation.md", RUN_EXPLANATION)

    # Save gates.json with gate name -> base gate function mapping
    _save_gates_registry(result, run_dir)

    # Save circuits at run level (from first trial - all trials share same circuit enumeration)
    _save_run_level_circuits(result, run_dir)

    # Save profiling at run level (aggregate from all trials)
    _save_run_level_profiling(result, run_dir)

    # Save subcircuits folder with rankings
    _save_subcircuits_folder(result, run_dir)

    # Save summary_by_gate.json with per-gate aggregated metrics
    _save_summary_by_gate(result, run_dir)

    # Save trial_org.json mapping sweep parameters to trial IDs
    _save_trial_organization(result, run_dir)

    # Create trials directory
    trials_dir = run_dir / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)

    # Save decision boundary visualizations (once for first trial's model)
    _save_decision_boundary_visualizations(result, run_dir, logger)

    # Save subcircuit-level decision boundary visualizations
    _save_subcircuit_decision_boundaries(result, run_dir, logger)

    # Save each trial
    for trial_id, trial in result.trials.items():
        trial_dir = trials_dir / trial_id
        trial_dir.mkdir(parents=True, exist_ok=True)

        # 1. Trial-level summary and explanation
        trial_summary = _generate_trial_summary(trial)
        _save_json(trial_summary, trial_dir / "summary.json")
        _save_explanation(trial_dir / "explanation.md", TRIAL_EXPLANATION)

        # 2. Setup JSON - trial parameters
        setup_data = filter_non_serializable(asdict(trial.setup))
        _save_json(setup_data, trial_dir / "setup.json")

        # 3. Config JSON - full effective config (experiment-level + trial-level)
        effective_config = {
            "device": result.config.device,
            "debug": result.config.debug,
            "trial_id": trial_id,
            **setup_data,
        }
        _save_json(effective_config, trial_dir / "config.json")

        # 4. Metrics JSON - training and analysis results
        metrics_data = filter_non_serializable(asdict(trial.metrics))
        metrics_data["status"] = trial.status
        metrics_data["trial_id"] = trial.trial_id
        _save_json(metrics_data, trial_dir / "metrics.json")

        # 5. Tensors PT - all tensor data
        _save_tensors(trial, trial_dir, logger)

        # 6. Models
        _save_models(trial, trial_dir, logger)

        # 7. Per-gate summaries
        _save_gate_summaries(trial, trial_dir)

    logger and logger.info(f"Saved results to {run_dir}")


def save_training_data(data, run_dir: str | Path, gate_names: list[str] = None):
    """Save training data to disk.

    Args:
        data: TrialData object with train/val/test datasets
        run_dir: Run directory path
        gate_names: Optional list of gate names for metadata

    Creates:
        training_data/
            train.pt   - {x: tensor, y: tensor}
            val.pt     - {x: tensor, y: tensor}
            test.pt    - {x: tensor, y: tensor}
            metadata.json - shapes and gate info
    """
    run_dir = Path(run_dir)
    data_dir = run_dir / "training_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Save tensors
    torch.save({"x": data.train.x, "y": data.train.y}, data_dir / "train.pt")
    torch.save({"x": data.val.x, "y": data.val.y}, data_dir / "val.pt")
    torch.save({"x": data.test.x, "y": data.test.y}, data_dir / "test.pt")

    # Save metadata
    metadata = {
        "train_shape": {"x": list(data.train.x.shape), "y": list(data.train.y.shape)},
        "val_shape": {"x": list(data.val.x.shape), "y": list(data.val.y.shape)},
        "test_shape": {"x": list(data.test.x.shape), "y": list(data.test.y.shape)},
        "gate_names": gate_names,
    }
    _save_json(metadata, data_dir / "metadata.json")


def _simplify_circuit_masks(subcircuit: dict) -> dict:
    """Simplify circuit masks for storage per T1.d.2.

    Per spec:
    - node_masks: Only include middle layers (input is never masked,
      output is always 1 for single gate)
    - edge_masks: Organize as if we have 1 output (since we always
      do separate_into_k_mlps)

    Args:
        subcircuit: Dict with 'node_masks' and 'edge_masks' keys

    Returns:
        Simplified subcircuit dict with only middle layer masks
    """
    node_masks = subcircuit.get("node_masks", [])
    edge_masks = subcircuit.get("edge_masks", [])

    # node_masks: exclude input (index 0) and output (last index)
    # Middle layers are indices 1 to n-2 inclusive
    simplified_node_masks = node_masks[1:-1] if len(node_masks) > 2 else []

    # edge_masks: keep as-is since each gate will use adapted version
    # but we could simplify to single output if needed
    # For now, keep edge_masks but note they apply to full architecture
    simplified_edge_masks = edge_masks

    return {
        "idx": subcircuit.get("idx"),
        "node_masks": simplified_node_masks,
        "edge_masks": simplified_edge_masks,
    }


def _save_run_level_circuits(result: ExperimentResult, run_dir: Path):
    """Save circuits.json at run level from first trial.

    Saves simplified circuit masks per T1.d.2:
    - node_masks: Only middle layers (not input/output)
    - edge_masks: Full connections (used with adapt_masks_for_gate)
    - subcircuit_idx, node_mask_idx, edge_mask_idx: Index mappings
    """
    # Get circuits from first trial (all trials share the same circuit enumeration)
    first_trial = next(iter(result.trials.values()), None)
    if first_trial is None:
        return

    # Track unique node patterns to assign node_mask_idx
    seen_node_patterns = {}  # tuple(hidden_masks) -> node_mask_idx
    node_mask_counter = 0
    edge_mask_counts = {}  # node_mask_idx -> count (for edge_mask_idx)

    simplified_subcircuits = []
    for sc in first_trial.subcircuits:
        subcircuit_idx = sc.get("idx", len(simplified_subcircuits))
        node_masks = sc.get("node_masks", [])
        edge_masks = sc.get("edge_masks", [])

        # Simplify node_masks: exclude input (index 0) and output (last index)
        simplified_node_masks = node_masks[1:-1] if len(node_masks) > 2 else []

        # Extract hidden layer pattern for node_mask_idx assignment
        hidden_pattern = tuple(tuple(m) for m in simplified_node_masks)

        # Assign node_mask_idx
        if hidden_pattern not in seen_node_patterns:
            seen_node_patterns[hidden_pattern] = node_mask_counter
            edge_mask_counts[node_mask_counter] = 0
            node_mask_counter += 1
        node_mask_idx = seen_node_patterns[hidden_pattern]

        # Assign edge_mask_idx (increments for each subcircuit with same node pattern)
        edge_mask_idx = edge_mask_counts[node_mask_idx]
        edge_mask_counts[node_mask_idx] += 1

        simplified_subcircuits.append(
            {
                "subcircuit_idx": subcircuit_idx,
                "node_pattern": node_mask_idx,
                "edge_variation": edge_mask_idx,
                "node_masks": simplified_node_masks,
                "edge_masks": edge_masks,
            }
        )

    circuits_data = {
        "subcircuits": simplified_subcircuits,
        "subcircuit_structure_analysis": first_trial.subcircuit_structure_analysis,
    }
    _save_json(circuits_data, run_dir / "circuits.json")


def _save_run_level_profiling(result: ExperimentResult, run_dir: Path):
    """Save profiling data at run level (aggregated from all trials)."""
    profiling_dir = run_dir / "profiling"
    profiling_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate profiling from all trials
    all_profiling = {}
    for trial_id, trial in result.trials.items():
        profiling_data = filter_non_serializable(asdict(trial.profiling))
        all_profiling[trial_id] = profiling_data

    _save_json(all_profiling, profiling_dir / "profiling.json")


def _save_tensors(trial, trial_dir: Path, logger=None):
    """Save trial tensors to tensors.pt."""
    tensors_path = trial_dir / "tensors.pt"
    data = {}

    if trial.test_x is not None:
        data["test_x"] = trial.test_x.cpu()
    if trial.test_y is not None:
        data["test_y"] = trial.test_y.cpu()
    if trial.test_y_pred is not None:
        data["test_y_pred"] = trial.test_y_pred.cpu()
    if trial.activations is not None:
        data["activations"] = [a.cpu() for a in trial.activations]
    if trial.canonical_activations is not None:
        data["canonical_activations"] = {
            label: [a.cpu() for a in acts]
            for label, acts in trial.canonical_activations.items()
        }
    if trial.mean_activations_by_range is not None:
        data["mean_activations_by_range"] = {
            label: [a.cpu() for a in acts]
            for label, acts in trial.mean_activations_by_range.items()
        }
    if trial.layer_weights is not None:
        data["layer_weights"] = [w.cpu() for w in trial.layer_weights]
    if trial.layer_biases is not None:
        data["layer_biases"] = [b.cpu() for b in trial.layer_biases]

    if data:
        torch.save(data, tensors_path)
        logger and logger.info(f"Saved tensors to {tensors_path}")


def _save_models(trial, trial_dir: Path, logger=None):
    """Save model files."""
    # All gates model
    all_gates_dir = trial_dir / "all_gates"
    all_gates_dir.mkdir(parents=True, exist_ok=True)

    if trial.model is not None:
        model_path = all_gates_dir / "model.pt"
        trial.model.save_to_file(str(model_path))


def _generate_run_summary(result: ExperimentResult) -> dict:
    """Generate run-level summary with ranked results across all trials and gates."""
    # Get epsilon from config (default 0.2)
    epsilon = 0.2
    if hasattr(result.config, "base_trial") and hasattr(
        result.config.base_trial, "constraints"
    ):
        epsilon = getattr(result.config.base_trial.constraints, "epsilon", 0.2)

    # Get normalized gate names from first trial (handles repeated gates like XOR, XOR_2)
    first_trial = next(iter(result.trials.values()), None)
    normalized_gates = (
        first_trial.setup.model_params.logic_gates
        if first_trial
        else list(result.config.target_logic_gates)
    )

    summary = {
        "experiment_id": result.experiment_id,
        "n_trials": len(result.trials),
        "gates": normalized_gates,
        "architecture": {
            "widths": result.config.widths,
            "depths": result.config.depths,
        },
        "epsilon": epsilon,
        "best_subcircuits_by_gate": {},
        "trial_results": [],
    }

    # Aggregate results by gate across all trials
    gate_results = {}  # gate -> list of subcircuit result dicts
    for trial_id, trial in result.trials.items():
        trial_summary = {
            "trial_id": trial_id,
            "test_acc": trial.metrics.test_acc,
            "val_acc": trial.metrics.val_acc,
            "gates": {},
        }

        # Get faithfulness data for this trial
        bests_keys = trial.metrics.per_gate_bests or {}
        bests_faith = trial.metrics.per_gate_bests_faith or {}

        for gate, gate_metrics in trial.metrics.per_gate_metrics.items():
            if gate not in gate_results:
                gate_results[gate] = []

            # Build idx -> faithfulness mapping
            gate_best_keys = bests_keys.get(gate, [])
            gate_faith_list = bests_faith.get(gate, [])
            idx_to_faith = {}
            for k, faith in zip(gate_best_keys, gate_faith_list):
                idx = k[0] if isinstance(k, (tuple, list)) else k
                if idx not in idx_to_faith:
                    idx_to_faith[idx] = faith

            # Get subcircuit metrics and find best ones
            sc_metrics = getattr(gate_metrics, "subcircuit_metrics", [])
            sorted_by_acc = sorted(
                sc_metrics, key=lambda x: getattr(x, "accuracy", 0) or 0, reverse=True
            )

            for sm in sorted_by_acc[:5]:  # Top 5 per gate
                acc = sm.accuracy or 0
                faith_data = idx_to_faith.get(sm.idx)
                ranked = filter_and_rank_subcircuit_metrics(
                    sm, faithfulness_data=faith_data
                )
                gate_results[gate].append(
                    {
                        "trial_id": trial_id,
                        "subcircuit_idx": sm.idx,
                        "passes_epsilon": acc >= (1 - epsilon),
                        "ranked_metrics": ranked,
                    }
                )

            trial_summary["gates"][gate] = {
                "best_idx": sorted_by_acc[0].idx if sorted_by_acc else None,
                "best_acc": sorted_by_acc[0].accuracy if sorted_by_acc else None,
            }

        summary["trial_results"].append(trial_summary)

    # Rank and store best subcircuits per gate (by first metric value, typically accuracy)
    for gate, results in gate_results.items():

        def get_first_metric(r):
            vals = r.get("ranked_metrics", {}).get("metric_value", [])
            return vals[0] if vals else 0

        sorted_results = sorted(results, key=get_first_metric, reverse=True)
        summary["best_subcircuits_by_gate"][gate] = sorted_results[:10]

    return summary


def _generate_trial_summary(trial) -> dict:
    """Generate trial-level summary with ranked results per gate."""
    # Get epsilon from trial setup
    epsilon = 0.2
    if hasattr(trial, "setup") and hasattr(trial.setup, "constraints"):
        epsilon = getattr(trial.setup.constraints, "epsilon", 0.2)

    gates_summary = {}
    for gate, gate_metrics in trial.metrics.per_gate_metrics.items():
        sc_metrics = getattr(gate_metrics, "subcircuit_metrics", [])
        sorted_by_acc = sorted(
            sc_metrics, key=lambda x: getattr(x, "accuracy", 0) or 0, reverse=True
        )
        gates_summary[gate] = {
            "gate_accuracy": getattr(gate_metrics, "test_acc", None),
            "best_subcircuits": [
                {
                    "subcircuit_idx": sm.idx,
                    "accuracy": sm.accuracy,
                    "bit_similarity": sm.bit_similarity,
                    "passes_epsilon": (sm.accuracy or 0) >= (1 - epsilon),
                }
                for sm in sorted_by_acc[:5]
            ],
        }

    return {
        "trial_id": trial.trial_id,
        "status": trial.status,
        "test_acc": trial.metrics.test_acc,
        "val_acc": trial.metrics.val_acc,
        "avg_loss": trial.metrics.avg_loss,
        "epsilon": epsilon,
        "gates": gates_summary,
    }


def _generate_gate_summary(gate_name: str, gate_metrics, epsilon: float = 0.2) -> dict:
    """Generate gate-level summary with ranked subcircuits."""
    subcircuit_ranking = []
    for sm in getattr(gate_metrics, "subcircuit_metrics", []):
        acc = sm.accuracy or 0
        subcircuit_ranking.append(
            {
                "subcircuit_idx": sm.idx,
                "accuracy": acc,
                "bit_similarity": sm.bit_similarity,
                "passes_epsilon": acc >= (1 - epsilon),
            }
        )
    subcircuit_ranking.sort(key=lambda x: x.get("accuracy", 0) or 0, reverse=True)

    return {
        "gate_name": gate_name,
        "gate_accuracy": gate_metrics.test_acc
        if hasattr(gate_metrics, "test_acc")
        else None,
        "epsilon": epsilon,
        "n_subcircuits_passing": sum(
            1 for s in subcircuit_ranking if s.get("passes_epsilon")
        ),
        "n_subcircuits_total": len(subcircuit_ranking),
        "ranked_subcircuits": subcircuit_ranking[:20],
    }


RUN_EXPLANATION = """# Run Summary

## Structure

- `summary.json`: Ranked results across all trials and gates
- `config.json`: Experiment configuration
- `trial_org.json`: Maps sweep parameters (width, depth, activation, etc.) to trial IDs
- `circuits.json`: All enumerated subcircuits (shared across trials)
- `profiling/`: Timing information
- `trials/`: Per-trial results

## Key Metrics

Let M = trained MLP, G_k = gate k's subnetwork, S_i = subcircuit i.

- **accuracy**: Pr[S_i(x) = G_k(x)] over test set
- **bit_similarity**: mean(1 - |S_i(x) - G_k(x)|) for binary outputs
- **faithfulness**: Combined metric from observational, interventional, counterfactual tests

## Reading summary.json

`best_subcircuits_by_gate[gate]` lists top subcircuits ranked by accuracy.
Each entry has: subcircuit_idx, accuracy, bit_similarity, faithfulness.
"""


TRIAL_EXPLANATION = """# Trial Summary

## Structure

- `summary.json`: Trial-level ranked results
- `config.json`: Full effective config (device, debug, + all trial parameters)
- `setup.json`: Trial hyperparameters (legacy, subset of config.json)
- `metrics.json`: Full metrics data
- `tensors.pt`: Test data, activations, weights
- `all_gates/model.pt`: Trained MLP
- `gates/{name}/`: Per-gate subcircuit analysis

## Key Metrics

- **test_acc**: Accuracy on test set
- **val_acc**: Accuracy on validation set
- **gates.{name}.best_subcircuits**: Top subcircuits for each gate
"""


GATE_EXPLANATION = """# Gate Analysis

## Structure

- `summary.json`: Ranked subcircuits for this gate
- Subcircuit folders contain decomposed models

## Metrics

Let G = this gate's subnetwork, S = subcircuit.

- **accuracy**: Pr[S(x) = G(x)]
- **passes_epsilon**: accuracy >= 1 - ε (identifiability threshold)
- **ranked_subcircuits**: Sorted by accuracy descending
"""


def _save_explanation(path: Path, content: str):
    """Save explanation markdown file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _save_subcircuits_folder(result: ExperimentResult, run_dir: Path):
    """Save subcircuits/ folder with score rankings and mappings.

    Creates:
        subcircuits/
            subcircuit_score_ranking.json      - Rankings by gate with all faithfulness scores
            subcircuit_score_ranking_per_trial.json - Per-trial rankings
            mask_idx_map.json                  - Mapping (node_mask_idx, edge_mask_idx) -> subcircuit_idx
            explanation.md                     - How to read these files
    """
    subcircuits_dir = run_dir / "subcircuits"
    subcircuits_dir.mkdir(parents=True, exist_ok=True)

    # Get epsilon and architecture from config
    epsilon = 0.2
    width = 3
    depth = 2
    if hasattr(result.config, "base_trial"):
        if hasattr(result.config.base_trial, "constraints"):
            epsilon = getattr(result.config.base_trial.constraints, "epsilon", 0.2)
        if hasattr(result.config.base_trial, "model_params"):
            width = getattr(result.config.base_trial.model_params, "width", 3)
            depth = getattr(result.config.base_trial.model_params, "depth", 2)

    # Aggregate scores by (gate, subcircuit_idx) across all trials
    # Now including faithfulness scores (observational, interventional, counterfactual)
    gate_subcircuit_data = {}  # gate -> subcircuit_idx -> list of score dicts

    per_trial_rankings = {}  # trial_id -> gate -> rankings

    for trial_id, trial in result.trials.items():
        per_trial_rankings[trial_id] = {}
        bests_keys = trial.metrics.per_gate_bests or {}
        bests_faith = trial.metrics.per_gate_bests_faith or {}

        for gate, gate_metrics in trial.metrics.per_gate_metrics.items():
            if gate not in gate_subcircuit_data:
                gate_subcircuit_data[gate] = {}

            # Build node_mask_idx -> first faithfulness mapping for this gate
            # Keys are (node_mask_idx, edge_mask_idx) tuples, we want to map by node_mask_idx
            gate_best_keys = bests_keys.get(gate, [])
            gate_faith_list = bests_faith.get(gate, [])
            node_mask_idx_to_faith = {}  # node_mask_idx -> first FaithfulnessMetrics
            for k, faith in zip(gate_best_keys, gate_faith_list):
                if isinstance(k, (tuple, list)):
                    node_mask_idx = k[0]
                else:
                    node_mask_idx = k
                # Only keep first faithfulness per node_mask_idx (best edge variant)
                if node_mask_idx not in node_mask_idx_to_faith:
                    node_mask_idx_to_faith[node_mask_idx] = faith

            sc_metrics = getattr(gate_metrics, "subcircuit_metrics", [])
            trial_ranking = []

            for sm in sc_metrics:
                idx = sm.idx
                acc = sm.accuracy or 0

                # Get faithfulness data for this subcircuit by node_mask_idx
                faith_data = node_mask_idx_to_faith.get(idx)

                # Extract faithfulness scores
                obs_score = None
                int_score = None
                cf_score = None
                if faith_data:
                    if (
                        hasattr(faith_data, "observational")
                        and faith_data.observational
                    ):
                        obs = faith_data.observational
                        obs_score = getattr(obs, "overall_observational", None)
                    if (
                        hasattr(faith_data, "interventional")
                        and faith_data.interventional
                    ):
                        int_data = faith_data.interventional
                        int_score = getattr(int_data, "overall_interventional", None)
                    if (
                        hasattr(faith_data, "counterfactual")
                        and faith_data.counterfactual
                    ):
                        cf_data = faith_data.counterfactual
                        cf_score = getattr(cf_data, "overall_counterfactual", None)

                if idx not in gate_subcircuit_data[gate]:
                    gate_subcircuit_data[gate][idx] = []

                gate_subcircuit_data[gate][idx].append(
                    {
                        "accuracy": acc,
                        "bit_similarity": sm.bit_similarity,
                        "observational": obs_score,
                        "interventional": int_score,
                        "counterfactual": cf_score,
                    }
                )

                trial_ranking.append(
                    {
                        "subcircuit_idx": idx,
                        "accuracy": acc,
                        "bit_similarity": sm.bit_similarity,
                        "observational": obs_score,
                        "interventional": int_score,
                        "counterfactual": cf_score,
                    }
                )

            trial_ranking.sort(key=lambda x: x.get("accuracy", 0), reverse=True)
            per_trial_rankings[trial_id][gate] = trial_ranking

    # Compute averaged rankings per gate with all scores
    averaged_rankings = {}
    for gate, subcircuit_data in gate_subcircuit_data.items():
        gate_rankings = []
        for idx, data_list in subcircuit_data.items():
            n = len(data_list)
            accs = [d["accuracy"] for d in data_list]
            avg_acc = sum(accs) / n if n > 0 else 0

            # Average faithfulness scores (ignoring None)
            obs_scores = [
                d["observational"] for d in data_list if d["observational"] is not None
            ]
            int_scores = [
                d["interventional"]
                for d in data_list
                if d["interventional"] is not None
            ]
            cf_scores = [
                d["counterfactual"]
                for d in data_list
                if d["counterfactual"] is not None
            ]

            gate_rankings.append(
                {
                    "subcircuit_idx": idx,
                    "avg_accuracy": avg_acc,
                    "n_trials": n,
                    "min_accuracy": min(accs) if accs else 0,
                    "max_accuracy": max(accs) if accs else 0,
                    "passes_epsilon": avg_acc >= (1 - epsilon),
                    "observational": {
                        "avg": sum(obs_scores) / len(obs_scores)
                        if obs_scores
                        else None,
                        "n": len(obs_scores),
                    },
                    "interventional": {
                        "avg": sum(int_scores) / len(int_scores)
                        if int_scores
                        else None,
                        "n": len(int_scores),
                    },
                    "counterfactual": {
                        "avg": sum(cf_scores) / len(cf_scores) if cf_scores else None,
                        "n": len(cf_scores),
                    },
                }
            )
        gate_rankings.sort(key=lambda x: x.get("avg_accuracy", 0), reverse=True)
        averaged_rankings[gate] = gate_rankings

    # Save averaged rankings
    _save_json(
        {
            "epsilon": epsilon,
            "n_trials": len(result.trials),
            "rankings_by_gate": averaged_rankings,
        },
        subcircuits_dir / "subcircuit_score_ranking.json",
    )

    # Save per-trial rankings
    _save_json(
        {
            "epsilon": epsilon,
            "per_trial": per_trial_rankings,
        },
        subcircuits_dir / "subcircuit_score_ranking_per_trial.json",
    )

    # Create mask_idx_map with node_mask_idx, edge_mask_idx -> subcircuit_idx mapping
    first_trial = next(iter(result.trials.values()), None)
    if first_trial and first_trial.subcircuits:
        # Track unique node patterns to assign node_mask_idx
        seen_node_patterns = {}  # tuple(hidden_masks) -> node_mask_idx
        node_mask_counter = 0

        mask_idx_entries = []
        idx_mapping = {}  # (node_mask_idx, edge_mask_idx) -> subcircuit_idx

        for sc in first_trial.subcircuits:
            subcircuit_idx = sc.get("idx", len(mask_idx_entries))
            node_masks = sc.get("node_masks", [])
            edge_masks = sc.get("edge_masks", [])

            # Extract hidden layer masks only (exclude input/output)
            hidden_masks = (
                tuple(tuple(m) for m in node_masks[1:-1]) if len(node_masks) > 2 else ()
            )

            # Assign node_mask_idx
            if hidden_masks not in seen_node_patterns:
                seen_node_patterns[hidden_masks] = node_mask_counter
                node_mask_counter += 1
            node_mask_idx = seen_node_patterns[hidden_masks]

            # For full_edges_only=True, edge_mask_idx is always 0 for each node pattern
            # Count how many subcircuits share this node pattern to assign edge_mask_idx
            edge_mask_idx = sum(
                1 for e in mask_idx_entries if e["node_pattern"] == node_mask_idx
            )

            # Compute structural metrics
            hidden_nodes = node_masks[1:-1] if len(node_masks) > 2 else node_masks
            total_hidden = sum(len(m) for m in hidden_nodes)
            active_hidden = sum(sum(m) for m in hidden_nodes)
            node_sparsity = active_hidden / total_hidden if total_hidden > 0 else 0

            total_edges = sum(len(e) * len(e[0]) for e in edge_masks if e and e[0])
            active_edges = sum(sum(sum(row) for row in e) for e in edge_masks if e)
            edge_sparsity = active_edges / total_edges if total_edges > 0 else 0

            # Compute additional topological metrics
            n_active_nodes_per_layer = [sum(m) for m in hidden_nodes]
            connectivity_density = edge_sparsity  # Same as edge_sparsity for now

            mask_idx_entries.append(
                {
                    "subcircuit_idx": subcircuit_idx,
                    "node_pattern": node_mask_idx,
                    "edge_variation": edge_mask_idx,
                    "node_sparsity": round(node_sparsity, 4),
                    "edge_sparsity": round(edge_sparsity, 4),
                    "n_hidden_layers": len(hidden_nodes),
                    "active_nodes_per_layer": n_active_nodes_per_layer,
                    "connectivity_density": round(connectivity_density, 4),
                }
            )

            idx_mapping[(node_mask_idx, edge_mask_idx)] = subcircuit_idx

        # Create readable mapping table
        mapping_table = [
            {"node_pattern": k[0], "edge_variation": k[1], "subcircuit_idx": v}
            for k, v in sorted(idx_mapping.items())
        ]

        _save_json(
            {
                "description": "Mapping (node_pattern, edge_variation) -> subcircuit_idx with structural metrics",
                "architecture": {"width": width, "depth": depth},
                "n_unique_node_patterns": node_mask_counter,
                "mapping": mapping_table,
                "subcircuits": mask_idx_entries,
            },
            subcircuits_dir / "mask_idx_map.json",
        )

    # Save summary.json with key statistics
    n_subcircuits = (
        len(first_trial.subcircuits) if first_trial and first_trial.subcircuits else 0
    )
    summary = {
        "n_subcircuits": n_subcircuits,
        "n_trials": len(result.trials),
        "epsilon": epsilon,
        "gates": list(gate_subcircuit_data.keys()),
        "best_by_gate": {
            gate: rankings[0] if rankings else None
            for gate, rankings in averaged_rankings.items()
        },
        "n_passing_by_gate": {
            gate: sum(1 for r in rankings if r.get("passes_epsilon", False))
            for gate, rankings in averaged_rankings.items()
        },
    }
    _save_json(summary, subcircuits_dir / "summary.json")

    # Compute rankings by node_mask_idx and edge_mask_idx for diagram naming
    # Uses SUBCIRCUIT_METRICS_RANKING for tuple-based sorting (all metrics considered)
    node_mask_metrics = {}  # node_mask_idx -> {metric_name -> list of values}
    edge_mask_metrics = {}  # edge_mask_idx -> {metric_name -> list of values}

    if first_trial and first_trial.subcircuits:
        # Build comprehensive metrics for each subcircuit (averaged across gates)
        subcircuit_all_metrics = {}  # sc_idx -> {metric_name -> list of values across gates}

        for gate, rankings in averaged_rankings.items():
            for r in rankings:
                sc_idx = r["subcircuit_idx"]
                if sc_idx not in subcircuit_all_metrics:
                    subcircuit_all_metrics[sc_idx] = {m: [] for m in SUBCIRCUIT_METRICS_RANKING}

                # Extract metrics from the ranking entry
                subcircuit_all_metrics[sc_idx]["accuracy"].append(r.get("avg_accuracy"))
                if r.get("observational") and r["observational"].get("avg") is not None:
                    subcircuit_all_metrics[sc_idx]["overall_observational"].append(
                        r["observational"]["avg"]
                    )
                if r.get("interventional") and r["interventional"].get("avg") is not None:
                    subcircuit_all_metrics[sc_idx]["overall_interventional"].append(
                        r["interventional"]["avg"]
                    )
                if r.get("counterfactual") and r["counterfactual"].get("avg") is not None:
                    subcircuit_all_metrics[sc_idx]["overall_counterfactual"].append(
                        r["counterfactual"]["avg"]
                    )

        # Average each metric across gates for each subcircuit
        def _avg_valid(values):
            valid = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
            return sum(valid) / len(valid) if valid else None

        subcircuit_avg_metrics = {}
        for sc_idx, metrics_dict in subcircuit_all_metrics.items():
            subcircuit_avg_metrics[sc_idx] = {
                name: _avg_valid(values) for name, values in metrics_dict.items()
            }

        # Map to node_pattern and edge_variation, collecting metrics
        for entry in mask_idx_entries:
            sc_idx = entry["subcircuit_idx"]
            nm_idx = entry["node_pattern"]
            em_idx = entry["edge_variation"]
            metrics = subcircuit_avg_metrics.get(sc_idx, {})

            if nm_idx not in node_mask_metrics:
                node_mask_metrics[nm_idx] = {m: [] for m in SUBCIRCUIT_METRICS_RANKING}
            for name in SUBCIRCUIT_METRICS_RANKING:
                val = metrics.get(name)
                if val is not None:
                    node_mask_metrics[nm_idx][name].append(val)

            if em_idx not in edge_mask_metrics:
                edge_mask_metrics[em_idx] = {m: [] for m in SUBCIRCUIT_METRICS_RANKING}
            for name in SUBCIRCUIT_METRICS_RANKING:
                val = metrics.get(name)
                if val is not None:
                    edge_mask_metrics[em_idx][name].append(val)

        # Compute best/avg for each node_pattern (best across edge variations)
        node_mask_rankings = []
        for nm_idx, metrics_dict in node_mask_metrics.items():
            ranking_entry = {"node_pattern": nm_idx}
            for name in SUBCIRCUIT_METRICS_RANKING:
                values = metrics_dict.get(name, [])
                valid = [v for v in values if v is not None]
                if valid:
                    ranking_entry[f"best_{name}"] = max(valid)
                    ranking_entry[f"avg_{name}"] = sum(valid) / len(valid)
                else:
                    ranking_entry[f"best_{name}"] = None
                    ranking_entry[f"avg_{name}"] = None
            node_mask_rankings.append(ranking_entry)

        # Sort using tuple of best values for each ranking metric
        def _node_sort_key(entry):
            return tuple(entry.get(f"best_{name}", float("-inf")) for name in SUBCIRCUIT_METRICS_RANKING)
        node_mask_rankings.sort(key=_node_sort_key, reverse=True)

        # Compute avg for each edge_variation (avg across node patterns)
        edge_mask_rankings = []
        for em_idx, metrics_dict in edge_mask_metrics.items():
            ranking_entry = {"edge_variation": em_idx, "n_patterns": 0}
            for name in SUBCIRCUIT_METRICS_RANKING:
                values = metrics_dict.get(name, [])
                valid = [v for v in values if v is not None]
                if valid:
                    ranking_entry[f"avg_{name}"] = sum(valid) / len(valid)
                    ranking_entry["n_patterns"] = max(ranking_entry["n_patterns"], len(valid))
                else:
                    ranking_entry[f"avg_{name}"] = None
            edge_mask_rankings.append(ranking_entry)

        # Sort using tuple of avg values for each ranking metric
        def _edge_sort_key(entry):
            return tuple(entry.get(f"avg_{name}", float("-inf")) for name in SUBCIRCUIT_METRICS_RANKING)
        edge_mask_rankings.sort(key=_edge_sort_key, reverse=True)
    else:
        node_mask_rankings = []
        edge_mask_rankings = []

    # Generate circuit diagrams (T1.h)
    _generate_circuit_diagrams(
        first_trial, subcircuits_dir, node_mask_rankings, edge_mask_rankings
    )

    # Save explanation
    _save_explanation(subcircuits_dir / "explanation.md", SUBCIRCUITS_EXPLANATION)


def _generate_circuit_diagrams(
    trial,
    subcircuits_dir: Path,
    node_mask_rankings: list = None,
    edge_mask_rankings: list = None,
):
    """Generate circuit diagrams for all subcircuits with rankings in filenames.

    Creates:
        circuit_diagrams/
            ranked_node_masks/rank{N:02d}_node{idx}.png       - ranked by best metrics across edge variations
            ranked_edge_masks/rank{N:02d}_edge{idx}.png       - ranked by avg metrics across node patterns
            ranked_subcircuit_masks/rank{N:02d}_sc{idx}.png   - all subcircuits ranked by node pattern rank
            ranking_metrics.json                               - metrics used for ranking
            node_rankings.json                                 - node pattern rankings
            edge_rankings.json                                 - edge variation rankings
    """
    from src.circuit import Circuit

    if not trial or not trial.subcircuits:
        return

    diagrams_dir = subcircuits_dir / "circuit_diagrams"
    ranked_node_masks_dir = diagrams_dir / "ranked_node_masks"
    ranked_edge_masks_dir = diagrams_dir / "ranked_edge_masks"
    ranked_subcircuit_masks_dir = diagrams_dir / "ranked_subcircuit_masks"

    ranked_node_masks_dir.mkdir(parents=True, exist_ok=True)
    ranked_edge_masks_dir.mkdir(parents=True, exist_ok=True)
    ranked_subcircuit_masks_dir.mkdir(parents=True, exist_ok=True)

    # Build ranking lookup: node_mask_idx -> rank (0-indexed)
    node_mask_rank = {}
    if node_mask_rankings:
        for rank, r in enumerate(node_mask_rankings):
            node_mask_rank[r["node_pattern"]] = rank

    edge_mask_rank = {}
    if edge_mask_rankings:
        for rank, r in enumerate(edge_mask_rankings):
            edge_mask_rank[r["edge_variation"]] = rank

    # Track unique node patterns and subcircuit info for ranking
    seen_node_patterns = {}  # hidden_pattern -> (node_mask_idx, circuit)
    node_mask_idx_counter = 0
    subcircuit_info = []  # [(subcircuit_idx, node_mask_idx, circuit), ...]

    max_diagrams = min(48, len(trial.subcircuits))

    for sc_data in trial.subcircuits[:max_diagrams]:
        subcircuit_idx = sc_data.get("idx", 0)
        try:
            circuit = Circuit.from_dict(sc_data)

            # Track unique node patterns for node_masks/
            node_masks = sc_data.get("node_masks", [])
            hidden_masks = (
                tuple(tuple(m) for m in node_masks[1:-1]) if len(node_masks) > 2 else ()
            )

            if hidden_masks not in seen_node_patterns:
                seen_node_patterns[hidden_masks] = (node_mask_idx_counter, circuit)
                node_mask_idx_counter += 1

            # Get node_mask_idx for this subcircuit
            nm_idx = seen_node_patterns[hidden_masks][0]
            subcircuit_info.append((subcircuit_idx, nm_idx, circuit))
        except Exception:
            pass  # Silently skip diagram generation failures

    # Generate ranked_subcircuit_masks PNGs with ranking prefix (based on node rank)
    for subcircuit_idx, nm_idx, circuit in subcircuit_info:
        rank = node_mask_rank.get(nm_idx, 99)
        sc_path = ranked_subcircuit_masks_dir / f"rank{rank:02d}_sc{subcircuit_idx}.png"
        try:
            circuit.visualize(file_path=str(sc_path), node_size="small")
        except Exception:
            pass

    # Generate ranked_node_masks PNGs with ranking prefix
    for hidden_pattern, (nm_idx, circuit) in seen_node_patterns.items():
        rank = node_mask_rank.get(nm_idx, 99)
        nm_path = ranked_node_masks_dir / f"rank{rank:02d}_node{nm_idx}.png"
        try:
            circuit.visualize(file_path=str(nm_path), node_size="small")
        except Exception:
            pass

    # Generate edge_masks PNGs with ranking prefix
    # Use edge_mask_idx from the circuit mapping, not unique edge patterns
    # With full_edges_only=True, edge_mask_idx is always 0
    seen_edge_mask_idx = {}  # edge_mask_idx -> circuit (first one for visualization)
    edge_counts_local = {}  # node_mask_idx -> count (to track edge_mask_idx)

    for sc_data in trial.subcircuits[:max_diagrams]:
        node_masks = sc_data.get("node_masks", [])
        hidden_masks = (
            tuple(tuple(m) for m in node_masks[1:-1]) if len(node_masks) > 2 else ()
        )

        # Compute node_mask_idx for this pattern
        nm_idx = None
        for hp, (idx, _) in seen_node_patterns.items():
            if hp == hidden_masks:
                nm_idx = idx
                break

        if nm_idx is not None:
            # Compute edge_mask_idx (how many times we've seen this node pattern)
            if nm_idx not in edge_counts_local:
                edge_counts_local[nm_idx] = 0
            em_idx = edge_counts_local[nm_idx]
            edge_counts_local[nm_idx] += 1

            # Store first circuit for each unique edge_mask_idx
            if em_idx not in seen_edge_mask_idx:
                try:
                    circuit = Circuit.from_dict(sc_data)
                    seen_edge_mask_idx[em_idx] = circuit
                except Exception:
                    pass

    for em_idx, circuit in seen_edge_mask_idx.items():
        rank = edge_mask_rank.get(em_idx, 99)
        em_path = ranked_edge_masks_dir / f"rank{rank:02d}_edge{em_idx}.png"
        try:
            circuit.visualize(file_path=str(em_path), node_size="small")
        except Exception:
            pass

    # Save rankings as separate JSON files
    _save_json(
        {
            "metrics": SUBCIRCUIT_METRICS_RANKING,
            "description": (
                "Metrics used for tuple-based ranking, in priority order. "
                "Higher values are better for all metrics."
            ),
        },
        diagrams_dir / "ranking_metrics.json",
    )

    _save_json(
        {
            "rankings": node_mask_rankings or [],
            "description": "Node patterns ranked by best metrics across edge variations. Files: rank{N}_node{idx}.png",
        },
        diagrams_dir / "node_rankings.json",
    )

    _save_json(
        {
            "rankings": edge_mask_rankings or [],
            "description": "Edge variations ranked by avg metrics across node patterns. Files: rank{N}_edge{idx}.png",
        },
        diagrams_dir / "edge_rankings.json",
    )


SUBCIRCUITS_EXPLANATION = """# Subcircuits Analysis

## Structure

- `subcircuit_score_ranking.json`: Rankings by gate with faithfulness scores
- `subcircuit_score_ranking_per_trial.json`: Per-trial granular rankings
- `mask_idx_map.json`: (node_pattern, edge_variation) → subcircuit_idx mapping
- `circuit_diagrams/`: Visual diagrams
  - `ranking_metrics.json`: Metrics used for ranking (in priority order)
  - `node_rankings.json`: Node pattern rankings with all metrics
  - `edge_rankings.json`: Edge variation rankings with all metrics
  - `ranked_node_masks/rank{N}_node{idx}.png`: Node pattern diagrams
  - `ranked_edge_masks/rank{N}_edge{idx}.png`: Edge variation diagrams
  - `ranked_subcircuit_masks/rank{N}_sc{idx}.png`: Full subcircuit diagrams

## Index Mapping

subcircuit_idx = f(node_mask_idx, edge_mask_idx)

- **node_mask_idx**: Identifies which hidden nodes are active
- **edge_mask_idx**: Identifies edge configuration for that node pattern
- With full_edges_only=True, typically edge_mask_idx=0

## Key Metrics

For subcircuit S_i and gate G:

- **accuracy**: Pr[S_i(x) = G(x)] — behavioral match
- **observational**: Robustness under input perturbations
- **interventional**: Faithfulness under activation patching
- **counterfactual**: Necessity/sufficiency scores
- **node_sparsity**: |active hidden nodes| / |total hidden nodes|
- **edge_sparsity**: |active edges| / |total possible edges|
- **passes_epsilon**: accuracy ≥ 1 - ε

## Reading Files

`rankings_by_gate[gate]` in `subcircuit_score_ranking.json` lists
subcircuits sorted by avg_accuracy. Each entry includes all faithfulness
scores averaged across trials.
"""


def _save_gate_summaries(trial, trial_dir: Path):
    """Save per-gate summary.json, explanation.md, and samples files."""
    # Get epsilon from trial setup
    epsilon = 0.2
    if hasattr(trial, "setup") and hasattr(trial.setup, "constraints"):
        epsilon = getattr(trial.setup.constraints, "epsilon", 0.2)

    gates_dir = trial_dir / "gates"
    gates_dir.mkdir(parents=True, exist_ok=True)

    # Get faithfulness results keyed by gate
    bests_keys = trial.metrics.per_gate_bests or {}
    bests_faith = trial.metrics.per_gate_bests_faith or {}

    for gate_name, gate_metrics in trial.metrics.per_gate_metrics.items():
        gate_dir = gates_dir / gate_name
        gate_dir.mkdir(parents=True, exist_ok=True)

        gate_summary = _generate_gate_summary(gate_name, gate_metrics, epsilon)
        _save_json(gate_summary, gate_dir / "summary.json")
        _save_explanation(gate_dir / "explanation.md", GATE_EXPLANATION)

        # Save samples for each subcircuit (T1.i)
        gate_keys = bests_keys.get(gate_name, [])
        gate_faith = bests_faith.get(gate_name, [])
        sc_metrics = getattr(gate_metrics, "subcircuit_metrics", [])
        _save_subcircuit_samples(gate_dir, gate_keys, gate_faith, sc_metrics)


def _save_subcircuit_samples(
    gate_dir: Path, keys: list, faith_results: list, sc_metrics_list: list = None
):
    """Save samples in folder structure per T1.i.

    Args:
        gate_dir: Path to gate directory
        keys: List of (node_mask_idx, edge_mask_idx) keys
        faith_results: List of FaithfulnessMetrics objects
        sc_metrics_list: List of SubcircuitMetrics objects for ranking

    Creates:
        {node_mask_idx}/{edge_mask_idx}/
            faithfulness/
                observational/
                    noise_perturbations/samples.json
                    out_distribution_transformations/samples.json
                interventional/
                    in_circuit/in_distribution/samples.json
                    in_circuit/out_distribution/samples.json
                    out_circuit/in_distribution/samples.json
                    out_circuit/out_distribution/samples.json
                counterfactual/
                    sufficiency/samples.json
                    completeness/samples.json
                    necessity/samples.json
                    independence/samples.json
    """
    from dataclasses import asdict

    # Build index from sc_metrics_list by idx
    sc_metrics_by_idx = {}
    if sc_metrics_list:
        for sm in sc_metrics_list:
            sc_metrics_by_idx[sm.idx] = sm

    for key, faith in zip(keys, faith_results):
        if faith is None:
            continue

        # Extract node_mask_idx and edge_mask_idx from key
        if isinstance(key, (tuple, list)) and len(key) >= 2:
            node_mask_idx, edge_mask_idx = key[0], key[1]
        else:
            node_mask_idx, edge_mask_idx = key, 0

        # Get corresponding SubcircuitMetrics
        sc_metrics = sc_metrics_by_idx.get(node_mask_idx)

        # Get ranked metrics for this subcircuit
        ranked = filter_and_rank_subcircuit_metrics(sc_metrics, faithfulness_data=faith)

        sc_dir = gate_dir / str(node_mask_idx) / str(edge_mask_idx)

        # Save subcircuit-level summary with ranked metrics
        sc_summary = {
            "node_pattern": node_mask_idx,
            "edge_variation": edge_mask_idx,
            "ranked_metrics": ranked,
            "subfolders": ["faithfulness"],
        }
        sc_dir.mkdir(parents=True, exist_ok=True)
        _save_json(sc_summary, sc_dir / "summary.json")

        # Create faithfulness directory to contain all faithfulness data
        faith_dir = sc_dir / "faithfulness"
        faith_dir.mkdir(parents=True, exist_ok=True)

        # Faithfulness summary
        faith_summary = {
            "subfolders": ["observational", "interventional", "counterfactual"],
        }

        # Observational samples
        if hasattr(faith, "observational") and faith.observational:
            obs = faith.observational
            obs_dir = faith_dir / "observational"
            obs_dir.mkdir(parents=True, exist_ok=True)

            # Observational summary
            obs_summary = {
                "overall_observational": getattr(obs, "overall_observational", None),
                "ranked_metrics": ranked,  # Include ranked metrics
                "subfolders": [
                    "noise_perturbations",
                    "out_distribution_transformations",
                ],
            }

            # Noise perturbations
            if hasattr(obs, "noise") and obs.noise:
                noise_dir = obs_dir / "noise_perturbations"
                noise_dir.mkdir(parents=True, exist_ok=True)
                obs_summary["noise_agreement_bit"] = getattr(
                    obs.noise, "agreement_bit", None
                )
                obs_summary["noise_agreement_logit"] = getattr(
                    obs.noise, "agreement_logit", None
                )
                if hasattr(obs.noise, "samples") and obs.noise.samples:
                    samples = [asdict(s) for s in obs.noise.samples]
                    _save_json(
                        {"samples": samples, "n": len(samples)},
                        noise_dir / "samples.json",
                    )

            # OOD transformations
            if hasattr(obs, "ood") and obs.ood:
                ood_dir = obs_dir / "out_distribution_transformations"
                ood_dir.mkdir(parents=True, exist_ok=True)
                obs_summary["ood_overall"] = getattr(obs.ood, "overall_agreement", None)
                if hasattr(obs.ood, "samples") and obs.ood.samples:
                    samples = [asdict(s) for s in obs.ood.samples]
                    _save_json(
                        {"samples": samples, "n": len(samples)},
                        ood_dir / "samples.json",
                    )

            _save_json(obs_summary, obs_dir / "summary.json")

        # Interventional samples
        if hasattr(faith, "interventional") and faith.interventional:
            intv = faith.interventional
            intv_dir = faith_dir / "interventional"
            intv_dir.mkdir(parents=True, exist_ok=True)

            # Interventional summary
            intv_summary = {
                "overall_interventional": getattr(intv, "overall_interventional", None),
                "mean_in_circuit_effect": getattr(intv, "mean_in_circuit_effect", None),
                "mean_out_circuit_effect": getattr(
                    intv, "mean_out_circuit_effect", None
                ),
                "ranked_metrics": ranked,  # Include ranked metrics
                "subfolders": ["in_circuit", "out_circuit"],
            }

            # In-circuit folder
            in_circuit_dir = intv_dir / "in_circuit"
            in_circuit_dir.mkdir(parents=True, exist_ok=True)
            in_circuit_summary = {
                "ranked_metrics": ranked,
                "subfolders": ["in_distribution", "out_distribution"],
            }

            if hasattr(intv, "in_circuit_stats") and intv.in_circuit_stats:
                in_in_dir = in_circuit_dir / "in_distribution"
                in_in_dir.mkdir(parents=True, exist_ok=True)
                _save_json({"stats": intv.in_circuit_stats}, in_in_dir / "samples.json")
                in_circuit_summary["in_distribution_n_patches"] = len(
                    intv.in_circuit_stats
                )

            if hasattr(intv, "in_circuit_stats_ood") and intv.in_circuit_stats_ood:
                in_out_dir = in_circuit_dir / "out_distribution"
                in_out_dir.mkdir(parents=True, exist_ok=True)
                _save_json(
                    {"stats": intv.in_circuit_stats_ood}, in_out_dir / "samples.json"
                )
                in_circuit_summary["out_distribution_n_patches"] = len(
                    intv.in_circuit_stats_ood
                )

            _save_json(in_circuit_summary, in_circuit_dir / "summary.json")

            # Out-circuit folder
            out_circuit_dir = intv_dir / "out_circuit"
            out_circuit_dir.mkdir(parents=True, exist_ok=True)
            out_circuit_summary = {
                "ranked_metrics": ranked,
                "subfolders": ["in_distribution", "out_distribution"],
            }

            if hasattr(intv, "out_circuit_stats") and intv.out_circuit_stats:
                out_in_dir = out_circuit_dir / "in_distribution"
                out_in_dir.mkdir(parents=True, exist_ok=True)
                _save_json(
                    {"stats": intv.out_circuit_stats}, out_in_dir / "samples.json"
                )
                out_circuit_summary["in_distribution_n_patches"] = len(
                    intv.out_circuit_stats
                )

            if hasattr(intv, "out_circuit_stats_ood") and intv.out_circuit_stats_ood:
                out_out_dir = out_circuit_dir / "out_distribution"
                out_out_dir.mkdir(parents=True, exist_ok=True)
                _save_json(
                    {"stats": intv.out_circuit_stats_ood}, out_out_dir / "samples.json"
                )
                out_circuit_summary["out_distribution_n_patches"] = len(
                    intv.out_circuit_stats_ood
                )

            _save_json(out_circuit_summary, out_circuit_dir / "summary.json")
            _save_json(intv_summary, intv_dir / "summary.json")

        # Counterfactual samples
        if hasattr(faith, "counterfactual") and faith.counterfactual:
            cf = faith.counterfactual
            cf_dir = faith_dir / "counterfactual"
            cf_dir.mkdir(parents=True, exist_ok=True)

            # Counterfactual summary (2x2 matrix)
            cf_summary = {
                "overall_counterfactual": getattr(cf, "overall_counterfactual", None),
                "mean_sufficiency": getattr(cf, "mean_sufficiency", None),
                "mean_completeness": getattr(cf, "mean_completeness", None),
                "mean_necessity": getattr(cf, "mean_necessity", None),
                "mean_independence": getattr(cf, "mean_independence", None),
                "ranked_metrics": ranked,  # Include ranked metrics
                "subfolders": [
                    "sufficiency",
                    "completeness",
                    "necessity",
                    "independence",
                ],
            }

            # Sufficiency
            if hasattr(cf, "sufficiency_effects") and cf.sufficiency_effects:
                suff_dir = cf_dir / "sufficiency"
                suff_dir.mkdir(parents=True, exist_ok=True)
                samples = [asdict(e) for e in cf.sufficiency_effects]
                _save_json(
                    {"samples": samples, "n": len(samples)}, suff_dir / "samples.json"
                )
                cf_summary["sufficiency_n_samples"] = len(samples)

            # Completeness
            if hasattr(cf, "completeness_effects") and cf.completeness_effects:
                comp_dir = cf_dir / "completeness"
                comp_dir.mkdir(parents=True, exist_ok=True)
                samples = [asdict(e) for e in cf.completeness_effects]
                _save_json(
                    {"samples": samples, "n": len(samples)}, comp_dir / "samples.json"
                )
                cf_summary["completeness_n_samples"] = len(samples)

            # Necessity
            if hasattr(cf, "necessity_effects") and cf.necessity_effects:
                nec_dir = cf_dir / "necessity"
                nec_dir.mkdir(parents=True, exist_ok=True)
                samples = [asdict(e) for e in cf.necessity_effects]
                _save_json(
                    {"samples": samples, "n": len(samples)}, nec_dir / "samples.json"
                )
                cf_summary["necessity_n_samples"] = len(samples)

            # Independence
            if hasattr(cf, "independence_effects") and cf.independence_effects:
                ind_dir = cf_dir / "independence"
                ind_dir.mkdir(parents=True, exist_ok=True)
                samples = [asdict(e) for e in cf.independence_effects]
                _save_json(
                    {"samples": samples, "n": len(samples)}, ind_dir / "samples.json"
                )
                cf_summary["independence_n_samples"] = len(samples)

            _save_json(cf_summary, cf_dir / "summary.json")

        # Save faithfulness summary
        _save_json(faith_summary, faith_dir / "summary.json")
