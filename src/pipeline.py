"""Pipeline orchestration for experiment execution.

Provides two modes:
- Monolith: Run all trials, save once at the end
- Iterative: Run and save each trial individually (for long experiments)
"""

from pathlib import Path
from typing import Any, Iterator

from src.experiment import experiment_run, run_experiment
from src.experiment_config import ExperimentConfig
from src.infra import print_profile, profile_fn
from src.infra.profiler import Trace
from src.persistence import load_results, save_results, save_training_data
from src.schemas import ExperimentResult, TrialResult
from src.spd import (
    SpdResults,
    load_spd_results,
    run_spd,
    run_trial_on_spd_results,
    save_spd_results,
)
from src.viz import save_per_gate_data, visualize_experiment, visualize_spd_experiment
from src.viz.viz_config import VizConfig, VizLevel


@profile_fn("do_viz_on_experiment")
def do_viz_on_experiment(
    result: ExperimentResult,
    run_dir: str,
    spd: bool,
    viz_config: VizConfig | None = None,
) -> None:
    """Run visualization on a complete experiment result."""
    if viz_config is None:
        viz_config = VizConfig()

    # ALWAYS save per-gate JSON data (independent of viz level)
    # Pass viz_config to control circuit.png generation
    save_per_gate_data(result, run_dir, viz_config=viz_config)

    # Skip PNG visualization if level is NONE
    if viz_config.skip_all_viz:
        return

    visualize_experiment(result, run_dir=run_dir, viz_config=viz_config)
    if spd:
        spd_result = load_spd_results(run_dir)
        if spd_result:
            visualize_spd_experiment(spd_result, run_dir=run_dir)


@profile_fn("do_spd_on_experiment")
def do_spd_on_experiment(
    result: ExperimentResult, run_dir: str, viz: bool, spd_device: str
) -> SpdResults:
    """Run SPD decomposition on a complete experiment result."""
    spd_result = run_spd(result, run_dir=run_dir, device=spd_device)
    save_spd_results(spd_result, run_dir=run_dir)

    # Run trial analysis on SPD-discovered subcircuits
    run_trial_on_spd_results(spd_result, result, run_dir=run_dir, device=spd_device)

    if viz:
        visualize_spd_experiment(spd_result, run_dir=run_dir)
    return spd_result


def do_viz_on_trial(
    trial_result: TrialResult,
    run_dir: str,
    spd: bool,
    viz_config: VizConfig | None = None,
) -> None:
    """Run visualization on a single trial result.

    Creates a temporary ExperimentResult wrapper for compatibility with existing viz.
    """
    if viz_config is None:
        viz_config = VizConfig()

    # Wrap trial in experiment result for viz compatibility
    from src.experiment_config import ExperimentConfig
    temp_result = ExperimentResult(config=ExperimentConfig())
    temp_result.trials[trial_result.trial_id] = trial_result

    # ALWAYS save per-gate JSON data (independent of viz level)
    # Pass viz_config to control circuit.png generation
    save_per_gate_data(temp_result, run_dir, viz_config=viz_config)

    # Skip PNG visualization if level is NONE
    if viz_config.skip_all_viz:
        return

    visualize_experiment(temp_result, run_dir=run_dir, viz_config=viz_config)
    if spd:
        spd_result = load_spd_results(run_dir)
        if spd_result:
            visualize_spd_experiment(spd_result, run_dir=run_dir)


def do_spd_on_trial(
    trial_result: TrialResult, run_dir: str, viz: bool, spd_device: str
) -> SpdResults | None:
    """Run SPD decomposition on a single trial result."""
    # Wrap trial in experiment result for SPD compatibility
    from src.experiment_config import ExperimentConfig
    temp_result = ExperimentResult(config=ExperimentConfig())
    temp_result.trials[trial_result.trial_id] = trial_result

    spd_result = run_spd(temp_result, run_dir=run_dir, device=spd_device)
    save_spd_results(spd_result, run_dir=run_dir)

    run_trial_on_spd_results(spd_result, temp_result, run_dir=run_dir, device=spd_device)

    if viz:
        visualize_spd_experiment(spd_result, run_dir=run_dir)
    return spd_result


def save_experiment_results(result: ExperimentResult, run_dir: str) -> None:
    """Save complete experiment results."""
    save_results(result, run_dir=run_dir)


def save_trial_result(trial_result: TrialResult, run_dir: str, cfg: ExperimentConfig) -> None:
    """Save a single trial result incrementally.

    OPTIMIZED: Saves only this trial's data without loading all previous trials.
    Run-level summaries are regenerated at the end of the experiment.
    """
    from src.persistence.save import save_single_trial

    run_dir = Path(run_dir)

    # Save just this trial (no O(n^2) loading of all previous trials)
    save_single_trial(trial_result, run_dir, cfg)


def print_experiment_summary(
    experiment_result: ExperimentResult, spd_result: SpdResults | None, logger: Any
) -> None:
    """Print summary of experiment results."""
    logger.info("\n\n\n\n")
    logger.info("experiment_result")
    logger.info("\n\n\n\n")
    summary = experiment_result.print_summary()
    logger.info(summary)
    if spd_result:
        logger.info("\n\n\n\n")
        logger.info("spd_result")
        logger.info("\n\n\n\n")
        spd_summary = spd_result.print_summary()
        logger.info(spd_summary)


@profile_fn("run_monolith")
def run_monolith(
    cfg: ExperimentConfig,
    run_dir: str,
    logger: Any,
    spd: bool = False,
    viz_config: VizConfig | None = None,
    spd_device: str = "cpu",
) -> ExperimentResult:
    """Run all trials at once, save at the end.

    Best for small experiments where total runtime is short.
    """
    if viz_config is None:
        viz_config = VizConfig()

    experiment_result, master_data = run_experiment(cfg, logger=logger)
    save_training_data(master_data, run_dir, gate_names=cfg.target_logic_gates)
    save_experiment_results(experiment_result, run_dir=run_dir)

    spd_result = None
    if spd:
        spd_result = do_spd_on_experiment(
            experiment_result, run_dir, viz=not viz_config.skip_all_viz, spd_device=spd_device
        )
    if not viz_config.skip_all_viz:
        do_viz_on_experiment(experiment_result, run_dir, spd=spd, viz_config=viz_config)

    print_experiment_summary(experiment_result, spd_result, logger)
    return experiment_result


@profile_fn("run_iteratively")
def run_iteratively(
    cfg: ExperimentConfig,
    run_dir: str,
    logger: Any,
    spd: bool = False,
    viz_config: VizConfig | None = None,
    spd_device: str = "cpu",
) -> ExperimentResult:
    """Run trials one at a time, saving after each.

    Best for long experiments where you want incremental saves.
    Results are saved after each trial completes, so partial progress
    is preserved if the experiment is interrupted.

    MEMORY OPTIMIZED: Trials are saved and then cleared from memory.
    Final result is loaded from disk at the end.
    """
    import gc

    if viz_config is None:
        viz_config = VizConfig()

    trial_iterator, master_data = experiment_run(cfg, logger=logger)
    save_training_data(master_data, run_dir, gate_names=cfg.target_logic_gates)

    trial_count = 0
    for trial_result in trial_iterator:
        trial_count += 1

        # Save this trial incrementally (without loading all previous trials)
        save_trial_result(trial_result, run_dir, cfg)

        # Optionally run SPD per trial
        if spd:
            do_spd_on_trial(trial_result, run_dir, viz=not viz_config.skip_all_viz, spd_device=spd_device)

        # ALWAYS run viz (saves JSON data regardless of viz level, only skips PNGs)
        do_viz_on_trial(trial_result, run_dir, spd=spd, viz_config=viz_config)

        # Clear trial from memory after processing
        del trial_result
        gc.collect()

    # Load final results from disk for summary (memory-efficient: only loads once)
    logger.info(f"Loading {trial_count} trials from disk for final summary...")
    experiment_result = load_results(str(run_dir))

    # Generate run-level summaries only (trials already saved individually)
    from src.persistence.save import save_run_summaries_only
    save_run_summaries_only(experiment_result, run_dir=run_dir, logger=logger)

    # Final summary
    spd_result = load_spd_results(run_dir) if spd else None
    print_experiment_summary(experiment_result, spd_result, logger)
    return experiment_result


@profile_fn("regenerate_from_models")
def regenerate_from_models(
    run_dir: str,
    device: str = "cpu",
    faith_config: Any = None,
    viz_config: VizConfig | None = None,
    trial_filter: str | None = None,
) -> None:
    """Regenerate faithfulness analysis from saved models.

    Loads models from disk and re-runs faithfulness analysis for each trial's
    best subcircuits. Useful for regenerating results after fixing bugs in
    the analysis pipeline.

    Args:
        run_dir: Path to the run directory (e.g., "runs/counter")
        device: Device for computation
        faith_config: FaithfulnessConfig controlling which analyses to run
        viz_config: VizConfig for visualization
        trial_filter: Optional trial ID to process only one trial
    """
    import json
    import torch
    from pathlib import Path

    from src.analysis.counterfactual import create_canonical_counterfactual_pairs
    from src.analysis.faithfulness import calculate_faithfulness_metrics
    from src.analysis.observational import calculate_observational_metrics
    from src.circuit import Circuit
    from src.domain import resolve_gate, generate_canonical_inputs
    from src.experiment_config import FaithfulnessConfig
    from src.model import MLP
    from src.viz.faithfulness_viz import (
        visualize_faithfulness_intervention_effects,
        visualize_faithfulness_circuit_samples,
    )
    from src.viz.observational_viz import (
        visualize_observational_circuits,
        visualize_observational_curves,
    )
    from src.viz.export import save_all_samples

    if faith_config is None:
        faith_config = FaithfulnessConfig()
    if viz_config is None:
        viz_config = VizConfig()

    run_path = Path(run_dir)
    trials_dir = run_path / "trials"

    if not trials_dir.exists():
        print(f"No trials directory found in {run_dir}")
        return

    # Get trial directories
    trial_dirs = [d for d in trials_dir.iterdir() if d.is_dir()]
    if trial_filter:
        trial_dirs = [d for d in trial_dirs if d.name == trial_filter]

    if not trial_dirs:
        print(f"No matching trials found in {run_dir}")
        return

    print(f"Found {len(trial_dirs)} trial(s) to regenerate")

    for trial_dir in sorted(trial_dirs):
        trial_id = trial_dir.name
        print(f"\n{'=' * 60}")
        print(f"Processing trial: {trial_id}")
        print("=" * 60)

        # Load the model
        model_path = trial_dir / "all_gates" / "model.pt"
        if not model_path.exists():
            print(f"  No model found at {model_path}, skipping")
            continue

        print(f"  Loading model from {model_path}")
        model = MLP.load_from_file(str(model_path), device=device)

        # Get layer weights for visualization
        layer_weights = [layer[0].weight.detach() for layer in model.layers]
        layer_biases = [layer[0].bias.detach() for layer in model.layers]

        # Load metrics.json to get circuit data
        metrics_path = trial_dir / "metrics.json"
        if not metrics_path.exists():
            print(f"  No metrics.json found, skipping")
            continue

        with open(metrics_path) as f:
            metrics = json.load(f)

        per_gate_circuits = metrics.get("per_gate_circuits", {})

        # Process each gate
        for gate_name, gate_circuits in per_gate_circuits.items():
            print(f"\n  Gate: {gate_name}")

            # Resolve gate for ground truth
            logic_gate = resolve_gate(gate_name)
            gate_idx = list(per_gate_circuits.keys()).index(gate_name)

            def ground_truth_fn(x):
                # Convert tensor to numpy for gate_fn, then back to tensor
                inp_np = x.cpu().numpy()
                result = logic_gate.gate_fn(inp_np)
                return torch.tensor(result, device=x.device, dtype=x.dtype)

            # Create counterfactual pairs
            print(f"    Creating counterfactual pairs...")
            pairs = create_canonical_counterfactual_pairs(
                model=model,
                gate_idx=gate_idx,
                n_inputs=logic_gate.n_inputs,
                ground_truth_fn=ground_truth_fn,
                device=device,
            )

            # Compute activations on canonical inputs
            canonical_inputs = generate_canonical_inputs(logic_gate.n_inputs, device)
            x_canonical = torch.cat([inp for inp in canonical_inputs.values()], dim=0)

            with torch.inference_mode():
                activations = model(x_canonical, return_activations=True)
                y = activations[-1]

            # Canonical activations dict for observational viz
            canonical_activations = {}
            idx = 0
            for label in canonical_inputs.keys():
                canonical_activations[label] = [a[idx:idx+1].detach() for a in activations]
                idx += 1

            # Find gate directory
            gate_dir = trial_dir / gate_name
            if not gate_dir.exists():
                print(f"    No gate directory {gate_dir}, skipping")
                continue

            # Process each subcircuit directory
            subcircuit_dirs = [d for d in gate_dir.iterdir() if d.is_dir() and d.name.startswith("node")]
            print(f"    Found {len(subcircuit_dirs)} node pattern directories")

            for sc_dir in sorted(subcircuit_dirs):
                # Check for "all" variant (all edges)
                all_dir = sc_dir / "all"
                if not all_dir.exists():
                    continue

                # Get subcircuit index from summary.json
                summary_path = all_dir / "summary.json"
                if not summary_path.exists():
                    continue

                with open(summary_path) as f:
                    summary = json.load(f)

                sc_idx = summary.get("subcircuit", {}).get("index")
                if sc_idx is None:
                    continue

                # Get circuit data from metrics
                sc_idx_str = str(sc_idx)
                if sc_idx_str not in gate_circuits:
                    print(f"      {sc_dir.name}/all (idx={sc_idx}): No circuit data, skipping")
                    continue

                circuit_data = gate_circuits[sc_idx_str]
                circuit = Circuit.from_dict(circuit_data)

                print(f"      Processing {sc_dir.name}/all (idx={sc_idx})...")

                # Create subcircuit model
                subcircuit = model.separate_subcircuit(circuit, gate_idx=gate_idx)

                # Get circuit structure for patches
                structure = circuit.analyze_structure()

                # ===== 1. Observational Analysis =====
                observational = None
                if not faith_config.skip_observational:
                    print(f"        Running observational analysis...")
                    observational = calculate_observational_metrics(
                        subcircuit=subcircuit,
                        full_model=model,
                        n_samples_per_base=100,
                        device=device,
                        n_inputs=logic_gate.n_inputs,
                    )

                # ===== 2. Faithfulness Analysis (interventional + counterfactual) =====
                print(f"        Running interventional + counterfactual analysis...")
                internal_config = FaithfulnessConfig(
                    skip_observational=True,  # We handle observational separately
                    skip_interventional=faith_config.skip_interventional,
                    skip_counterfactual=faith_config.skip_counterfactual,
                )

                faithfulness = calculate_faithfulness_metrics(
                    x=x_canonical,
                    y=y,
                    model=model,
                    activations=activations,
                    subcircuit=subcircuit,
                    structure=structure,
                    counterfactual_pairs=pairs,
                    config=internal_config,
                    device=device,
                )

                # Attach observational to faithfulness
                if observational:
                    faithfulness.observational = observational

                # ===== 3. Generate Visualizations =====
                faithfulness_dir = all_dir / "faithfulness"
                faithfulness_dir.mkdir(exist_ok=True)

                print(f"        Saving samples...")
                save_all_samples(str(faithfulness_dir), faithfulness, sc_idx)

                if not viz_config.skip_all_viz:
                    print(f"        Generating visualizations...")

                    # Observational visualizations
                    if observational:
                        obs_dir = faithfulness_dir / "observational"
                        obs_dir.mkdir(exist_ok=True)

                        visualize_observational_curves(
                            observational=observational,
                            output_dir=str(obs_dir),
                            gate_name=f"{gate_name} ({sc_dir.name}/all)",
                        )

                        visualize_observational_circuits(
                            observational=observational,
                            circuit=circuit,
                            layer_weights=layer_weights,
                            output_dir=str(obs_dir),
                            n_samples_per_grid=4,
                            canonical_activations=canonical_activations,
                            layer_biases=layer_biases,
                        )

                    # Intervention effect visualizations
                    visualize_faithfulness_intervention_effects(
                        faithfulness=faithfulness,
                        output_dir=str(faithfulness_dir),
                        gate_name=f"{gate_name} ({sc_dir.name}/all)",
                    )

                    # Circuit sample visualizations
                    visualize_faithfulness_circuit_samples(
                        faithfulness=faithfulness,
                        circuit=circuit,
                        layer_weights=layer_weights,
                        layer_biases=layer_biases,
                        output_dir=str(faithfulness_dir),
                        n_samples_per_grid=4,
                    )

                print(f"        Done with {sc_dir.name}/all")

    print("\n" + "=" * 60)
    print("Regeneration complete!")
    print("=" * 60)
