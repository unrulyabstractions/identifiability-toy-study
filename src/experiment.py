"""
An experiment has many runs. Each run is called a trial.
The experiment as a whole will have an ExperimentConfig.
Each individual trial will have a TrialSetup.
Look at .common.schemas for full definitions.
"""

import copy
import random
from collections.abc import Iterator
from itertools import combinations, product

from src.circuit.precompute import precompute_circuits_for_architectures
from src.domain import get_max_n_inputs, normalize_gate_names
from src.experiment_config import ExperimentConfig, TrialSetting, TrialSetup
from src.infra import parallel_map
from src.infra.profiler import trace
from src.schemas import ExperimentResult, TrialData, TrialResult
from src.training import generate_trial_data

from .trial import run_trial


def build_gate_combinations(
    target_gates: list[str],
    num_gates_per_run: int | list[int] | None = None,
) -> list[tuple[str, ...]]:
    """Build shuffled list of gate combinations for each requested size."""
    if num_gates_per_run is None:
        return [tuple(target_gates)]

    sizes = (
        num_gates_per_run
        if isinstance(num_gates_per_run, list)
        else [num_gates_per_run]
    )
    combos = [
        combo
        for size in sizes
        if size <= len(target_gates)
        for combo in combinations(target_gates, size)
    ]

    if not combos:
        return [tuple(target_gates)]

    random.shuffle(combos)
    return combos


def build_trial_settings(
    cfg: ExperimentConfig, logger=None
) -> tuple[list[TrialSetting], TrialData]:
    """Build all trial settings for an experiment.

    Generates training data ONCE for all possible gates, then each trial
    adapts the data by selecting appropriate output columns.

    Returns:
        Tuple of (trial_settings, master_data) where:
        - trial_settings: List of TrialSetting objects
        - master_data: TrialData for ALL gates (trials select columns from this)
    """
    # Normalize gate names to handle duplicates (e.g., ["XOR", "XOR"] -> ["XOR", "XOR_2"])
    normalized_gates = normalize_gate_names(cfg.target_logic_gates)
    logger and logger.info(f"Normalized gates: {normalized_gates}")

    # Build mapping from gate name to column index in master data
    gate_to_index = {gate: idx for idx, gate in enumerate(normalized_gates)}

    gates_combinations = build_gate_combinations(
        normalized_gates, cfg.num_gates_per_run
    )

    # Determine input size from gates - use max across all gates to support mixed sizes
    input_size = get_max_n_inputs(normalized_gates)

    # Determine output size for circuits (max gates across all combinations)
    output_size = max(len(combo) for combo in gates_combinations)

    # Generate data ONCE for ALL gates
    logger and logger.info("\nGenerating data for all gates...")
    data_params = copy.deepcopy(cfg.base_trial.data_params)
    master_data = generate_trial_data(
        data_params, normalized_gates, cfg.device, logger=logger, debug=cfg.debug
    )

    # Pre-compute circuits for all architectures
    logger and logger.info("\nPre-computing circuits for all architectures...")
    circuits_cache = precompute_circuits_for_architectures(
        cfg.widths, cfg.depths, input_size, output_size, logger
    )

    # Collect all trial settings
    trial_settings = []

    logger and logger.info(
        f"\nBuilding {cfg.num_runs} runs, each with {len(gates_combinations)} gate combinations\n"
    )
    for seed_offset in range(cfg.num_runs):
        for logic_gates in gates_combinations:
            # Compute which columns this trial needs from master data
            gate_indices = [gate_to_index[gate] for gate in logic_gates]

            for width, depth, activation, lr in product(
                cfg.widths, cfg.depths, cfg.activations, cfg.learning_rates
            ):
                # Each trial should have its independent set of params
                model_params = copy.deepcopy(cfg.base_trial.model_params)
                train_params = copy.deepcopy(cfg.base_trial.train_params)
                constraints = copy.deepcopy(cfg.base_trial.constraints)

                train_params.learning_rate = lr
                model_params.width = width
                model_params.depth = depth
                model_params.activation = activation
                model_params.logic_gates = list(logic_gates)

                trial_setup = TrialSetup(
                    seed=cfg.base_trial.seed + seed_offset,
                    data_params=data_params,
                    model_params=model_params,
                    train_params=train_params,
                    constraints=constraints,
                )

                # Get pre-computed circuits for this architecture
                subcircuits, subcircuit_structures = circuits_cache[(width, depth)]

                trial_settings.append(
                    TrialSetting(
                        setup=trial_setup,
                        gate_indices=gate_indices,
                        config=cfg,
                        subcircuits=subcircuits,
                        subcircuit_structures=subcircuit_structures,
                    )
                )

    return trial_settings, master_data


def experiment_run(
    cfg: ExperimentConfig, logger=None
) -> tuple[Iterator[TrialResult], TrialData]:
    """Prepare experiment and return iterator for trial results.

    Use this for iterative execution where you want to save/process
    each trial as it completes.

    Returns:
        Tuple of (trial_iterator, master_data) where:
        - trial_iterator yields TrialResult for each completed trial
        - master_data contains training/val/test data for all gates
    """
    trace(
        "experiment_run starting",
        widths=cfg.widths,
        depths=cfg.depths,
        activations=cfg.activations,
    )
    logger and logger.info(f"\n\n ExperimentConfig: \n {cfg} \n\n")

    trial_settings, master_data = build_trial_settings(cfg, logger)

    logger and logger.info(f"\nRunning {len(trial_settings)} trials iteratively\n")

    def trial_iterator():
        for idx, setting in enumerate(trial_settings):
            logger and logger.info(f"\n[Trial {idx + 1}/{len(trial_settings)}]")

            try:
                # Adapt master data for this trial's gates
                trial_data = master_data.select_gates(setting.gate_indices)

                trial_result = run_trial(
                    setting.setup,
                    trial_data,
                    device=cfg.device,
                    logger=logger,
                    debug=cfg.debug,
                    precomputed_circuits=(
                        setting.subcircuits,
                        setting.subcircuit_structures,
                    ),
                )
                yield trial_result
            except Exception as e:
                logger and logger.error(f"Trial {idx + 1} failed: {e}")

    return trial_iterator(), master_data
            # Continue with next trial rather than stopping


def run_experiment(
    cfg: ExperimentConfig,
    logger=None,
    max_parallel_trials: int = 1,
) -> tuple[ExperimentResult, TrialData]:
    """Run all trials in an experiment (parallel or serial).

    For long experiments, consider using experiment_run() generator
    with iterative saving instead.

    Returns:
        Tuple of (experiment_result, master_data) where master_data contains
        the training/val/test data for all gates.
    """
    trace(
        "run_experiment starting",
        widths=cfg.widths,
        depths=cfg.depths,
        activations=cfg.activations,
    )
    logger and logger.info(f"\n\n ExperimentConfig: \n {cfg} \n\n")

    experiment_result = ExperimentResult(config=cfg)

    # Get all trial settings and master data
    trial_settings, master_data = build_trial_settings(cfg, logger)

    logger and logger.info(f"\nRunning {len(trial_settings)} trials\n")

    # Prepare contexts for parallel execution - each includes adapted data
    trial_contexts = [(setting, master_data, cfg, logger) for setting in trial_settings]

    results = parallel_map(
        trial_contexts,
        run_trial_from_setting_parallel,
        max_workers=max_parallel_trials,
        desc="trials",
        single_item_fn=run_trial_from_setting_series,
    )

    for idx, trial_result, error in results:
        if error:
            print(f"[EXP] Trial {idx + 1} failed: {error}")
        else:
            experiment_result.trials[trial_result.trial_id] = trial_result

    return experiment_result, master_data


def run_trial_from_setting_parallel(ctx):
    """Run trial from TrialSetting context (for parallel execution)."""
    setting, master_data, cfg, logger = ctx
    trial_data = master_data.select_gates(setting.gate_indices)
    return run_trial(
        setting.setup,
        trial_data,
        device=cfg.device,
        logger=None,  # Disable logging in parallel to avoid interleaving
        debug=cfg.debug,
        precomputed_circuits=(setting.subcircuits, setting.subcircuit_structures),
    )


def run_trial_from_setting_series(ctx):
    """Run trial from TrialSetting context (for serial execution)."""
    setting, master_data, cfg, logger = ctx
    trial_data = master_data.select_gates(setting.gate_indices)
    return run_trial(
        setting.setup,
        trial_data,
        device=cfg.device,
        logger=logger,
        debug=cfg.debug,
        precomputed_circuits=(setting.subcircuits, setting.subcircuit_structures),
    )
