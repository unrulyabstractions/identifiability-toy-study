"""
An experiment has many runs. Each run is called a trial.
The experiment as a whole will have an ExperimentConfig.
Each individual trial will have a TrialSetup.
Look at .common.schemas for full definitions.
"""

import copy
import random
from itertools import combinations, product

from src.circuit.precompute import precompute_circuits_for_architectures
from src.domain import get_max_n_inputs, normalize_gate_names
from src.experiment_config import ExperimentConfig, TrialSetup
from src.infra import parallel_map
from src.infra.profiler import trace, traced
from src.schemas import ExperimentResult
from src.training import generate_trial_data

from .trial import run_trial_in_parallel, run_trial_in_series


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


def run_experiment(
    cfg: ExperimentConfig,
    logger=None,
    max_parallel_trials: int = 1,
) -> ExperimentResult:
    trace("run_experiment starting", widths=cfg.widths, depths=cfg.depths, activations=cfg.activations)
    logger and logger.info(f"\n\n ExperimentConfig: \n {cfg} \n\n")

    experiment_result = ExperimentResult(config=cfg)

    # Normalize gate names to handle duplicates (e.g., ["XOR", "XOR"] -> ["XOR", "XOR_2"])
    normalized_gates = normalize_gate_names(cfg.target_logic_gates)
    logger and logger.info(f"Normalized gates: {normalized_gates}")

    gates_combinations = build_gate_combinations(
        normalized_gates, cfg.num_gates_per_run
    )

    # Determine input size from gates - use max across all gates to support mixed sizes
    first_combo = gates_combinations[0]
    input_size = get_max_n_inputs(first_combo)

    # Determine output size (max gates across all combinations)
    output_size = max(len(combo) for combo in gates_combinations)

    # Pre-compute circuits for all architectures
    logger and logger.info("\nPre-computing circuits for all architectures...")
    circuits_cache = precompute_circuits_for_architectures(
        cfg.widths, cfg.depths, input_size, output_size, logger
    )

    # Collect all trial configurations
    trial_settings = []

    logger and logger.info(
        f"\nStarting {cfg.num_runs} runs, each with {len(gates_combinations)}\n"
    )
    for seed_offset in range(cfg.num_runs):
        for logic_gates in gates_combinations:
            data_params = copy.deepcopy(cfg.base_trial.data_params)
            trial_data = generate_trial_data(
                data_params, logic_gates, cfg.device, logger=logger, debug=cfg.debug
            )

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
                model_params.logic_gates = logic_gates

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
                    (
                        trial_setup,
                        trial_data,
                        cfg,
                        logger,
                        subcircuits,
                        subcircuit_structures,
                    )
                )

    results = parallel_map(
        trial_settings,
        run_trial_in_parallel,
        max_workers=max_parallel_trials,
        desc="trials",
        single_item_fn=run_trial_in_series,
    )

    for idx, trial_result, error in results:
        if error:
            print(f"[EXP] Trial {idx + 1} failed: {error}")
        else:
            experiment_result.trials[trial_result.trial_id] = trial_result

    return experiment_result
