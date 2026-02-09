import copy
import random
from itertools import combinations, product

from src.experiment_config import ExperimentConfig, TrialSetup
from src.infra import parallel_map
from src.schemas import ExperimentResult
from src.training import generate_trial_data

from .trial import run_trial_in_parallel, run_trial_in_series

"""
An experiment has many runs. Each run is called a trial
The experiment as a whole will have a ExperimentConfig.
Each individual trial will have a TrialSetup
Look at .common.schemas for full definitions
"""


def run_experiment(
    cfg: ExperimentConfig,
    logger=None,
    max_parallel_trials: int = 1,
) -> ExperimentResult:
    logger and logger.info(f"\n\n ExperimentConfig: \n {cfg} \n\n")

    experiment_result = ExperimentResult(config=cfg)

    # Build gate combinations for trials
    # Each element should be a tuple/list of gate names for one trial
    if cfg.num_gates_per_run and cfg.num_gates_per_run <= len(cfg.target_logic_gates):
        gates_combinations = list(
            combinations(cfg.target_logic_gates, cfg.num_gates_per_run)
        )
        random.shuffle(gates_combinations)
    else:
        # Use all gates as a single combination
        gates_combinations = [tuple(cfg.target_logic_gates)]

    # Collect all trial configurations
    trial_configs = []

    logger and logger.info(
        f"\nStarting {cfg.num_runs} runs, each with {len(gates_combinations)}\n"
    )
    for seed_offset in range(cfg.num_runs):
        for logic_gates in gates_combinations:
            data_params = copy.deepcopy(cfg.base_trial.data_params)
            trial_data = generate_trial_data(
                data_params, logic_gates, cfg.device, logger=logger, debug=cfg.debug
            )

            for width, depth, lr in product(cfg.widths, cfg.depths, cfg.learning_rates):
                # Each trial should have its independent set of params
                model_params = copy.deepcopy(cfg.base_trial.model_params)
                train_params = copy.deepcopy(cfg.base_trial.train_params)
                constraints = copy.deepcopy(cfg.base_trial.constraints)

                train_params.learning_rate = lr
                model_params.width = width
                model_params.depth = depth
                model_params.logic_gates = logic_gates

                trial_setup = TrialSetup(
                    seed=cfg.base_trial.seed + seed_offset,
                    data_params=data_params,
                    model_params=model_params,
                    train_params=train_params,
                    constraints=constraints,
                )

                trial_configs.append((trial_setup, trial_data, cfg, logger))

    results = parallel_map(
        trial_configs,
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
