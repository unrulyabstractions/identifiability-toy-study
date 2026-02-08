import copy
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from typing import Optional

from src.experiment_config import ExperimentConfig, TrialSetup
from src.infra import ParallelConfig, set_seeds
from src.schemas import ExperimentResult
from src.training import generate_trial_data

from .trial import run_trial

"""
An experiment has many runs. Each run is called a trial
The experiment as a whole will have a ExperimentConfig.
Each individual trial will have a TrialSetup
Look at .common.schemas for full definitions
"""


def run_experiment(
    cfg: ExperimentConfig,
    logger=None,
    max_parallel_trials: int = 4,
    parallel_config: Optional[ParallelConfig] = None,
) -> ExperimentResult:
    logger and logger.info(f"\n\n ExperimentConfig: \n {cfg} \n\n")

    experiment_result = ExperimentResult(config=cfg)

    # If num_gates_per_run is None, use all gates (no sampling)
    num_gates_list = (
        cfg.num_gates_per_run
        if cfg.num_gates_per_run
        else [len(cfg.target_logic_gates)]
    )

    # Collect all trial configurations
    trial_configs = []

    for n_gates, seed_offset in product(num_gates_list, range(cfg.num_runs)):
        # If n_gates equals total gates, use all; otherwise sample
        if n_gates == len(cfg.target_logic_gates):
            logic_gates = list(cfg.target_logic_gates)
        else:
            logic_gates = random.sample(cfg.target_logic_gates, k=n_gates)
        seed = cfg.base_trial.seed + seed_offset
        set_seeds(seed)

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
                seed=seed,
                data_params=data_params,
                model_params=model_params,
                train_params=train_params,
                constraints=constraints,
            )

            trial_configs.append((trial_setup, trial_data))

    # Run trials in parallel
    if len(trial_configs) == 1:
        # Single trial - run directly without threading overhead
        trial_setup, trial_data = trial_configs[0]
        trial_result = run_trial(
            trial_setup,
            trial_data,
            device=cfg.device,
            logger=logger,
            debug=cfg.debug,
            parallel_config=parallel_config,
        )
        experiment_result.trials[trial_result.trial_id] = trial_result
    else:
        # Multiple trials - run in parallel
        n_workers = min(len(trial_configs), max_parallel_trials)
        print(
            f"[EXP] Running {len(trial_configs)} trials with {n_workers} parallel workers"
        )

        def run_single_trial(config):
            trial_setup, trial_data = config
            return run_trial(
                trial_setup,
                trial_data,
                device=cfg.device,
                logger=None,  # Disable logging in parallel to avoid interleaving
                debug=cfg.debug,
                parallel_config=parallel_config,
            )

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(run_single_trial, config): i
                for i, config in enumerate(trial_configs)
            }

            for future in as_completed(futures):
                trial_idx = futures[future]
                try:
                    trial_result = future.result()
                    experiment_result.trials[trial_result.trial_id] = trial_result
                    print(
                        f"[EXP] Trial {trial_idx + 1}/{len(trial_configs)} completed: {trial_result.trial_id[:8]}..."
                    )
                except Exception as e:
                    print(
                        f"[EXP] Trial {trial_idx + 1}/{len(trial_configs)} failed: {e}"
                    )

    return experiment_result
