import copy
import random
from itertools import product

from .common.helpers import (
    generate_trial_data,
)
from .common.schemas import (
    ExperimentConfig,
    ExperimentResult,
    TrialSetup,
)
from .common.utils import (
    set_seeds,
)
from .trial import (
    run_trial,
)


def run_experiment(cfg: ExperimentConfig, logger=None) -> ExperimentResult:
    logger and logger.info(f"\n\n ExperimentConfig: \n {cfg} \n\n")

    experiment_result = ExperimentResult(config=cfg)

    for n_gates, seed_offset in product(cfg.num_gates_per_run, range(cfg.num_runs)):
        logic_gates = random.sample(cfg.target_logic_gates, k=n_gates)
        seed = cfg.base_trial.seed + seed_offset
        set_seeds(seed)

        data_params = copy.deepcopy(cfg.base_trial.data_params)
        trial_data = generate_trial_data(
            data_params, logic_gates, cfg.device, logger=logger, debug=cfg.debug
        )

        for width, depth, loss_target, lr in product(
            cfg.widths, cfg.depths, cfg.loss_targets, cfg.learning_rates
        ):
            # Each trial should have its independent set of params
            model_params = copy.deepcopy(cfg.base_trial.model_params)
            train_params = copy.deepcopy(cfg.base_trial.train_params)
            iden_constraints = copy.deepcopy(cfg.base_trial.iden_constraints)

            train_params.learning_rate = lr
            train_params.loss_target = loss_target
            model_params.width = width
            model_params.depth = depth
            model_params.logic_gates = logic_gates

            trial_setup = TrialSetup(
                seed=seed,
                data_params=data_params,
                model_params=model_params,
                train_params=train_params,
                iden_constraints=iden_constraints,
            )
            trial_result = run_trial(
                trial_setup,
                trial_data,
                device=cfg.device,
                logger=logger,
                debug=cfg.debug,
                model_dir=cfg.model_dir,
                try_load_model=(not cfg.from_scratch),
                try_save_model=True,
            )
            experiment_result.trials[trial_result.trial_id] = trial_result

    return experiment_result
