import random
from dataclasses import asdict
from itertools import product

from identifiability_toy_study.mi_identifiability.utils import (
    set_seeds,
)

from .study_core import (
    DataParams,
    IdentifiabilityConstraints,
    ModelParams,
    TrainParams,
    TrialSetup,
)
from .study_utils import (
    generate_trial_data,
)
from .trial import (
    run_trial,
)


def run_experiment(
    args,
    logger=None,
    model_dir: str = "",
):
    logger and logger.info("Configuration in use:")
    logger and logger.info(args)

    experiment_result = {}

    for n_gates, seed_offset in product(args.n_gates, range(args.n_experiments)):
        logic_gates = random.sample(args.target_logic_gates, k=n_gates)
        seed = args.seed + seed_offset
        set_seeds(seed)

        data_params = DataParams(
            n_samples_train=args.n_samples_train,
            n_samples_val=args.n_samples_val,
            n_samples_test=args.n_samples_test,
            noise_std=args.noise_std,
            skewed_distribution=args.skewed_distribution,
        )
        data = generate_trial_data(data_params, logic_gates, args.device)
        if args.debug:
            logger and logger.info(
                f"checking_device: x_train: {data.train.x.device}, y_train: {data.train.x.device}"
            )
            logger and logger.info(
                f"checking_device: x_val: {data.val.x.device}, y_val: {data.val.x.device}"
            )
            logger and logger.info(
                f"checking_device: x_test: {data.test.x.device}, y_test: {data.test.x.device}"
            )

        WIDTHS = [ModelParams([]).width]
        DEPTHS = [ModelParams([]).depth]
        for width, depth, loss_target, lr in product(
            WIDTHS, DEPTHS, args.loss_target, args.learning_rate
        ):
            setup = TrialSetup(
                seed=seed,
                model_params=ModelParams(
                    width=width, depth=depth, logic_gates=logic_gates
                ),
                train_params=TrainParams(
                    learning_rate=lr,
                    loss_target=loss_target,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    acc_target=args.accuracy_threshold,
                    val_frequency=args.val_frequency,
                ),
                data_params=data_params,
                iden_constraints=IdentifiabilityConstraints(
                    min_sparsity=args.min_sparsity,
                    acc_threshold=args.accuracy_threshold,
                    is_perfect_circuit=True,
                    is_causal_abstraction=False,
                    non_transport_stable=False,
                    param_decomp=False,
                ),
            )
            result = run_trial(
                setup,
                data,
                device=args.device,
                logger=logger,
                debug=args.debug,
                model_dir=model_dir,
                try_load_model=(not args.from_scratch),
                try_save_model=True,
            )
            experiment_result[result.trial_id] = asdict(result)

    return experiment_result
