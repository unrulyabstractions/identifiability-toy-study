from identifiability_toy_study.iden_utils import filter_by_constraints
from identifiability_toy_study.mi_identifiability.circuit import (
    enumerate_all_valid_circuit,
)
from identifiability_toy_study.mi_identifiability.logic_gates import (
    ALL_LOGIC_GATES,
)
from identifiability_toy_study.mi_identifiability.utils import (
    set_seeds,
)
from identifiability_toy_study.study_core import (
    TrialData,
    TrialResult,
    TrialSetup,
)
from identifiability_toy_study.study_utils import (
    load_model,
    save_model,
    train_model,
    update_status_fx,
)


def get_model(
    setup: TrialSetup,
    device: str,
    data: TrialData = None,
    try_load_model: bool = True,
    try_save_model: bool = True,
    model_dir: str = "",
    logger=None,
    debug: bool = False,
    status_fx=None,
):
    model = None
    avg_loss = None
    val_acc = None
    if try_load_model:
        model, avg_loss, val_acc = load_model(
            model_dir=model_dir,
            model_params=setup.model_params,
            train_params=setup.train_params,
            device=device,
            logger=logger,
        )

    if model is None and data is not None:
        model, avg_loss, val_acc = train_model(
            device=device,
            train_params=setup.train_params,
            model_params=setup.model_params,
            data=data,
            status_fx=status_fx,
            logger=logger,
            debug=debug,
            input_size=ALL_LOGIC_GATES[
                setup.model_params.logic_gates[0]
            ].n_inputs,  # We assume all tasks have same input_size
            output_size=len(setup.model_params.logic_gates),
        )
        if try_save_model:
            save_model(
                model_dir=model_dir,
                model_params=setup.model_params,
                model=model,
                avg_loss=avg_loss,
                val_acc=val_acc,
                device=device,
                logger=logger,
            )

    return model, avg_loss, val_acc


def run_trial(
    setup: TrialSetup,
    data: TrialData,
    device: str,
    logger=None,
    debug: bool = False,
    try_load_model: bool = True,
    try_save_model: bool = True,
    model_dir: str = "",
):
    # This also sets deterministic behavior
    set_seeds(setup.seed)

    trial_result = TrialResult(
        setup=setup,
    )
    trial_metrics = trial_result.metrics
    update_status = update_status_fx(trial_result, logger)

    # ===== Get Model: Load or Train =====
    model, avg_loss, val_acc = get_model(
        setup=setup,
        device=device,
        data=data,
        try_load_model=try_load_model,
        try_save_model=try_save_model,
        model_dir=model_dir,
        logger=logger,
        debug=debug,
    )
    if model is None:
        return trial_result

    test_acc = model.do_eval(data.test.x, data.test.y)

    trial_metrics.avg_loss = avg_loss
    trial_metrics.val_acc = val_acc
    trial_metrics.test_acc = test_acc

    # ===== Analysis =====
    update_status("STARTED_ANALYSIS_PER_CIRCUIT")

    submodels = model.separate_into_k_mlps()
    for i, submodel in enumerate(submodels):
        gate_name = setup.model_params.logic_gates[i]

        # ===== Find Circuits =====
        update_status("STARTED_CIRCUIT_FINDING")
        all_circuits_for_task = enumerate_all_valid_circuit(model, use_tqdm=True)
        update_status("FINISHED_CIRCUIT_FINDING")

        # ===== Filter by Constraints =====
        update_status("STARTED_FILTERING_BY_CONSTRAINTS")
        gate_metrics = filter_by_constraints(
            constraints=setup.iden_constraints,
            dataset=data.test,
            model=submodel,
            circuits=all_circuits_for_task,
            device=device,
            logger=None,
        )
        update_status("FINISHED_FILTERING_BY_CONSTRAINTS")

        trial_metrics.per_gate[gate_name] = gate_metrics

        logger and logger.info(
            f"Submodel #{i} {gate_name} has {len(gate_metrics.faithful_circuits_idx)} good circuits from {gate_metrics.num_total_circuits}"
        )

    update_status("FINISHED_ANALYSIS_PER_CIRCUIT")

    return trial_result
