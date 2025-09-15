import glob
import os
import random
from datetime import datetime
from typing import Optional

import torch
import torch.nn as nn

from identifiability_toy_study.mi_identifiability.logic_gates import (
    ALL_LOGIC_GATES,
    generate_noisy_multi_gate_data,
)
from identifiability_toy_study.mi_identifiability.neural_model import MLP

from .study_core import (
    DataParams,
    Dataset,
    ModelParams,
    TrainParams,
    TrialData,
    TrialResult,
)


def update_status_fx(trial_result: TrialResult, logger=None):
    def update_status(status: str, mssg: Optional[str] = None):
        trial_result.status = status
        if logger:
            if mssg:
                logger.info(
                    f" trial_id:{trial_result.trial_id} trial_status:{trial_result.status} mssg:{mssg}"
                )

            else:
                logger.info(
                    f" trial_id:{trial_result.trial_id} trial_status:{trial_result.status}"
                )

    return update_status


def generate_dataset(
    gate_names: list[str],
    device: str,
    n_repeats: int = 1,
    noise_std: float = 0.0,
    weights=None,
    logger=None,
) -> Dataset:
    gates = [ALL_LOGIC_GATES[gate_name] for gate_name in gate_names]
    n_gates = len(gates)
    if n_gates > 1:
        x, y = generate_noisy_multi_gate_data(
            gates,
            n_repeats=n_repeats,
            noise_std=noise_std,
            device=device,
        )
    else:
        x, y = gates[0].generate_noisy_data(
            n_repeats=n_repeats,
            weights=weights,
            noise_std=noise_std,
            device=device,
        )
    return Dataset(x=x, y=y)


def generate_trial_data(
    data_params: DataParams,
    logic_gates: list[str],
    device: str,
    logger=None,
) -> TrialData:
    # TODO: Keep this?
    weights = None
    if data_params.skewed_distribution:
        n_inputs = 2  # We assume all have same n_inputs
        logger.info("using skewed_distribution")
        weights = [random.random() for _ in range(2**n_inputs)]

    train_data = generate_dataset(
        logic_gates,
        n_repeats=data_params.n_samples_train,
        weights=weights,
        noise_std=data_params.noise_std,
        device=device,
    )
    val_data = generate_dataset(
        logic_gates,
        n_repeats=data_params.n_samples_val,
        weights=weights,
        noise_std=data_params.noise_std,
        device=device,
    )
    test_data = generate_dataset(
        logic_gates,
        n_repeats=data_params.n_samples_test,
        weights=weights,
        noise_std=data_params.noise_std,
        device=device,
    )

    return TrialData(
        train=train_data,
        val=val_data,
        test=test_data,
    )


def train_model(
    train_params: TrainParams,
    model_params: ModelParams,
    data: TrialData,
    device: str,
    input_size: int = 2,
    output_size: int = 1,
    status_fx=None,
    logger=None,
    debug: bool = False,
):
    status_fx and status_fx("STARTED_TRAINING")

    model = MLP(
        hidden_sizes=([model_params.width] * model_params.depth),
        input_size=input_size,
        output_size=output_size,
        device=device,
        debug=debug,
        logger=logger,
    )

    avg_loss = model.do_train(
        x=data.train.x,
        y=data.train.y,
        x_val=data.val.x,
        y_val=data.val.y,
        batch_size=train_params.batch_size,
        learning_rate=train_params.learning_rate,
        epochs=train_params.epochs,
        loss_target=train_params.loss_target,
        val_frequency=train_params.val_frequency,
        logger=logger,
    )

    val_acc = model.do_eval(data.val.x, data.val.y)

    mssg = f"avg_loss={avg_loss} val_acc={val_acc}"
    if val_acc < train_params.acc_target or avg_loss > train_params.loss_target:
        status_fx and status_fx("FAILED_TRAINING", mssg=mssg)
        return None, avg_loss, val_acc

    status_fx and status_fx("FINISHED_TRAINING", mssg=mssg)
    return model, avg_loss, val_acc


def load_model(
    model_dir: str,
    model_params: ModelParams,
    train_params: TrainParams,
    device: str,
    logger=None,
):
    model_id = model_params.get_id()

    # Find all model files matching the model_id pattern
    pattern = os.path.join(model_dir, f"model_{model_id}_*.pt")
    model_files = glob.glob(pattern)

    if not model_files:
        logger and logger.info(f"No model files found for model_id: {model_id}")
        return None, None, None

    # Parse filenames to extract avg_loss and val_acc
    best_loss = float("inf")
    best_val_acc = -1.0
    best_file = None

    for file_path in model_files:
        try:
            # Extract filename without path and extension
            filename = os.path.basename(file_path).replace(".pt", "")
            # Expected format: model_{model_id}_loss_{avg_loss}_acc_{val_acc}
            parts = filename.split("_")
            loss_idx = parts.index("loss") + 1
            acc_idx = parts.index("acc") + 1

            avg_loss = float(parts[loss_idx])
            val_acc = float(parts[acc_idx])

            # Select best model: lowest avg_loss, tie-break by highest val_acc
            if (avg_loss < best_loss) or (
                avg_loss == best_loss and val_acc > best_val_acc
            ):
                best_loss = avg_loss
                best_val_acc = val_acc
                best_file = file_path

        except (ValueError, IndexError) as e:
            logger and logger.warning(
                f"Could not parse model filename {file_path}: {e}"
            )
            continue

    if best_file is None:
        logger and logger.error(f"No valid model files found for model_id: {model_id}")
        return None, None, None

    try:
        # Load the model state dict to the specified device
        checkpoint = torch.load(best_file, map_location=device)

        # Create a new model instance with the same parameters
        gates = [ALL_LOGIC_GATES[gate] for gate in model_params.logic_gates]
        input_size = gates[0].n_inputs
        output_size = len(gates)

        model = MLP(
            hidden_sizes=([model_params.width] * model_params.depth),
            input_size=input_size,
            output_size=output_size,
            device=device,
        )

        # Load the state dict
        model.load_state_dict(checkpoint["model_state_dict"])

        logger and logger.info(
            f"Successfully loaded model from {best_file} to device {device} (loss: {best_loss}, acc: {best_val_acc})"
        )

        if (
            best_val_acc < train_params.acc_target
            or best_loss > train_params.loss_target
        ):
            logger and logger.info(
                f"Loaded  model from {best_file} is not good enough for acc_target or loss_target"
            )
            return None, avg_loss, val_acc

        return model, best_loss, best_val_acc

    except Exception as e:
        logger and logger.error(f"Failed to load model from {best_file}: {e}")
        return None, None, None


def save_model(
    model_dir: str,
    model_params: ModelParams,
    model: nn.Module,
    avg_loss: float,
    val_acc: float,
    device: str,
    logger=None,
):
    model_id = model_params.get_id()

    # Create filename with model_id, avg_loss, and val_acc
    # Format: model_{model_id}_loss_{avg_loss:.6f}_acc_{val_acc:.6f}_{timestamp}.pt
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"model_{model_id}_loss_{avg_loss:.6f}_acc_{val_acc:.6f}_{timestamp}.pt"
    filepath = os.path.join(model_dir, filename)

    try:
        # Ensure model directory exists
        os.makedirs(model_dir, exist_ok=True)

        # Create checkpoint dictionary
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_params": {
                "logic_gates": model_params.logic_gates,
                "width": model_params.width,
                "depth": model_params.depth,
            },
            "avg_loss": avg_loss,
            "val_acc": val_acc,
            "model_id": model_id,
            "timestamp": timestamp,
        }

        # Save the model
        torch.save(checkpoint, filepath)

        logger and logger.info(f"Successfully saved model to {filepath}")
        return filepath

    except Exception as e:
        logger and logger.error(f"Failed to save model to {filepath}: {e}")
        return None
