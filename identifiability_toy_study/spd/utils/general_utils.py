import copy
import importlib
import json
import random
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Type, TypeVar

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import yaml
from jaxtyping import Float
from pydantic import BaseModel, PositiveFloat
from pydantic.v1.utils import deep_update
from torch import Tensor

from spd.log import logger
from spd.utils.run_utils import save_file

T = TypeVar('T', bound=BaseModel)
BaseModelType = TypeVar('BaseModelType', bound=BaseModel)
T_Runtime = TypeVar('T_Runtime')

# Avoid seaborn package installation (sns.color_palette("colorblind").as_hex())
COLOR_PALETTE = [
    "#0173B2",
    "#DE8F05",
    "#029E73",
    "#D55E00",
    "#CC78BC",
    "#CA9161",
    "#FBAFE4",
    "#949494",
    "#ECE133",
    "#56B4E9",
]


def set_seed(seed: int | None) -> None:
    """Set the random seed for random, PyTorch and NumPy"""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)


def generate_sweep_id() -> str:
    """Generate a unique sweep ID based on timestamp."""
    return f"sweep_id-{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def load_config(
    config_path_or_obj: Path | str | dict[str, Any] | T, config_model: Type[T]
) -> T:
    """Load the config of class `config_model`, from various sources.

    Args:
        config_path_or_obj (Union[Path, str, dict, `config_model`]): Can be:
            - config object: must be instance of `config_model`
            - dict: config dictionary
            - str starting with 'json:': JSON string with prefix
            - other str: treated as path to a .yaml file
            - Path: path to a .yaml file
        config_model: the class of the config that we are loading
    """
    if isinstance(config_path_or_obj, config_model):
        return config_path_or_obj

    if isinstance(config_path_or_obj, dict):
        return config_model(**config_path_or_obj)

    if isinstance(config_path_or_obj, str):
        # Check if it's a prefixed JSON string
        if config_path_or_obj.startswith("json:"):
            config_dict = json.loads(config_path_or_obj[5:])
            return config_model(**config_dict)
        else:
            # Treat as file path
            config_path_or_obj = Path(config_path_or_obj)

    assert isinstance(config_path_or_obj, Path), (
        f"passed config is of invalid type {type(config_path_or_obj)}"
    )
    assert config_path_or_obj.suffix == ".yaml", (
        f"Config file {config_path_or_obj} must be a YAML file."
    )
    assert Path(config_path_or_obj).exists(), f"Config file {config_path_or_obj} does not exist."
    with open(config_path_or_obj) as f:
        config_dict = yaml.safe_load(f)
    return config_model(**config_dict)


def replace_pydantic_model(
    model: BaseModelType, *updates: dict[str, Any]
) -> BaseModelType:
    """Create a new model with (potentially nested) updates in the form of dictionaries.

    Args:
        model: The model to update.
        updates: The zero or more dictionaries of updates that will be applied sequentially.

    Returns:
        A replica of the model with the updates applied.

    Examples:
        >>> class Foo(BaseModel):
        ...     a: int
        ...     b: int
        >>> foo = Foo(a=1, b=2)
        >>> foo2 = replace_pydantic_model(foo, {"a": 3})
        >>> foo2
        Foo(a=3, b=2)
        >>> class Bar(BaseModel):
        ...     foo: Foo
        >>> bar = Bar(foo={"a": 1, "b": 2})
        >>> bar2 = replace_pydantic_model(bar, {"foo": {"a": 3}})
        >>> bar2
        Bar(foo=Foo(a=3, b=2))
    """
    return model.__class__(**deep_update(model.model_dump(), *updates))


def compute_feature_importances(
    batch_size: int,
    n_features: int,
    importance_val: float | None,
    device: str,
) -> Float[Tensor, "batch_size n_features"]:
    # Defines a tensor where the i^th feature has importance importance^i
    if importance_val is None or importance_val == 1.0:
        importance_tensor = torch.ones(batch_size, n_features, device=device)
    else:
        powers = torch.arange(n_features, device=device)
        importances = torch.pow(importance_val, powers)
        importance_tensor = einops.repeat(
            importances, "n_features -> batch_size n_features", batch_size=batch_size
        )
    return importance_tensor


def get_lr_schedule_fn(
    lr_schedule: Literal["linear", "constant", "cosine", "exponential"],
    lr_exponential_halflife: PositiveFloat | None = None,
) -> Callable[[int, int], float]:
    """Get a function that returns the learning rate at a given step.

    Args:
        lr_schedule: The learning rate schedule to use
        lr_exponential_halflife: The halflife of the exponential learning rate schedule
    """
    if lr_schedule == "linear":
        return lambda step, steps: 1 - (step / steps)
    elif lr_schedule == "constant":
        return lambda *_: 1.0
    elif lr_schedule == "cosine":
        return lambda step, steps: 1.0 if steps == 1 else np.cos(0.5 * np.pi * step / (steps - 1))
    else:
        # Exponential
        assert lr_exponential_halflife is not None  # Should have been caught by model validator
        halflife = lr_exponential_halflife
        gamma = 0.5 ** (1 / halflife)
        logger.info(f"Using exponential LR schedule with halflife {halflife} steps (gamma {gamma})")
        return lambda step, steps: gamma**step


def get_lr_with_warmup(
    step: int,
    steps: int,
    lr: float,
    lr_schedule_fn: Callable[[int, int], float],
    lr_warmup_pct: float,
) -> float:
    warmup_steps = int(steps * lr_warmup_pct)
    if step < warmup_steps:
        return lr * (step / warmup_steps)
    return lr * lr_schedule_fn(step - warmup_steps, steps - warmup_steps)


def replace_deprecated_param_names(
    params: dict[str, Float[Tensor, "..."]], name_map: dict[str, str]
) -> dict[str, Float[Tensor, "..."]]:
    """Replace old parameter names with new parameter names in a dictionary.

    Args:
        params: The dictionary of parameters to fix
        name_map: A dictionary mapping old parameter names to new parameter names
    """
    for k in list(params.keys()):
        for old_name, new_name in name_map.items():
            if old_name in k:
                params[k.replace(old_name, new_name)] = params[k]
                del params[k]
    return params


def resolve_class(path: str) -> Type[nn.Module]:
    """Load a class from a string indicating its import path.

    Args:
        path: The path to the class, e.g. "transformers.LlamaForCausalLM" or
            "spd.experiments.resid_mlp.models.ResidualMLP"
    """
    module_path, _, class_name = path.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def extract_batch_data(
    batch_item: dict[str, Any] | tuple[Tensor, ...] | Tensor,
    input_key: str = "input_ids",
) -> Tensor:
    """Extract input data from various batch formats.

    This utility function handles different batch formats commonly used across the codebase:
    1. Dictionary format: {"input_ids": tensor, ...} - common in LM tasks
    2. Tuple format: (input_tensor, labels) - common in SPD optimization
    3. Direct tensor: when batch is already the input tensor

    Args:
        batch_item: The batch item from a data loader
        input_key: Key to use for dictionary format (default: "input_ids")

    Returns:
        The input tensor extracted from the batch
    """
    assert isinstance(batch_item, dict | tuple | Tensor), (
        f"Unsupported batch format: {type(batch_item)}. Must be a dictionary, tuple, or tensor."
    )
    if isinstance(batch_item, dict):
        # Dictionary format: extract the specified key
        if input_key not in batch_item:
            available_keys = list(batch_item.keys())
            raise KeyError(
                f"Key '{input_key}' not found in batch. Available keys: {available_keys}"
            )
        tensor = batch_item[input_key]
    elif isinstance(batch_item, tuple):
        # Assume input is the first element
        tensor = batch_item[0]
    else:
        # Direct tensor format
        tensor = batch_item

    return tensor


def calc_kl_divergence_lm(
    pred: Float[Tensor, "... vocab"],
    target: Float[Tensor, "... vocab"],
) -> Float[Tensor, ""]:
    """Calculate the KL divergence between two logits."""
    assert pred.shape == target.shape
    log_q = torch.log_softmax(pred, dim=-1)  # log Q
    p = torch.softmax(target, dim=-1)  # P
    kl = F.kl_div(log_q, p, reduction="none")  # P · (log P − log Q)
    return kl.sum(dim=-1).mean()  # Σ_vocab / (batch·seq)


def apply_nested_updates(base_dict: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Apply nested updates to a dictionary."""
    result = copy.deepcopy(base_dict)

    for key, value in updates.items():
        if "." in key:
            # Handle nested keys
            keys = key.split(".")
            current = result

            # Navigate to the parent of the final key
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]

            # Set the final value
            current[keys[-1]] = value
        else:
            # Simple key
            result[key] = value

    return result


def runtime_cast(type_: Type[T_Runtime], obj: Any) -> T_Runtime:
    """typecast with a runtime check"""
    if not isinstance(obj, type_):
        raise TypeError(f"Expected {type_}, got {type(obj)}")
    return obj


def _fetch_latest_checkpoint_name(filenames: list[str], prefix: str | None = None) -> str:
    """Fetch the latest checkpoint name from a list of .pth files.

    Assumes format is <name>_<step>.pth or <name>.pth.
    """
    if prefix:
        filenames = [filename for filename in filenames if filename.startswith(prefix)]
    if not filenames:
        raise ValueError(f"No files found with prefix {prefix}")
    if len(filenames) == 1:
        latest_checkpoint_name = filenames[0]
    else:
        latest_checkpoint_name = sorted(
            filenames, key=lambda x: int(x.split(".pth")[0].split("_")[-1])
        )[-1]
    return latest_checkpoint_name


def fetch_latest_local_checkpoint(run_dir: Path, prefix: str | None = None) -> Path:
    """Fetch the latest checkpoint from a local run directory."""
    filenames = [file.name for file in run_dir.iterdir() if file.name.endswith(".pth")]
    latest_checkpoint_name = _fetch_latest_checkpoint_name(filenames, prefix)
    latest_checkpoint_local = run_dir / latest_checkpoint_name
    return latest_checkpoint_local


def save_pre_run_info(
    save_to_wandb: bool,
    out_dir: Path,
    spd_config: BaseModel,
    sweep_params: dict[str, Any] | None,
    target_model: nn.Module | None,
    train_config: BaseModel | None,
    task_name: str | None,
) -> None:
    """Save run information locally and optionally to wandb."""

    files_to_save = {
        "final_config.yaml": spd_config.model_dump(mode="json"),
    }

    if target_model is not None:
        files_to_save[f"{task_name}.pth"] = target_model.state_dict()

    if train_config is not None:
        files_to_save[f"{task_name}_train_config.yaml"] = train_config.model_dump(mode="json")

    if sweep_params is not None:
        files_to_save["sweep_params.yaml"] = sweep_params

    for filename, data in files_to_save.items():
        filepath = out_dir / filename
        save_file(data, filepath)

        if save_to_wandb:
            wandb.save(str(filepath), base_path=out_dir, policy="now")


def get_linear_annealed_p(
    step: int,
    steps: int,
    initial_p: float,
    p_anneal_start_frac: float,
    p_anneal_final_p: float | None,
    p_anneal_end_frac: float = 1.0,
) -> float:
    """Calculate the linearly annealed p value for L_p sparsity loss.

    Args:
        step: Current training step
        steps: Total number of training steps
        initial_p: Starting p value
        p_anneal_start_frac: Fraction of training after which to start annealing
        p_anneal_final_p: Final p value to anneal to
        p_anneal_end_frac: Fraction of training when annealing ends. We stay at the final p value from this point onward

    Returns:
        Current p value based on linear annealing schedule
    """
    if p_anneal_final_p is None or p_anneal_start_frac >= 1.0:
        return initial_p

    assert p_anneal_end_frac >= p_anneal_start_frac, (
        f"p_anneal_end_frac ({p_anneal_end_frac}) must be >= "
        f"p_anneal_start_frac ({p_anneal_start_frac})"
    )

    cur_frac = step / steps

    if cur_frac < p_anneal_start_frac:
        return initial_p
    elif cur_frac >= p_anneal_end_frac:
        return p_anneal_final_p
    else:
        # Linear interpolation between start and end fractions
        progress = (cur_frac - p_anneal_start_frac) / (p_anneal_end_frac - p_anneal_start_frac)
        return initial_p + (p_anneal_final_p - initial_p) * progress
