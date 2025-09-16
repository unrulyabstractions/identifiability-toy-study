import logging
import os
import pickle
import random
from itertools import chain, combinations

import numpy as np
import torch
from matplotlib import pyplot as plt


def setup_logging(log_path=None, debug=False):
    """
    Create a custom logger with a stream handler and a file handler

    Args:
        log_path: The path of the log file

    Returns:
        The logger
    """
    logger = logging.getLogger()
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(funcName)s - %(message)s")
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    if log_path:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def visualize_as_grid(objs, obj_type='', names=None, path=None, **kwargs):
    """
    Visualize a list of objects as a grid of subplots

    Args:
        objs: The list of objects to visualize (each class must implement the `visualize` method)
        obj_type: The name of the category of objects
        names: An optional dictionary mapping each object to an id or name
        path: The path to save the figure
        **kwargs: Additional keyword arguments to pass to the `visualize` method
    """
    n_items = len(objs)

    # Choose the number of rows and columns to make the grid as close as possible to a rectangle
    n_cols = int(np.ceil(np.sqrt(n_items)))
    n_rows = int(np.ceil(n_items / n_cols))

    # Create a figure and a grid of subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 3))

    # Flatten axes array for easier iteration (works even if n_rows or n_cols is 1)
    axes = axes.flatten()

    for idx, (obj, ax) in enumerate(zip(objs, axes)):
        obj.visualize(ax=ax, **kwargs)
        if obj_type and names:
            ax.set_title(f"{obj_type} {names[idx]}")

    # Hide any unused subplots
    for ax in axes[len(objs):]:
        ax.axis('off')

    plt.tight_layout()

    if path:
        plt.savefig(path)
        plt.close()
    else:
        plt.show()


def get_node_size(node_size):
    """
    Helper function to get the node size based on a string

    Args:
        node_size: "small", "medium", or "large"

    Returns:
        The corresponding node size
    """
    if node_size == 'small':
        return 500
    elif node_size == 'medium':
        return 1000
    elif node_size == 'large':
        return 1400
    else:
        raise ValueError(f"Unknown node size: {node_size}")


def set_seeds(seed):
    """
    Helper function to set seeds for reproducibility

    Args:
        seed: The seed to set
    """
    # Set the seed for Python's random module
    random.seed(seed)

    # Set the seed for numpy
    np.random.seed(seed)

    # Set the seed for PyTorch CPU and GPU (if available)
    torch.manual_seed(seed)

    # Ensure deterministic behavior for PyTorch
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model(log_dir):
    """
    Loads a MLP model from a directory

    Args:
        log_dir: The directory containing the model

    Returns:
        The loaded model
    """
    from mi_identifiability.neural_model import MLP

    model_names = [f for f in os.listdir(log_dir) if f.endswith('.pt')]
    assert len(model_names) == 1, 'Not a single model found'

    model_path = os.path.join(log_dir, model_names[0])
    return MLP.load_from_file(model_path)


def load_binary(filepath):
    """
    Loads a binary file

    Args:
        filepath: The path to the binary file

    Returns:
        The loaded binary data
    """
    with open(filepath, 'rb') as f:
        bin_data = pickle.loads(f.read())
    return bin_data


def save_binary(data, filepath):
    """
    Save a binary file to a directory

    Args:
        data: The data to save
        filepath: The path to the binary file
    """
    with open(filepath, 'wb') as f:
        f.write(pickle.dumps(data))


def powerset(iterable):
    """
    Enumerate all subsets of an iterable (https://stackoverflow.com/a/1482316)

    Args:
        iterable: The iterable

    Returns:
        A generator of all subsets
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))



from dataclasses import asdict, is_dataclass, dataclass, field
from typing import Any
from decimal import Decimal, ROUND_HALF_EVEN
import hashlib, json, math

def _qfloat(x: float, places: int = 8) -> float:
    # stable decimal rounding: converts via str -> Decimal -> quantize
    q = Decimal(1) / (Decimal(10) ** places)     # e.g. 1e-8
    d = Decimal(str(x)).quantize(q, rounding=ROUND_HALF_EVEN)
    # normalize -0.0 to 0.0 for stability
    f = float(d)
    return 0.0 if f == 0.0 else f

def _canon(obj: Any, places: int = 8):
    if isinstance(obj, float):
        if math.isnan(obj):  # avoid NaN destabilizing hashes
            return "NaN"
        return _qfloat(obj, places)
    if is_dataclass(obj):
        return _canon(asdict(obj), places)
    if isinstance(obj, dict):
        return {k: _canon(v, places) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_canon(v, places) for v in obj]
    return obj

def deterministic_id_from_dataclass(data_class_obj: Any, places: int = 8, digest_bytes: int = 16) -> str:
    canonical = _canon(data_class_obj, places)
    payload = json.dumps(
        canonical,
        sort_keys=True,                  # stable key order
        separators=(',', ':'),           # remove whitespace
        ensure_ascii=False,
        allow_nan=False
    )
    # fast, strong hash in the stdlib
    h = hashlib.blake2b(payload.encode("utf-8"), digest_size=digest_bytes)
    return h.hexdigest()
