import logging
import random
from dataclasses import asdict, is_dataclass
from itertools import chain, combinations

import numpy as np
import torch
from matplotlib import pyplot as plt


def nn_sample_complexity(width, depth, n_gates=7, noise_std=0.0):
    """
    Estimate samples for training NN on boolean circuits.

    samples ≈ k * w * d² * n_gates * noise_factor
    """

    # Noise increases sample complexity: ~1/(1-noise)² scaling
    # From standard PAC learning bounds with label noise
    noise_factor = 1 / max(1 - noise_std, 0.1) ** 2

    # Base multiplier: NN sample inefficiency vs information-theoretic minimum
    # No hard reference — this is an empirical starting point to calibrate.
    # Malach & Shalev-Shwartz (2019) show poly complexity but don't give constants.
    # Start here, then fit to your experiments.
    nn_inefficiency = 100

    samples = int(nn_inefficiency * width * (depth ** 2) * n_gates * noise_factor)

    return samples


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
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(funcName)s - %(message)s"
    )
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    if log_path:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def visualize_as_grid(objs, obj_type="", names=None, path=None, **kwargs):
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
    for ax in axes[len(objs) :]:
        ax.axis("off")

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
    if node_size == "small":
        return 500
    elif node_size == "medium":
        return 1000
    elif node_size == "large":
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


