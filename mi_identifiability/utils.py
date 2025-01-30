import logging
import os
import pickle
import random
from itertools import chain, combinations

import numpy as np
import torch
from matplotlib import pyplot as plt


def setup_logging(output_dir=None, log_file=None):
    """
    Create a custom logger with a stream handler and a file handler

    Args:
        output_dir: The directory to save the log file
        log_file: The name of the log file

    Returns:
        The logger
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    if log_file and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        log_path = os.path.join(output_dir, log_file)
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
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups

    # Ensure deterministic behavior for PyTorch
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
