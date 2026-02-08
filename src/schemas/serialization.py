"""Serialization utilities for schema dataclasses.

These functions handle JSON serialization and deterministic ID generation
for dataclass objects, filtering out non-serializable fields like nn.Module and Tensor.
"""

import hashlib
import json
import math
from dataclasses import asdict, is_dataclass
from decimal import ROUND_HALF_EVEN, Decimal
from typing import Any

import torch


def filter_non_serializable(obj):
    """Recursively filter out non-JSON-serializable objects (like nn.Module, Tensor)."""
    if isinstance(obj, dict):
        return {
            k: filter_non_serializable(v)
            for k, v in obj.items()
            if v is not None and not isinstance(v, torch.nn.Module)
        }
    elif isinstance(obj, list):
        return [
            filter_non_serializable(item)
            for item in obj
            if not isinstance(item, torch.nn.Module)
        ]
    elif isinstance(obj, torch.nn.Module):
        return None
    elif isinstance(obj, torch.Tensor):
        return None  # Don't serialize tensors to JSON - use tensors.pt
    elif is_dataclass(obj) and not isinstance(obj, type):
        # Handle dataclass instances - check for to_dict method first
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        # Otherwise convert to dict and filter
        return filter_non_serializable(asdict(obj))
    return obj


def _qfloat(x: float, places: int = 8) -> float:
    """Stable decimal rounding: converts via str -> Decimal -> quantize."""
    q = Decimal(1) / (Decimal(10) ** places)  # e.g. 1e-8
    d = Decimal(str(x)).quantize(q, rounding=ROUND_HALF_EVEN)
    # normalize -0.0 to 0.0 for stability
    f = float(d)
    return 0.0 if f == 0.0 else f


def _canon(obj: Any, places: int = 8):
    """Canonicalize an object for deterministic hashing."""
    if isinstance(obj, float):
        if math.isnan(obj):  # avoid NaN destabilizing hashes
            return "NaN"
        return _qfloat(obj, places)
    if is_dataclass(obj):
        # Filter non-serializable fields (nn.Module, Tensor) before canonicalizing
        return _canon(filter_non_serializable(asdict(obj)), places)
    if isinstance(obj, dict):
        return {k: _canon(v, places) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_canon(v, places) for v in obj]
    return obj


def deterministic_id_from_dataclass(
    data_class_obj: Any, places: int = 8, digest_bytes: int = 16
) -> str:
    """Generate a deterministic ID from a dataclass object.

    Uses blake2b hash of the canonicalized JSON representation.
    """
    canonical = _canon(data_class_obj, places)
    payload = json.dumps(
        canonical,
        sort_keys=True,  # stable key order
        separators=(",", ":"),  # remove whitespace
        ensure_ascii=False,
        allow_nan=False,
    )
    # fast, strong hash in the stdlib
    h = hashlib.blake2b(payload.encode("utf-8"), digest_size=digest_bytes)
    return h.hexdigest()
