"""Base schema class for all dataclasses.

SchemaClass provides:
- Deterministic ID generation based on field values
- JSON-serializable string representation
- Deep copy semantics for immutability
"""

import copy
import json
from dataclasses import asdict, dataclass, fields

from ..utils import deterministic_id_from_dataclass, filter_non_serializable


@dataclass
class SchemaClass:
    # Each schema gets unique id based on values
    def get_id(self) -> str:
        return deterministic_id_from_dataclass(self)

    # For logging ease - filters out non-serializable fields like nn.Module
    def __str__(self) -> str:
        result_dict = asdict(self)
        filtered = filter_non_serializable(result_dict)
        return json.dumps(filtered, indent=4)

    # Each trial should have their own set of params
    # We want to make sure schemas are unique and immutable
    def __post_init__(self):
        for f in fields(self):
            if hasattr(self, f.name):
                setattr(self, f.name, copy.deepcopy(getattr(self, f.name)))

    def __copy__(self):
        return self.__deepcopy__({})

    def __deepcopy__(self, memo):
        cls = self.__class__
        kwargs = {
            f.name: copy.deepcopy(getattr(self, f.name), memo) for f in fields(self)
        }
        return cls(**kwargs)

    def __setattr__(self, name, value):
        super().__setattr__(name, copy.deepcopy(value))
