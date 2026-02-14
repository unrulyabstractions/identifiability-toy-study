"""Base schema class for all dataclasses.

SchemaClass provides:
- Deterministic ID generation based on field values
- JSON-serializable string representation
- Standard copy semantics (explicit deepcopy when needed)
"""

import copy
import json
from dataclasses import asdict, dataclass, fields

from src.serialization import deterministic_id_from_dataclass, filter_non_serializable


@dataclass
class SchemaClass:
    def __post_init__(self):
        """Base post_init - subclasses can call super().__post_init__() safely."""
        pass

    # Each schema gets unique id based on values
    def get_id(self) -> str:
        return deterministic_id_from_dataclass(self)

    # For logging ease - filters out non-serializable fields like nn.Module
    def __str__(self) -> str:
        result_dict = asdict(self)
        filtered = filter_non_serializable(result_dict)
        return json.dumps(filtered, indent=4)

    def __copy__(self):
        return self.__deepcopy__({})

    def __deepcopy__(self, memo):
        cls = self.__class__
        kwargs = {
            f.name: copy.deepcopy(getattr(self, f.name), memo) for f in fields(self)
        }
        return cls(**kwargs)
