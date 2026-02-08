"""Sample schema classes for robustness and intervention testing.

Contains sample dataclasses for storing test results:
- InterventionSample: Result of a single intervention test
- RobustnessSample: Result of a single robustness test (a.k.a. ObservationalSample)
- SampleType: Enum for robustness sample types
"""

from dataclasses import dataclass, field
from enum import Enum

from .base import SchemaClass


class SampleType(str, Enum):
    """Types of samples for robustness testing.

    Inherits from str for JSON serialization compatibility.
    """
    NOISE = "noise"
    MULTIPLY_POSITIVE = "multiply_positive"
    MULTIPLY_NEGATIVE = "multiply_negative"
    ADD = "add"
    SUBTRACT = "subtract"
    BIMODAL = "bimodal"
    BIMODAL_INV = "bimodal_inv"


@dataclass
class InterventionSample(SchemaClass):
    """Result of a single intervention test (like RobustnessSample for faithfulness).

    NOTE: Activations are pre-computed here so visualization code
    NEVER needs to run models. Visualization is READ-ONLY.
    """

    # Patch info
    patch_key: str  # String representation of PatchShape
    patch_layer: int  # Layer index
    patch_indices: list[int]  # Neuron indices

    # Intervention values (what we patched in)
    intervention_values: list[float]

    # Outputs
    gate_output: float  # Output from full gate model under intervention
    subcircuit_output: float  # Output from subcircuit under intervention

    # Agreement metrics
    logit_similarity: float  # 1 - MSE between outputs
    bit_agreement: bool  # round(gate_output) == round(subcircuit_output)
    mse: float  # (gate_output - subcircuit_output)^2

    # Pre-computed activations for visualization (NO model runs during viz!)
    # These are the activations AFTER the intervention/patch is applied
    gate_activations: list[list[float]] = field(default_factory=list)
    subcircuit_activations: list[list[float]] = field(default_factory=list)
    # Original activations BEFORE the intervention (for showing two-value display)
    original_gate_activations: list[list[float]] = field(default_factory=list)
    original_subcircuit_activations: list[list[float]] = field(default_factory=list)


@dataclass
class RobustnessSample(SchemaClass):
    """Result of a single robustness test on one input.

    NOTE: Activations are pre-computed here so visualization code
    NEVER needs to run models. Visualization is READ-ONLY.

    Sample types:
    - noise: Gaussian noise perturbation
    - multiply_positive: Scale by factor > 1
    - multiply_negative: Scale by factor < 0
    - add: Add large positive value
    - subtract: Subtract large value
    - bimodal: Map [0,1] -> [-1,1] order-preserving (0->-1, 1->1)
    - bimodal_inv: Map [0,1] -> [-1,1] inverted (0->1, 1->-1)
    """

    input_values: list[float]  # The perturbed input [x0, x1]
    base_input: list[float]  # The original binary input [0, 1]
    noise_magnitude: float  # L2 norm of noise or transformation magnitude
    ground_truth: float  # GT output for base input (e.g., XOR(0,1)=1)

    # Outputs from both models on the SAME perturbed input
    gate_output: float  # Output from gate_model(perturbed_input)
    subcircuit_output: float  # Output from subcircuit(perturbed_input)

    # Accuracy to ground truth
    gate_correct: bool  # round(gate_output) == ground_truth
    subcircuit_correct: bool  # round(subcircuit_output) == ground_truth

    # Agreement between models
    agreement_bit: bool  # round(gate_output) == round(subcircuit_output)
    agreement_best: bool  # clamp_to_binary(gate) == clamp_to_binary(subcircuit)
    mse: float  # (gate_output - subcircuit_output)^2

    # Sample type for organizing visualizations
    sample_type: str = SampleType.NOISE  # Can be string or SampleType enum

    # Pre-computed activations for visualization (NO model runs during viz!)
    # Each is a list of lists: [[layer0_acts], [layer1_acts], ...]
    gate_activations: list[list[float]] = field(default_factory=list)
    subcircuit_activations: list[list[float]] = field(default_factory=list)


# Alias for semantic clarity - RobustnessSample can also be called ObservationalSample
ObservationalSample = RobustnessSample
