from dataclasses import dataclass
from typing import Literal, Optional

import torch

Axis = Literal["neuron", "edge"]
Mode = Literal["set", "mul", "add"]


@dataclass(frozen=True)
class PatchShape:
    """
    Where to patch.

    axis=="neuron": patch activations h^{(L)} at layer index L (0=input, …, L=logits).
                    'indices' are neuron columns at those L. 'layers' may contain many L.
    axis=="edge"  : patch Linear weights W^{(L)} for layer index L in 0..L-1.
                    'indices' unused (pass ()); values must broadcast to W^{(L)}.

    If multiple layers are given, the shape expands to one per layer.
    """

    layers: tuple[int, ...]
    indices: tuple[int, ...] = ()
    axis: Axis = "neuron"

    def single_layers(self) -> tuple[int, ...]:
        return tuple(self.layers)

    @property
    def is_multi(self) -> bool:
        return len(self.layers) > 1

    def for_layer(self, layer: int) -> "PatchShape":
        return PatchShape(layers=(layer,), indices=self.indices, axis=self.axis)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "layers": list(self.layers),
            "indices": list(self.indices),
            "axis": self.axis,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PatchShape":
        """Create from dict."""
        return cls(
            layers=tuple(data["layers"]),
            indices=tuple(data.get("indices", ())),
            axis=data.get("axis", "neuron"),
        )


@dataclass
class Intervention:
    """
    A bag of patches. Each entry maps PatchShape -> (mode, values).

    - axis=="neuron": values broadcast to [B, k], k=len(indices)
    - axis=="edge"  : values broadcast to W^{(L)} shape [out, in]
    - mode: "set" | "mul" | "add" (elementwise)
    """

    patches: dict[PatchShape, tuple[Mode, torch.Tensor]] = None

    def __post_init__(self):
        if self.patches is None:
            self.patches = {}

    def to(
        self, device: torch.device, dtype: Optional[torch.dtype] = None
    ) -> "Intervention":
        new = {}
        for ps, (mode, v) in self.patches.items():
            new[ps] = (mode, v.to(device=device, dtype=(dtype or v.dtype)))
        return Intervention(patches=new)

    def merge(self, other: "Intervention") -> "Intervention":
        merged = dict(self.patches)
        merged.update(other.patches)  # 'other' wins on conflicts
        return Intervention(patches=merged)

    @staticmethod
    def from_states_dict(
        states: dict[tuple[int, tuple[int, ...]], torch.Tensor],
        mode: Mode = "set",
        axis: Axis = "neuron",
    ) -> "Intervention":
        patches = {}
        for (layer, idxs), vals in states.items():
            ps = PatchShape(layers=(layer,), indices=tuple(idxs), axis=axis)
            patches[ps] = (mode, vals)
        return Intervention(patches=patches)

    @staticmethod
    def create_random_interventions(
        patch_shape: PatchShape,
        n_interventions: int = 10,
        mode: Mode = "add",
        value_range: Optional[list] = None,
        device: str = "cpu",
    ) -> list["Intervention"]:
        """Create n random interventions for the given patch shape.

        For neuron axis: samples values uniformly from value_range for each neuron index.

        Args:
            value_range: Either [min, max] for single range, or [[min1, max1], [min2, max2], ...]
                        for union of ranges (samples uniformly across the union).
        """
        if value_range is None:
            value_range = [0.1, 0.5]  # [min, max]

        # Check if value_range is a list of ranges (union) or single range
        is_union = isinstance(value_range[0], (list, tuple))

        def sample_from_range(shape, device):
            """Sample uniformly from value_range (single or union of ranges)."""
            if not is_union:
                min_val, max_val = value_range
                return torch.empty(shape, device=device).uniform_(min_val, max_val)
            else:
                # Union of ranges: sample uniformly across all ranges
                # Weight each range by its width for uniform sampling
                ranges = value_range
                widths = [r[1] - r[0] for r in ranges]
                total_width = sum(widths)
                probs = [w / total_width for w in widths]

                # Sample which range to use for each element
                result = torch.empty(shape, device=device)
                flat_result = result.view(-1)
                for i in range(flat_result.numel()):
                    # Select range based on probability
                    r = ranges[torch.multinomial(torch.tensor(probs), 1).item()]
                    flat_result[i] = torch.empty(1).uniform_(r[0], r[1]).item()
                return result

        interventions = []

        for _ in range(n_interventions):
            patches = {}
            if patch_shape.axis == "neuron":
                k = len(patch_shape.indices)
                for layer in patch_shape.single_layers():
                    # Sample random values for each neuron in indices
                    vals = sample_from_range((1, k), device)
                    single_ps = PatchShape(
                        layers=(layer,), indices=patch_shape.indices, axis="neuron"
                    )
                    patches[single_ps] = (mode, vals)
            else:
                # Edge interventions - just use scalar broadcast
                for layer in patch_shape.single_layers():
                    val = sample_from_range((1,), device)
                    single_ps = PatchShape(layers=(layer,), indices=(), axis="edge")
                    patches[single_ps] = (mode, val)

            interventions.append(Intervention(patches=patches))

        return interventions

    # Grouping utilities the MLP will use
    def group_neuron_by_layer(
        self,
    ) -> dict[int, list[tuple[Mode, tuple[int, ...], torch.Tensor]]]:
        grouped: dict[int, list[tuple[Mode, tuple[int, ...], torch.Tensor]]] = {}
        for ps, (mode, vals) in self.patches.items():
            if ps.axis != "neuron":
                continue
            for L in ps.single_layers():
                grouped.setdefault(L, []).append((mode, ps.indices, vals))
        return grouped

    def group_edge_by_layer(self) -> dict[int, list[tuple[Mode, torch.Tensor]]]:
        grouped: dict[int, list[tuple[Mode, torch.Tensor]]] = {}
        for ps, (mode, vals) in self.patches.items():
            if ps.axis != "edge":
                continue
            for L in ps.single_layers():
                grouped.setdefault(L, []).append((mode, vals))
        return grouped

    def to_device(self, device: str) -> "Intervention":
        """Move all tensor values to the specified device."""
        new_patches = {}
        for ps, (mode, v) in self.patches.items():
            new_patches[ps] = (mode, v.to(device=device))
        return Intervention(patches=new_patches)


@dataclass
class InterventionEffect:
    """Result of applying an intervention and comparing target vs proxy outputs.

    NOTE: Activations are pre-computed here so visualization code
    NEVER needs to run models. Visualization is READ-ONLY.
    """

    intervention: "Intervention"
    y_target: torch.Tensor
    y_proxy: torch.Tensor
    logit_similarity: float  # 1 - MSE between logits
    bit_similarity: float  # Match rate after rounding
    best_similarity: float  # Match rate after clamping to binary

    # Pre-computed activations for visualization (NO model runs during viz!)
    # These are lists of tensors: [layer0_acts, layer1_acts, ...]
    # AFTER intervention (for visualizing the intervened state)
    target_activations: list = None  # type: ignore
    proxy_activations: list = None  # type: ignore
    # BEFORE intervention (for showing two-value comparison in viz)
    original_target_activations: list = None  # type: ignore
    original_proxy_activations: list = None  # type: ignore
