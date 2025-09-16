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


@dataclass
class Intervention:
    """
    A bag of patches. Each entry maps PatchShape -> (mode, values).

    - axis=="neuron": values broadcast to [B, k], k=len(indices)
    - axis=="edge"  : values broadcast to W^{(L)} shape [out, in]
    - mode: "set" | "mul" | "add" (elementwise)
    """

    patches: dict[PatchShape, tuple[Mode, torch.Tensor]]

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
