import pytest
import torch
from dataclasses import dataclass

from identifiability_toy_study.common.causal import (
    PatchShape,
    Intervention,
    Axis,
    Mode,
)


class TestPatchShape:
    def test_patch_shape_initialization(self):
        """Test PatchShape initialization"""
        shape = PatchShape(layers=(1, 2), indices=(0, 1), axis="neuron")

        assert shape.layers == (1, 2)
        assert shape.indices == (0, 1)
        assert shape.axis == "neuron"

    def test_patch_shape_default_values(self):
        """Test PatchShape default values"""
        shape = PatchShape(layers=(1,))

        assert shape.indices == ()
        assert shape.axis == "neuron"

    def test_patch_shape_single_layers(self):
        """Test single_layers method"""
        shape = PatchShape(layers=(1, 2, 3))

        result = shape.single_layers()

        assert result == (1, 2, 3)

    def test_patch_shape_is_multi(self):
        """Test is_multi property"""
        shape_single = PatchShape(layers=(1,))
        shape_multi = PatchShape(layers=(1, 2))

        assert not shape_single.is_multi
        assert shape_multi.is_multi

    def test_patch_shape_for_layer(self):
        """Test for_layer method"""
        shape = PatchShape(layers=(1, 2), indices=(0, 1), axis="edge")

        new_shape = shape.for_layer(3)

        assert new_shape.layers == (3,)
        assert new_shape.indices == (0, 1)
        assert new_shape.axis == "edge"

    def test_patch_shape_frozen(self):
        """Test that PatchShape is frozen (immutable)"""
        shape = PatchShape(layers=(1,))

        with pytest.raises(Exception):  # Should raise some kind of frozen/immutable error
            shape.layers = (2,)

    def test_patch_shape_equality(self):
        """Test PatchShape equality"""
        shape1 = PatchShape(layers=(1, 2), indices=(0, 1), axis="neuron")
        shape2 = PatchShape(layers=(1, 2), indices=(0, 1), axis="neuron")
        shape3 = PatchShape(layers=(1, 2), indices=(0, 1), axis="edge")

        assert shape1 == shape2
        assert shape1 != shape3

    def test_patch_shape_hashable(self):
        """Test that PatchShape is hashable (can be used as dict key)"""
        shape = PatchShape(layers=(1,), indices=(0, 1), axis="neuron")

        # Should not raise an error
        d = {shape: "value"}
        assert d[shape] == "value"


class TestIntervention:
    def test_intervention_initialization(self):
        """Test Intervention initialization"""
        patches = {
            PatchShape(layers=(1,), indices=(0, 1), axis="neuron"):
            ("set", torch.tensor([1.0, 2.0]))
        }

        intervention = Intervention(patches)

        assert len(intervention.patches) == 1
        assert isinstance(list(intervention.patches.keys())[0], PatchShape)

    def test_intervention_to_device(self):
        """Test moving intervention to device"""
        patches = {
            PatchShape(layers=(1,), indices=(0, 1), axis="neuron"):
            ("set", torch.tensor([1.0, 2.0]))
        }
        intervention = Intervention(patches)

        # Move to CPU (should work regardless of current device)
        new_intervention = intervention.to(torch.device("cpu"))

        assert isinstance(new_intervention, Intervention)
        for shape, (mode, tensor) in new_intervention.patches.items():
            assert tensor.device.type == "cpu"

    def test_intervention_to_device_with_dtype(self):
        """Test moving intervention to device with dtype conversion"""
        patches = {
            PatchShape(layers=(1,), indices=(0, 1), axis="neuron"):
            ("set", torch.tensor([1.0, 2.0], dtype=torch.float32))
        }
        intervention = Intervention(patches)

        new_intervention = intervention.to(torch.device("cpu"), dtype=torch.float64)

        for shape, (mode, tensor) in new_intervention.patches.items():
            assert tensor.device.type == "cpu"
            assert tensor.dtype == torch.float64

    def test_intervention_merge(self):
        """Test merging interventions"""
        patches1 = {
            PatchShape(layers=(1,), indices=(0,), axis="neuron"):
            ("set", torch.tensor([1.0]))
        }
        patches2 = {
            PatchShape(layers=(1,), indices=(1,), axis="neuron"):
            ("set", torch.tensor([2.0]))
        }

        intervention1 = Intervention(patches1)
        intervention2 = Intervention(patches2)

        merged = intervention1.merge(intervention2)

        assert len(merged.patches) == 2
        assert isinstance(merged, Intervention)

    def test_intervention_merge_conflict_resolution(self):
        """Test that merge resolves conflicts by letting 'other' win"""
        shape = PatchShape(layers=(1,), indices=(0,), axis="neuron")

        patches1 = {shape: ("set", torch.tensor([1.0]))}
        patches2 = {shape: ("set", torch.tensor([2.0]))}

        intervention1 = Intervention(patches1)
        intervention2 = Intervention(patches2)

        merged = intervention1.merge(intervention2)

        assert len(merged.patches) == 1
        # intervention2 should win
        assert torch.equal(merged.patches[shape][1], torch.tensor([2.0]))

    def test_intervention_from_states_dict(self):
        """Test creating intervention from states dictionary"""
        states = {
            (1, (0, 1)): torch.tensor([1.0, 2.0]),
            (2, (0,)): torch.tensor([3.0])
        }

        intervention = Intervention.from_states_dict(states, mode="set", axis="neuron")

        assert len(intervention.patches) == 2

        # Check that patches were created correctly
        found_patches = 0
        for shape, (mode, values) in intervention.patches.items():
            assert mode == "set"
            assert shape.axis == "neuron"
            if shape.layers == (1,) and shape.indices == (0, 1):
                assert torch.equal(values, torch.tensor([1.0, 2.0]))
                found_patches += 1
            elif shape.layers == (2,) and shape.indices == (0,):
                assert torch.equal(values, torch.tensor([3.0]))
                found_patches += 1

        assert found_patches == 2

    def test_intervention_from_states_dict_with_mode_and_axis(self):
        """Test creating intervention with custom mode and axis"""
        states = {(1, (0,)): torch.tensor([1.0])}

        intervention = Intervention.from_states_dict(states, mode="mul", axis="edge")

        shape, (mode, values) = list(intervention.patches.items())[0]
        assert mode == "mul"
        assert shape.axis == "edge"

    def test_intervention_group_neuron_by_layer(self):
        """Test grouping neuron patches by layer"""
        patches = {
            PatchShape(layers=(1,), indices=(0, 1), axis="neuron"):
            ("set", torch.tensor([1.0, 2.0])),
            PatchShape(layers=(1,), indices=(2,), axis="neuron"):
            ("mul", torch.tensor([3.0])),
            PatchShape(layers=(2,), indices=(0,), axis="neuron"):
            ("add", torch.tensor([4.0])),
            PatchShape(layers=(1,), indices=(), axis="edge"):
            ("set", torch.tensor([[1.0, 2.0]]))  # Should be ignored
        }

        intervention = Intervention(patches)
        grouped = intervention.group_neuron_by_layer()

        assert 1 in grouped
        assert 2 in grouped
        assert len(grouped[1]) == 2  # Two neuron patches for layer 1
        assert len(grouped[2]) == 1  # One neuron patch for layer 2

        # Check contents of layer 1
        layer1_patches = grouped[1]
        modes = [patch[0] for patch in layer1_patches]
        assert "set" in modes
        assert "mul" in modes

    def test_intervention_group_edge_by_layer(self):
        """Test grouping edge patches by layer"""
        patches = {
            PatchShape(layers=(0,), indices=(), axis="edge"):
            ("set", torch.tensor([[1.0, 2.0]])),
            PatchShape(layers=(1,), indices=(), axis="edge"):
            ("mul", torch.tensor([[3.0, 4.0]])),
            PatchShape(layers=(1,), indices=(0,), axis="neuron"):
            ("set", torch.tensor([5.0]))  # Should be ignored
        }

        intervention = Intervention(patches)
        grouped = intervention.group_edge_by_layer()

        assert 0 in grouped
        assert 1 in grouped
        assert len(grouped[0]) == 1
        assert len(grouped[1]) == 1

        # Check contents
        assert grouped[0][0][0] == "set"  # mode
        assert grouped[1][0][0] == "mul"  # mode

    def test_intervention_group_multi_layer_patches(self):
        """Test grouping patches that span multiple layers"""
        patches = {
            PatchShape(layers=(1, 2), indices=(0,), axis="neuron"):
            ("set", torch.tensor([1.0]))
        }

        intervention = Intervention(patches)
        grouped = intervention.group_neuron_by_layer()

        # Should create separate entries for each layer
        assert 1 in grouped
        assert 2 in grouped
        assert len(grouped[1]) == 1
        assert len(grouped[2]) == 1

        # Both should have the same patch content
        assert grouped[1][0] == grouped[2][0]

    def test_intervention_empty_patches(self):
        """Test intervention with no patches"""
        intervention = Intervention(patches={})

        grouped_neurons = intervention.group_neuron_by_layer()
        grouped_edges = intervention.group_edge_by_layer()

        assert len(grouped_neurons) == 0
        assert len(grouped_edges) == 0


class TestAxisAndMode:
    def test_axis_type_hints(self):
        """Test that Axis type exists and works"""
        # These should be valid axis values
        valid_axes = ["neuron", "edge"]

        for axis in valid_axes:
            shape = PatchShape(layers=(1,), axis=axis)
            assert shape.axis == axis

    def test_mode_type_hints(self):
        """Test that Mode type exists and works"""
        # These should be valid mode values
        valid_modes = ["set", "mul", "add"]

        for mode in valid_modes:
            patches = {
                PatchShape(layers=(1,), indices=(0,), axis="neuron"):
                (mode, torch.tensor([1.0]))
            }
            intervention = Intervention(patches)
            assert list(intervention.patches.values())[0][0] == mode


class TestIntegration:
    def test_patch_shape_intervention_integration(self):
        """Integration test for PatchShape and Intervention"""
        # Create multiple patch shapes
        shape1 = PatchShape(layers=(1,), indices=(0, 1), axis="neuron")
        shape2 = PatchShape(layers=(2,), indices=(0,), axis="edge")

        patches = {
            shape1: ("set", torch.tensor([1.0, 2.0])),
            shape2: ("mul", torch.tensor([[0.5, 0.8]]))
        }

        intervention = Intervention(patches)

        # Test grouping
        neuron_groups = intervention.group_neuron_by_layer()
        edge_groups = intervention.group_edge_by_layer()

        assert 1 in neuron_groups
        assert 2 in edge_groups
        assert len(neuron_groups[1]) == 1
        assert len(edge_groups[2]) == 1

    def test_intervention_complex_merge_scenario(self):
        """Test complex merge scenario with multiple conflicts"""
        # Create overlapping and non-overlapping patches
        shape1 = PatchShape(layers=(1,), indices=(0,), axis="neuron")
        shape2 = PatchShape(layers=(1,), indices=(1,), axis="neuron")
        shape3 = PatchShape(layers=(2,), indices=(), axis="edge")

        intervention1 = Intervention({
            shape1: ("set", torch.tensor([1.0])),
            shape2: ("mul", torch.tensor([2.0])),
            shape3: ("add", torch.tensor([[1.0, 2.0]]))
        })

        intervention2 = Intervention({
            shape1: ("mul", torch.tensor([3.0])),  # Conflict with intervention1
            shape3: ("set", torch.tensor([[3.0, 4.0]]))  # Conflict with intervention1
        })

        merged = intervention1.merge(intervention2)

        assert len(merged.patches) == 3

        # intervention2 should win conflicts
        assert merged.patches[shape1][0] == "mul"
        assert torch.equal(merged.patches[shape1][1], torch.tensor([3.0]))

        # No conflict for shape2, should keep intervention1's value
        assert merged.patches[shape2][0] == "mul"
        assert torch.equal(merged.patches[shape2][1], torch.tensor([2.0]))

        # intervention2 should win for shape3
        assert merged.patches[shape3][0] == "set"
        assert torch.equal(merged.patches[shape3][1], torch.tensor([[3.0, 4.0]]))

    def test_intervention_device_and_merge_integration(self):
        """Test device movement and merging together"""
        patches1 = {
            PatchShape(layers=(1,), indices=(0,), axis="neuron"):
            ("set", torch.tensor([1.0]))
        }
        patches2 = {
            PatchShape(layers=(1,), indices=(1,), axis="neuron"):
            ("set", torch.tensor([2.0]))
        }

        intervention1 = Intervention(patches1)
        intervention2 = Intervention(patches2)

        # Move to device and merge
        merged = intervention1.to(torch.device("cpu")).merge(
            intervention2.to(torch.device("cpu"))
        )

        assert len(merged.patches) == 2
        for shape, (mode, tensor) in merged.patches.items():
            assert tensor.device.type == "cpu"

    def test_from_states_dict_and_grouping_integration(self):
        """Test creating from states dict and then grouping"""
        states = {
            (1, (0, 1)): torch.tensor([1.0, 2.0]),
            (1, (2,)): torch.tensor([3.0]),
            (2, (0,)): torch.tensor([4.0])
        }

        intervention = Intervention.from_states_dict(states, mode="set", axis="neuron")
        grouped = intervention.group_neuron_by_layer()

        assert 1 in grouped
        assert 2 in grouped
        assert len(grouped[1]) == 2  # Two patches for layer 1
        assert len(grouped[2]) == 1  # One patch for layer 2

        # Verify the content
        layer1_indices = [patch[1] for patch in grouped[1]]
        assert (0, 1) in layer1_indices
        assert (2,) in layer1_indices