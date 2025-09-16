import json
import logging
import os
import pickle
import tempfile
import numpy as np
import pytest
import torch
from unittest.mock import Mock, patch, mock_open
from matplotlib import pyplot as plt

from identifiability_toy_study.common.utils import (
    setup_logging,
    visualize_as_grid,
    get_node_size,
    set_seeds,
    load_model,
    load_binary,
    save_binary,
    powerset,
    deterministic_id_from_dataclass,
    _qfloat,
    _canon,
)


class TestSetupLogging:
    def test_setup_logging_console_only(self):
        """Test logging setup with console handler only"""
        logger = setup_logging()

        assert logger.level == logging.INFO
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], logging.StreamHandler)

    def test_setup_logging_with_debug(self):
        """Test logging setup with debug level"""
        logger = setup_logging(debug=True)

        assert logger.level == logging.DEBUG

    def test_setup_logging_with_file(self):
        """Test logging setup with file handler"""
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            try:
                logger = setup_logging(log_path=tmp_file.name)

                assert len(logger.handlers) == 2
                assert any(isinstance(h, logging.FileHandler) for h in logger.handlers)
            finally:
                os.unlink(tmp_file.name)


class TestVisualizeAsGrid:
    def test_visualize_as_grid_basic(self):
        """Test basic grid visualization"""
        mock_obj1 = Mock()
        mock_obj1.visualize = Mock()
        mock_obj2 = Mock()
        mock_obj2.visualize = Mock()

        objs = [mock_obj1, mock_obj2]

        with patch('matplotlib.pyplot.subplots') as mock_subplots, \
             patch('matplotlib.pyplot.tight_layout'), \
             patch('matplotlib.pyplot.show'):

            mock_fig = Mock()
            mock_axes = [Mock(), Mock(), Mock(), Mock()]
            for ax in mock_axes:
                ax.axis = Mock()
            mock_subplots.return_value = (mock_fig, np.array(mock_axes))

            visualize_as_grid(objs)

            mock_obj1.visualize.assert_called_once()
            mock_obj2.visualize.assert_called_once()
            mock_subplots.assert_called_once()

    def test_visualize_as_grid_with_names(self):
        """Test grid visualization with names"""
        mock_obj = Mock()
        mock_obj.visualize = Mock()
        objs = [mock_obj]
        names = {0: "Test Object"}

        with patch('matplotlib.pyplot.subplots') as mock_subplots, \
             patch('matplotlib.pyplot.tight_layout'), \
             patch('matplotlib.pyplot.show'):

            mock_fig = Mock()
            mock_ax = Mock()
            mock_ax.set_title = Mock()
            mock_ax.axis = Mock()
            mock_subplots.return_value = (mock_fig, np.array([mock_ax]))

            visualize_as_grid(objs, obj_type="TestType", names=names)

            mock_ax.set_title.assert_called_once_with("TestType Test Object")

    def test_visualize_as_grid_save_path(self):
        """Test grid visualization with save path"""
        mock_obj = Mock()
        mock_obj.visualize = Mock()
        objs = [mock_obj]

        with patch('matplotlib.pyplot.subplots') as mock_subplots, \
             patch('matplotlib.pyplot.tight_layout'), \
             patch('matplotlib.pyplot.savefig') as mock_savefig, \
             patch('matplotlib.pyplot.close'):

            mock_fig = Mock()
            mock_ax = Mock()
            mock_ax.axis = Mock()
            mock_subplots.return_value = (mock_fig, np.array([mock_ax]))

            visualize_as_grid(objs, path="test.png")

            mock_savefig.assert_called_once_with("test.png")


class TestGetNodeSize:
    def test_get_node_size_small(self):
        """Test small node size"""
        assert get_node_size("small") == 500

    def test_get_node_size_medium(self):
        """Test medium node size"""
        assert get_node_size("medium") == 1000

    def test_get_node_size_large(self):
        """Test large node size"""
        assert get_node_size("large") == 1400

    def test_get_node_size_invalid(self):
        """Test invalid node size"""
        with pytest.raises(ValueError, match="Unknown node size"):
            get_node_size("invalid")


class TestSetSeeds:
    def test_set_seeds(self):
        """Test seed setting"""
        # This should not raise any exceptions
        set_seeds(42)

        # Check that seeds are actually set
        import random
        random_val1 = random.random()
        np_val1 = np.random.random()
        torch_val1 = torch.rand(1).item()

        # Reset seeds and check reproducibility
        set_seeds(42)
        random_val2 = random.random()
        np_val2 = np.random.random()
        torch_val2 = torch.rand(1).item()

        assert random_val1 == random_val2
        assert np_val1 == np_val2
        assert torch_val1 == torch_val2


class TestLoadModel:
    @patch('identifiability_toy_study.common.utils.MLP')
    def test_load_model_success(self, mock_mlp_class):
        """Test successful model loading"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a mock model file
            model_file = os.path.join(temp_dir, "test_model.pt")
            with open(model_file, "w") as f:
                f.write("dummy content")

            mock_model = Mock()
            mock_mlp_class.load_from_file.return_value = mock_model

            result = load_model(temp_dir)

            assert result == mock_model
            mock_mlp_class.load_from_file.assert_called_once_with(model_file)

    def test_load_model_no_model(self):
        """Test model loading when no model exists"""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(AssertionError, match="Not a single model found"):
                load_model(temp_dir)

    def test_load_model_multiple_models(self):
        """Test model loading when multiple models exist"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create multiple model files
            with open(os.path.join(temp_dir, "model1.pt"), "w") as f:
                f.write("dummy")
            with open(os.path.join(temp_dir, "model2.pt"), "w") as f:
                f.write("dummy")

            with pytest.raises(AssertionError, match="Not a single model found"):
                load_model(temp_dir)


class TestLoadBinary:
    def test_load_binary_success(self):
        """Test successful binary loading"""
        test_data = {"key": "value", "number": 42}

        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            try:
                tmp_file.write(pickle.dumps(test_data))
                tmp_file.flush()

                result = load_binary(tmp_file.name)

                assert result == test_data
            finally:
                os.unlink(tmp_file.name)

    def test_load_binary_file_not_found(self):
        """Test binary loading with non-existent file"""
        with pytest.raises(FileNotFoundError):
            load_binary("non_existent_file.pkl")


class TestSaveBinary:
    def test_save_binary_success(self):
        """Test successful binary saving"""
        test_data = {"key": "value", "number": 42}

        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            try:
                save_binary(test_data, tmp_file.name)

                # Verify the file was written correctly
                with open(tmp_file.name, 'rb') as f:
                    loaded_data = pickle.loads(f.read())

                assert loaded_data == test_data
            finally:
                os.unlink(tmp_file.name)


class TestPowerset:
    def test_powerset_empty(self):
        """Test powerset of empty iterable"""
        result = list(powerset([]))
        assert result == [()]

    def test_powerset_single_element(self):
        """Test powerset of single element"""
        result = list(powerset([1]))
        expected = [(), (1,)]
        assert result == expected

    def test_powerset_multiple_elements(self):
        """Test powerset of multiple elements"""
        result = list(powerset([1, 2]))
        expected = [(), (1,), (2,), (1, 2)]
        assert result == expected

    def test_powerset_string(self):
        """Test powerset of string"""
        result = list(powerset("ab"))
        expected = [(), ('a',), ('b',), ('a', 'b')]
        assert result == expected


class TestQfloat:
    def test_qfloat_basic(self):
        """Test basic qfloat functionality"""
        result = _qfloat(3.14159265, places=4)
        assert abs(result - 3.1416) < 1e-6

    def test_qfloat_negative_zero(self):
        """Test qfloat handling of negative zero"""
        result = _qfloat(-0.0)
        assert result == 0.0

    def test_qfloat_rounding(self):
        """Test qfloat rounding behavior"""
        result = _qfloat(1.23456789, places=3)
        assert abs(result - 1.235) < 1e-6


class TestCanon:
    def test_canon_float(self):
        """Test canonical representation of float"""
        result = _canon(3.14159, places=3)
        assert isinstance(result, float)

    def test_canon_nan(self):
        """Test canonical representation of NaN"""
        result = _canon(float('nan'))
        assert result == "NaN"

    def test_canon_dict(self):
        """Test canonical representation of dict"""
        data = {"a": 1.23456, "b": 2.0}
        result = _canon(data, places=3)
        assert isinstance(result, dict)
        assert "a" in result
        assert "b" in result

    def test_canon_list(self):
        """Test canonical representation of list"""
        data = [1.23456, 2.0, 3]
        result = _canon(data, places=3)
        assert isinstance(result, list)
        assert len(result) == 3

    def test_canon_tuple(self):
        """Test canonical representation of tuple"""
        data = (1.23456, 2.0)
        result = _canon(data, places=3)
        assert isinstance(result, list)  # tuples become lists
        assert len(result) == 2

    def test_canon_other_types(self):
        """Test canonical representation of other types"""
        assert _canon("string") == "string"
        assert _canon(42) == 42
        assert _canon(None) is None

    def test_canon_dataclass(self):
        """Test canonical representation of dataclass"""
        from dataclasses import dataclass

        @dataclass
        class TestClass:
            x: float
            y: int

        obj = TestClass(x=1.23456, y=42)
        result = _canon(obj, places=3)
        assert isinstance(result, dict)
        assert "x" in result
        assert "y" in result


class TestDeterministicIdFromDataclass:
    def test_deterministic_id_consistency(self):
        """Test that same dataclass produces same ID"""
        from dataclasses import dataclass

        @dataclass
        class TestClass:
            x: float
            y: int

        obj1 = TestClass(x=1.23456, y=42)
        obj2 = TestClass(x=1.23456, y=42)

        id1 = deterministic_id_from_dataclass(obj1)
        id2 = deterministic_id_from_dataclass(obj2)

        assert id1 == id2
        assert isinstance(id1, str)
        assert len(id1) == 32  # 16 bytes * 2 hex chars

    def test_deterministic_id_different_objects(self):
        """Test that different dataclasses produce different IDs"""
        from dataclasses import dataclass

        @dataclass
        class TestClass:
            x: float
            y: int

        obj1 = TestClass(x=1.23456, y=42)
        obj2 = TestClass(x=1.23457, y=42)

        id1 = deterministic_id_from_dataclass(obj1)
        id2 = deterministic_id_from_dataclass(obj2)

        assert id1 != id2

    def test_deterministic_id_float_precision(self):
        """Test that float precision affects ID"""
        from dataclasses import dataclass

        @dataclass
        class TestClass:
            x: float

        obj1 = TestClass(x=1.123456789)
        obj2 = TestClass(x=1.123456790)

        # With default precision, these should be the same
        id1 = deterministic_id_from_dataclass(obj1, places=8)
        id2 = deterministic_id_from_dataclass(obj2, places=8)

        # But with higher precision, they should be different
        id3 = deterministic_id_from_dataclass(obj1, places=10)
        id4 = deterministic_id_from_dataclass(obj2, places=10)

        assert id1 == id2  # Same with 8 decimal places
        assert id3 != id4  # Different with 10 decimal places

    def test_deterministic_id_custom_digest_bytes(self):
        """Test custom digest bytes parameter"""
        from dataclasses import dataclass

        @dataclass
        class TestClass:
            x: int

        obj = TestClass(x=42)

        id_16 = deterministic_id_from_dataclass(obj, digest_bytes=16)
        id_8 = deterministic_id_from_dataclass(obj, digest_bytes=8)

        assert len(id_16) == 32  # 16 bytes * 2 hex chars
        assert len(id_8) == 16   # 8 bytes * 2 hex chars

    def test_deterministic_id_with_nan(self):
        """Test that NaN values are handled correctly"""
        from dataclasses import dataclass

        @dataclass
        class TestClass:
            x: float

        obj1 = TestClass(x=float('nan'))
        obj2 = TestClass(x=float('nan'))

        id1 = deterministic_id_from_dataclass(obj1)
        id2 = deterministic_id_from_dataclass(obj2)

        # NaN values should produce consistent IDs
        assert id1 == id2


class TestIntegration:
    def test_logging_and_visualization_integration(self):
        """Integration test combining logging and visualization"""
        mock_obj = Mock()
        mock_obj.visualize = Mock()

        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            try:
                logger = setup_logging(log_path=tmp_file.name)
                logger.info("Starting visualization test")

                with patch('matplotlib.pyplot.subplots') as mock_subplots, \
                     patch('matplotlib.pyplot.tight_layout'), \
                     patch('matplotlib.pyplot.show'):

                    mock_fig = Mock()
                    mock_ax = Mock()
                    mock_ax.axis = Mock()
                    mock_subplots.return_value = (mock_fig, np.array([mock_ax]))

                    visualize_as_grid([mock_obj])

                logger.info("Visualization test completed")

                # Verify log file was written
                assert os.path.exists(tmp_file.name)
                assert os.path.getsize(tmp_file.name) > 0

            finally:
                os.unlink(tmp_file.name)

    def test_binary_save_load_integration(self):
        """Integration test for binary save/load operations"""
        test_data = {
            "float_val": 3.14159,
            "int_val": 42,
            "list_val": [1, 2, 3],
            "nested": {"key": "value"}
        }

        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            try:
                # Save and load
                save_binary(test_data, tmp_file.name)
                loaded_data = load_binary(tmp_file.name)

                assert loaded_data == test_data

            finally:
                os.unlink(tmp_file.name)

    def test_deterministic_id_consistency_across_operations(self):
        """Test that deterministic IDs remain consistent across different operations"""
        from dataclasses import dataclass

        @dataclass
        class TestClass:
            values: list
            factor: float

        original_data = TestClass(values=[1, 2, 3], factor=1.5)

        # Create multiple copies and verify they have the same ID
        id1 = deterministic_id_from_dataclass(original_data)

        # Save and load the dataclass
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            try:
                save_binary(original_data, tmp_file.name)
                loaded_data = load_binary(tmp_file.name)

                id2 = deterministic_id_from_dataclass(loaded_data)
                assert id1 == id2

            finally:
                os.unlink(tmp_file.name)