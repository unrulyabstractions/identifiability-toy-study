import glob
import os
import tempfile
import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from identifiability_toy_study.common.helpers import (
    update_status_fx,
    generate_dataset,
    generate_trial_data,
    train_model,
    load_model,
    save_model,
)
from identifiability_toy_study.common.schemas import (
    DataParams,
    ModelParams,
    TrainParams,
    TrialResult,
    TrialSetup,
    TrialData,
    Dataset,
    IdentifiabilityConstraints,
)


class TestUpdateStatusFx:
    def test_update_status_fx_basic(self):
        """Test basic update_status_fx functionality"""
        trial_result = Mock()
        trial_result.trial_id = "test_id"
        mock_logger = Mock()

        update_status = update_status_fx(trial_result, logger=mock_logger)

        update_status("RUNNING")

        assert trial_result.status == "RUNNING"
        mock_logger.info.assert_called()

    def test_update_status_fx_with_message(self):
        """Test update_status_fx with message"""
        trial_result = Mock()
        trial_result.trial_id = "test_id"
        mock_logger = Mock()

        update_status = update_status_fx(trial_result, logger=mock_logger)

        update_status("COMPLETED", mssg="Success")

        assert trial_result.status == "COMPLETED"
        mock_logger.info.assert_called()
        # Check that message is included in log call
        call_args = mock_logger.info.call_args[0][0]
        assert "Success" in call_args

    def test_update_status_fx_no_logger(self):
        """Test update_status_fx without logger"""
        trial_result = Mock()
        trial_result.trial_id = "test_id"

        update_status = update_status_fx(trial_result, logger=None)

        # Should not raise any errors
        update_status("RUNNING")

        assert trial_result.status == "RUNNING"


class TestGenerateDataset:
    @patch('identifiability_toy_study.common.helpers.ALL_LOGIC_GATES')
    def test_generate_dataset_single_gate(self, mock_all_gates):
        """Test dataset generation with single gate"""
        mock_gate = Mock()
        mock_gate.generate_noisy_data.return_value = (
            torch.tensor([[0.0, 1.0]], dtype=torch.float32),
            torch.tensor([[1.0]], dtype=torch.float32)
        )
        mock_all_gates.__getitem__.return_value = mock_gate

        dataset = generate_dataset(
            gate_names=["AND"],
            device="cpu",
            n_repeats=10,
            noise_std=0.1
        )

        assert isinstance(dataset, Dataset)
        mock_gate.generate_noisy_data.assert_called_once_with(
            n_repeats=10,
            weights=None,
            noise_std=0.1,
            device="cpu"
        )

    @patch('identifiability_toy_study.common.helpers.generate_noisy_multi_gate_data')
    @patch('identifiability_toy_study.common.helpers.ALL_LOGIC_GATES')
    def test_generate_dataset_multiple_gates(self, mock_all_gates, mock_multi_gate_data):
        """Test dataset generation with multiple gates"""
        mock_gate1 = Mock()
        mock_gate2 = Mock()
        mock_all_gates.__getitem__.side_effect = lambda name: {
            "AND": mock_gate1,
            "OR": mock_gate2
        }[name]

        mock_multi_gate_data.return_value = (
            torch.tensor([[0.0, 1.0]], dtype=torch.float32),
            torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        )

        dataset = generate_dataset(
            gate_names=["AND", "OR"],
            device="cpu",
            n_repeats=10,
            noise_std=0.1
        )

        assert isinstance(dataset, Dataset)
        mock_multi_gate_data.assert_called_once()

    @patch('identifiability_toy_study.common.helpers.ALL_LOGIC_GATES')
    def test_generate_dataset_with_weights(self, mock_all_gates):
        """Test dataset generation with custom weights"""
        mock_gate = Mock()
        mock_gate.generate_noisy_data.return_value = (
            torch.tensor([[0.0, 1.0]], dtype=torch.float32),
            torch.tensor([[1.0]], dtype=torch.float32)
        )
        mock_all_gates.__getitem__.return_value = mock_gate

        weights = [0.1, 0.3, 0.4, 0.2]
        dataset = generate_dataset(
            gate_names=["AND"],
            device="cpu",
            weights=weights
        )

        assert isinstance(dataset, Dataset)
        mock_gate.generate_noisy_data.assert_called_once()
        call_args = mock_gate.generate_noisy_data.call_args[1]
        assert call_args["weights"] == weights


class TestGenerateTrialData:
    @patch('identifiability_toy_study.common.helpers.generate_dataset')
    def test_generate_trial_data_basic(self, mock_generate_dataset):
        """Test basic trial data generation"""
        mock_dataset = Mock()
        mock_generate_dataset.return_value = mock_dataset

        data_params = DataParams(
            n_samples_train=1000,
            n_samples_val=200,
            n_samples_test=300,
            noise_std=0.1,
            skewed_distribution=False
        )

        trial_data = generate_trial_data(
            data_params=data_params,
            logic_gates=["AND"],
            device="cpu"
        )

        assert isinstance(trial_data, TrialData)
        assert trial_data.train == mock_dataset
        assert trial_data.val == mock_dataset
        assert trial_data.test == mock_dataset

        # Should be called 3 times (train, val, test)
        assert mock_generate_dataset.call_count == 3

    @patch('identifiability_toy_study.common.helpers.generate_dataset')
    @patch('random.random')
    def test_generate_trial_data_skewed_distribution(self, mock_random, mock_generate_dataset):
        """Test trial data generation with skewed distribution"""
        mock_dataset = Mock()
        mock_generate_dataset.return_value = mock_dataset
        mock_random.side_effect = [0.1, 0.3, 0.4, 0.2]  # 4 weights for 2^2 inputs

        mock_logger = Mock()

        data_params = DataParams(
            n_samples_train=1000,
            n_samples_val=200,
            n_samples_test=300,
            noise_std=0.1,
            skewed_distribution=True
        )

        trial_data = generate_trial_data(
            data_params=data_params,
            logic_gates=["AND"],
            device="cpu",
            logger=mock_logger
        )

        assert isinstance(trial_data, TrialData)
        mock_logger.info.assert_called_with("using skewed_distribution")

        # Check that weights were passed to generate_dataset
        for call in mock_generate_dataset.call_args_list:
            assert "weights" in call[1]
            assert call[1]["weights"] == [0.1, 0.3, 0.4, 0.2]


class TestTrainModel:
    @patch('identifiability_toy_study.common.helpers.MLP')
    def test_train_model_success(self, mock_mlp_class):
        """Test successful model training"""
        mock_model = Mock()
        mock_model.do_train.return_value = 0.005  # Low loss
        mock_model.do_eval.return_value = 0.96  # High accuracy
        mock_mlp_class.return_value = mock_model

        train_params = TrainParams(
            learning_rate=0.001,
            loss_target=0.01,
            acc_target=0.95,
            batch_size=32,
            epochs=100,
            val_frequency=10
        )
        model_params = ModelParams(width=4, depth=2)

        # Create mock data
        train_data = Mock()
        val_data = Mock()
        data = Mock()
        data.train = train_data
        data.val = val_data

        mock_status_fx = Mock()

        model, avg_loss, val_acc = train_model(
            train_params=train_params,
            model_params=model_params,
            data=data,
            device="cpu",
            status_fx=mock_status_fx
        )

        assert model == mock_model
        assert avg_loss == 0.005
        assert val_acc == 0.96

        # Check status updates
        mock_status_fx.assert_any_call("STARTED_TRAINING")
        mock_status_fx.assert_any_call("FINISHED_TRAINING", mssg="avg_loss=0.005 val_acc=0.96")

    @patch('identifiability_toy_study.common.helpers.MLP')
    def test_train_model_failure_low_accuracy(self, mock_mlp_class):
        """Test model training failure due to low accuracy"""
        mock_model = Mock()
        mock_model.do_train.return_value = 0.005
        mock_model.do_eval.return_value = 0.90  # Below target
        mock_mlp_class.return_value = mock_model

        train_params = TrainParams(
            learning_rate=0.001,
            loss_target=0.01,
            acc_target=0.95,
            batch_size=32,
            epochs=100,
            val_frequency=10
        )
        model_params = ModelParams(width=4, depth=2)

        data = Mock()
        data.train = Mock()
        data.val = Mock()

        mock_status_fx = Mock()

        model, avg_loss, val_acc = train_model(
            train_params=train_params,
            model_params=model_params,
            data=data,
            device="cpu",
            status_fx=mock_status_fx
        )

        assert model is None
        assert avg_loss == 0.005
        assert val_acc == 0.90

        mock_status_fx.assert_any_call("FAILED_TRAINING", mssg="avg_loss=0.005 val_acc=0.9")

    @patch('identifiability_toy_study.common.helpers.MLP')
    def test_train_model_failure_high_loss(self, mock_mlp_class):
        """Test model training failure due to high loss"""
        mock_model = Mock()
        mock_model.do_train.return_value = 0.05  # Above target
        mock_model.do_eval.return_value = 0.96
        mock_mlp_class.return_value = mock_model

        train_params = TrainParams(
            learning_rate=0.001,
            loss_target=0.01,
            acc_target=0.95,
            batch_size=32,
            epochs=100,
            val_frequency=10
        )
        model_params = ModelParams(width=4, depth=2)

        data = Mock()
        data.train = Mock()
        data.val = Mock()

        mock_status_fx = Mock()

        model, avg_loss, val_acc = train_model(
            train_params=train_params,
            model_params=model_params,
            data=data,
            device="cpu",
            status_fx=mock_status_fx
        )

        assert model is None


class TestLoadModel:
    @patch('identifiability_toy_study.common.helpers.ALL_LOGIC_GATES')
    @patch('identifiability_toy_study.common.helpers.MLP')
    @patch('torch.load')
    @patch('glob.glob')
    def test_load_model_success(self, mock_glob, mock_torch_load, mock_mlp_class, mock_all_gates):
        """Test successful model loading"""
        # Setup mocks
        mock_gate = Mock()
        mock_gate.n_inputs = 2
        mock_all_gates.__getitem__.return_value = mock_gate

        model_params = ModelParams(logic_gates=["AND"], width=4, depth=2)
        train_params = TrainParams(
            learning_rate=0.001, loss_target=0.01, acc_target=0.95,
            batch_size=32, epochs=100, val_frequency=10
        )

        # Mock file discovery
        model_id = model_params.get_id()
        mock_file_path = f"/models/model_{model_id}_loss_0.005000_acc_0.960000_20240101_120000.pt"
        mock_glob.return_value = [mock_file_path]

        # Mock torch.load
        mock_checkpoint = {
            "model_state_dict": {"layer1.weight": torch.tensor([[1.0, 2.0]])}
        }
        mock_torch_load.return_value = mock_checkpoint

        # Mock MLP
        mock_model = Mock()
        mock_mlp_class.return_value = mock_model

        mock_logger = Mock()

        result = load_model(
            model_dir="/models",
            model_params=model_params,
            train_params=train_params,
            device="cpu",
            logger=mock_logger
        )

        model, avg_loss, val_acc = result

        assert model == mock_model
        assert avg_loss == 0.005
        assert val_acc == 0.96

        mock_model.load_state_dict.assert_called_once_with(mock_checkpoint["model_state_dict"])

    @patch('glob.glob')
    def test_load_model_no_files(self, mock_glob):
        """Test model loading when no files are found"""
        mock_glob.return_value = []

        model_params = ModelParams(logic_gates=["AND"])
        train_params = TrainParams(
            learning_rate=0.001, loss_target=0.01, acc_target=0.95,
            batch_size=32, epochs=100, val_frequency=10
        )

        mock_logger = Mock()

        result = load_model(
            model_dir="/models",
            model_params=model_params,
            train_params=train_params,
            device="cpu",
            logger=mock_logger
        )

        assert result == (None, None, None)

    @patch('glob.glob')
    def test_load_model_parsing_error(self, mock_glob):
        """Test model loading with filename parsing errors"""
        mock_glob.return_value = ["invalid_filename.pt"]

        model_params = ModelParams(logic_gates=["AND"])
        train_params = TrainParams(
            learning_rate=0.001, loss_target=0.01, acc_target=0.95,
            batch_size=32, epochs=100, val_frequency=10
        )

        mock_logger = Mock()

        result = load_model(
            model_dir="/models",
            model_params=model_params,
            train_params=train_params,
            device="cpu",
            logger=mock_logger
        )

        assert result == (None, None, None)
        mock_logger.warning.assert_called()

    @patch('identifiability_toy_study.common.helpers.ALL_LOGIC_GATES')
    @patch('glob.glob')
    def test_load_model_best_selection(self, mock_glob, mock_all_gates):
        """Test that load_model selects the best model"""
        mock_gate = Mock()
        mock_gate.n_inputs = 2
        mock_all_gates.__getitem__.return_value = mock_gate

        model_params = ModelParams(logic_gates=["AND"])
        train_params = TrainParams(
            learning_rate=0.001, loss_target=0.01, acc_target=0.95,
            batch_size=32, epochs=100, val_frequency=10
        )

        model_id = model_params.get_id()

        # Multiple model files with different metrics
        mock_files = [
            f"/models/model_{model_id}_loss_0.010000_acc_0.950000_20240101_120000.pt",
            f"/models/model_{model_id}_loss_0.005000_acc_0.960000_20240101_130000.pt",  # Best
            f"/models/model_{model_id}_loss_0.008000_acc_0.955000_20240101_140000.pt",
        ]
        mock_glob.return_value = mock_files

        with patch('torch.load') as mock_torch_load, \
             patch('identifiability_toy_study.common.helpers.MLP') as mock_mlp_class:

            mock_checkpoint = {"model_state_dict": {}}
            mock_torch_load.return_value = mock_checkpoint
            mock_model = Mock()
            mock_mlp_class.return_value = mock_model

            result = load_model(
                model_dir="/models",
                model_params=model_params,
                train_params=train_params,
                device="cpu"
            )

            model, avg_loss, val_acc = result

            # Should load the best model (lowest loss)
            assert avg_loss == 0.005
            assert val_acc == 0.96

            # Should load from the correct file
            expected_file = f"/models/model_{model_id}_loss_0.005000_acc_0.960000_20240101_130000.pt"
            mock_torch_load.assert_called_with(expected_file, map_location="cpu")


class TestSaveModel:
    @patch('torch.save')
    @patch('os.makedirs')
    @patch('identifiability_toy_study.common.helpers.datetime')
    def test_save_model_success(self, mock_datetime, mock_makedirs, mock_torch_save):
        """Test successful model saving"""
        # Mock datetime
        mock_now = Mock()
        mock_now.strftime.return_value = "20240101_120000"
        mock_datetime.now.return_value = mock_now

        model_params = ModelParams(logic_gates=["AND"], width=4, depth=2)
        mock_model = Mock()
        mock_model.state_dict.return_value = {"layer1.weight": torch.tensor([[1.0, 2.0]])}

        mock_logger = Mock()

        result = save_model(
            model_dir="/models",
            model_params=model_params,
            model=mock_model,
            avg_loss=0.005,
            val_acc=0.96,
            device="cpu",
            logger=mock_logger
        )

        assert result is not None
        assert "model_" in result
        assert "_loss_0.005000" in result
        assert "_acc_0.960000" in result

        mock_makedirs.assert_called_once_with("/models", exist_ok=True)
        mock_torch_save.assert_called_once()

        # Check the saved checkpoint structure
        saved_checkpoint = mock_torch_save.call_args[0][0]
        assert "model_state_dict" in saved_checkpoint
        assert "model_params" in saved_checkpoint
        assert "avg_loss" in saved_checkpoint
        assert "val_acc" in saved_checkpoint

    @patch('torch.save')
    def test_save_model_failure(self, mock_torch_save):
        """Test model saving failure"""
        mock_torch_save.side_effect = Exception("Disk full")

        model_params = ModelParams(logic_gates=["AND"])
        mock_model = Mock()
        mock_logger = Mock()

        result = save_model(
            model_dir="/models",
            model_params=model_params,
            model=mock_model,
            avg_loss=0.005,
            val_acc=0.96,
            device="cpu",
            logger=mock_logger
        )

        assert result is None
        mock_logger.error.assert_called()


class TestIntegration:
    def test_complete_training_workflow(self):
        """Integration test for complete training workflow"""
        # Create parameters
        model_params = ModelParams(logic_gates=["AND"], width=3, depth=1)
        train_params = TrainParams(
            learning_rate=0.1,
            loss_target=0.5,
            acc_target=0.8,
            batch_size=4,
            epochs=10,
            val_frequency=5
        )
        data_params = DataParams(
            n_samples_train=16,
            n_samples_val=8,
            n_samples_test=8
        )

        # Generate trial data
        with patch('identifiability_toy_study.common.helpers.ALL_LOGIC_GATES') as mock_gates:
            mock_gate = Mock()
            mock_gate.generate_noisy_data.return_value = (
                torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float32),
                torch.tensor([[0.0], [0.0], [0.0], [1.0]], dtype=torch.float32)  # AND gate
            )
            mock_gates.__getitem__.return_value = mock_gate

            trial_data = generate_trial_data(
                data_params=data_params,
                logic_gates=["AND"],
                device="cpu"
            )

        # Create trial result for status tracking
        trial_setup = TrialSetup(
            seed=42,
            model_params=model_params,
            train_params=train_params,
            data_params=data_params,
            iden_constraints=IdentifiabilityConstraints()
        )
        trial_result = TrialResult(setup=trial_setup)
        mock_logger = Mock()
        status_fx = update_status_fx(trial_result, logger=mock_logger)

        # Train model
        with patch('identifiability_toy_study.common.helpers.MLP') as mock_mlp_class:
            mock_model = Mock()
            mock_model.do_train.return_value = 0.1  # Good loss
            mock_model.do_eval.return_value = 0.9  # Good accuracy
            mock_mlp_class.return_value = mock_model

            model, avg_loss, val_acc = train_model(
                train_params=train_params,
                model_params=model_params,
                data=trial_data,
                device="cpu",
                status_fx=status_fx
            )

        # Verify results
        assert model == mock_model
        assert avg_loss == 0.1
        assert val_acc == 0.9
        assert trial_result.status == "FINISHED_TRAINING"

    def test_save_and_load_workflow(self):
        """Integration test for save and load workflow"""
        model_params = ModelParams(logic_gates=["AND"], width=3, depth=1)
        train_params = TrainParams(
            learning_rate=0.001, loss_target=0.01, acc_target=0.95,
            batch_size=32, epochs=100, val_frequency=10
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            # Save a model
            mock_model = Mock()
            mock_model.state_dict.return_value = {"test": torch.tensor([1.0])}

            with patch('identifiability_toy_study.common.helpers.datetime') as mock_datetime:
                mock_now = Mock()
                mock_now.strftime.return_value = "20240101_120000"
                mock_datetime.now.return_value = mock_now

                file_path = save_model(
                    model_dir=temp_dir,
                    model_params=model_params,
                    model=mock_model,
                    avg_loss=0.005,
                    val_acc=0.96,
                    device="cpu"
                )

            assert file_path is not None
            assert os.path.exists(file_path)

            # Load the model back
            with patch('identifiability_toy_study.common.helpers.ALL_LOGIC_GATES') as mock_gates, \
                 patch('identifiability_toy_study.common.helpers.MLP') as mock_mlp_class:

                mock_gate = Mock()
                mock_gate.n_inputs = 2
                mock_gates.__getitem__.return_value = mock_gate

                mock_loaded_model = Mock()
                mock_mlp_class.return_value = mock_loaded_model

                result = load_model(
                    model_dir=temp_dir,
                    model_params=model_params,
                    train_params=train_params,
                    device="cpu"
                )

                loaded_model, loaded_loss, loaded_acc = result

                assert loaded_model == mock_loaded_model
                assert loaded_loss == 0.005
                assert loaded_acc == 0.96