import json
import pytest
import torch
from dataclasses import dataclass, asdict

from identifiability_toy_study.common.schemas import (
    DataClass,
    DataParams,
    ModelParams,
    TrainParams,
    IdentifiabilityConstraints,
    TrialSetup,
    CircuitMetrics,
    GateMetrics,
    Metrics,
    ProfilingData,
    TrialResult,
    Dataset,
    TrialData,
)


class TestDataClass:
    def test_dataclass_get_id(self):
        """Test DataClass get_id method"""
        @dataclass
        class TestClass(DataClass):
            value: int

        obj = TestClass(value=42)
        id_str = obj.get_id()

        assert isinstance(id_str, str)
        assert len(id_str) > 0

        # Same object should produce same ID
        id_str2 = obj.get_id()
        assert id_str == id_str2

    def test_dataclass_get_id_different_objects(self):
        """Test that different objects produce different IDs"""
        @dataclass
        class TestClass(DataClass):
            value: int

        obj1 = TestClass(value=42)
        obj2 = TestClass(value=43)

        assert obj1.get_id() != obj2.get_id()

    def test_dataclass_str(self):
        """Test DataClass string representation"""
        @dataclass
        class TestClass(DataClass):
            value: int
            name: str

        obj = TestClass(value=42, name="test")
        str_repr = str(obj)

        # Should be valid JSON
        parsed = json.loads(str_repr)
        assert parsed["value"] == 42
        assert parsed["name"] == "test"


class TestDataParams:
    def test_data_params_initialization(self):
        """Test DataParams initialization"""
        params = DataParams(
            n_samples_train=1000,
            n_samples_val=200,
            n_samples_test=300,
            noise_std=0.1,
            skewed_distribution=True
        )

        assert params.n_samples_train == 1000
        assert params.n_samples_val == 200
        assert params.n_samples_test == 300
        assert params.noise_std == 0.1
        assert params.skewed_distribution is True

    def test_data_params_defaults(self):
        """Test DataParams default values"""
        params = DataParams(
            n_samples_train=1000,
            n_samples_val=200,
            n_samples_test=300
        )

        assert params.noise_std == 0.0
        assert params.skewed_distribution is False

    def test_data_params_inheritance(self):
        """Test that DataParams inherits from DataClass"""
        params = DataParams(
            n_samples_train=1000,
            n_samples_val=200,
            n_samples_test=300
        )

        assert isinstance(params, DataClass)
        assert hasattr(params, 'get_id')
        assert hasattr(params, '__str__')


class TestModelParams:
    def test_model_params_initialization(self):
        """Test ModelParams initialization"""
        params = ModelParams(
            logic_gates=["AND", "OR"],
            width=5,
            depth=3
        )

        assert params.logic_gates == ["AND", "OR"]
        assert params.width == 5
        assert params.depth == 3

    def test_model_params_defaults(self):
        """Test ModelParams default values"""
        params = ModelParams()

        assert params.logic_gates == []
        assert params.width == 3
        assert params.depth == 2

    def test_model_params_inheritance(self):
        """Test that ModelParams inherits from DataClass"""
        params = ModelParams()

        assert isinstance(params, DataClass)
        id_str = params.get_id()
        assert isinstance(id_str, str)


class TestTrainParams:
    def test_train_params_initialization(self):
        """Test TrainParams initialization"""
        params = TrainParams(
            learning_rate=0.001,
            loss_target=0.01,
            acc_target=0.95,
            batch_size=64,
            epochs=100,
            val_frequency=10
        )

        assert params.learning_rate == 0.001
        assert params.loss_target == 0.01
        assert params.acc_target == 0.95
        assert params.batch_size == 64
        assert params.epochs == 100
        assert params.val_frequency == 10

    def test_train_params_inheritance(self):
        """Test that TrainParams inherits from DataClass"""
        params = TrainParams(
            learning_rate=0.001,
            loss_target=0.01,
            acc_target=0.95,
            batch_size=64,
            epochs=100,
            val_frequency=10
        )

        assert isinstance(params, DataClass)


class TestIdentifiabilityConstraints:
    def test_identifiability_constraints_initialization(self):
        """Test IdentifiabilityConstraints initialization"""
        constraints = IdentifiabilityConstraints(
            min_sparsity=0.1,
            acc_threshold=0.99,
            is_perfect_circuit=True,
            is_causal_abstraction=False,
            non_transport_stable=True,
            param_decomp=True,
            require_commutation=True,
            commutation_atol=1e-6,
            faithfulness_min=0.8,
            iia_min=0.7,
            iia_num_pairs=256,
            iia_max_vars_per_layer=10
        )

        assert constraints.min_sparsity == 0.1
        assert constraints.acc_threshold == 0.99
        assert constraints.is_perfect_circuit is True
        assert constraints.is_causal_abstraction is False
        assert constraints.non_transport_stable is True
        assert constraints.param_decomp is True
        assert constraints.require_commutation is True
        assert constraints.commutation_atol == 1e-6
        assert constraints.faithfulness_min == 0.8
        assert constraints.iia_min == 0.7
        assert constraints.iia_num_pairs == 256
        assert constraints.iia_max_vars_per_layer == 10

    def test_identifiability_constraints_defaults(self):
        """Test IdentifiabilityConstraints default values"""
        constraints = IdentifiabilityConstraints()

        assert constraints.min_sparsity == 0.0
        assert constraints.acc_threshold == 0.99
        assert constraints.is_perfect_circuit is True
        assert constraints.is_causal_abstraction is False
        assert constraints.non_transport_stable is False
        assert constraints.param_decomp is False
        assert constraints.require_commutation is False
        assert constraints.commutation_atol == 0.0
        assert constraints.faithfulness_min == 0.0
        assert constraints.iia_min == 0.0
        assert constraints.iia_num_pairs == 128
        assert constraints.iia_max_vars_per_layer is None

    def test_identifiability_constraints_inheritance(self):
        """Test that IdentifiabilityConstraints inherits from DataClass"""
        constraints = IdentifiabilityConstraints()

        assert isinstance(constraints, DataClass)


class TestTrialSetup:
    def test_trial_setup_initialization(self):
        """Test TrialSetup initialization"""
        model_params = ModelParams(logic_gates=["AND"], width=3, depth=2)
        train_params = TrainParams(
            learning_rate=0.001, loss_target=0.01, acc_target=0.95,
            batch_size=32, epochs=50, val_frequency=5
        )
        data_params = DataParams(
            n_samples_train=1000, n_samples_val=200, n_samples_test=300
        )
        iden_constraints = IdentifiabilityConstraints(min_sparsity=0.1)

        setup = TrialSetup(
            seed=42,
            model_params=model_params,
            train_params=train_params,
            data_params=data_params,
            iden_constraints=iden_constraints
        )

        assert setup.seed == 42
        assert setup.model_params == model_params
        assert setup.train_params == train_params
        assert setup.data_params == data_params
        assert setup.iden_constraints == iden_constraints

    def test_trial_setup_str(self):
        """Test TrialSetup string representation includes trial_id"""
        model_params = ModelParams()
        train_params = TrainParams(
            learning_rate=0.001, loss_target=0.01, acc_target=0.95,
            batch_size=32, epochs=50, val_frequency=5
        )
        data_params = DataParams(
            n_samples_train=1000, n_samples_val=200, n_samples_test=300
        )
        iden_constraints = IdentifiabilityConstraints()

        setup = TrialSetup(
            seed=42,
            model_params=model_params,
            train_params=train_params,
            data_params=data_params,
            iden_constraints=iden_constraints
        )

        str_repr = str(setup)
        parsed = json.loads(str_repr)

        assert "trial_id" in parsed
        assert parsed["seed"] == 42

    def test_trial_setup_inheritance(self):
        """Test that TrialSetup inherits from DataClass"""
        model_params = ModelParams()
        train_params = TrainParams(
            learning_rate=0.001, loss_target=0.01, acc_target=0.95,
            batch_size=32, epochs=50, val_frequency=5
        )
        data_params = DataParams(
            n_samples_train=1000, n_samples_val=200, n_samples_test=300
        )
        iden_constraints = IdentifiabilityConstraints()

        setup = TrialSetup(
            seed=42,
            model_params=model_params,
            train_params=train_params,
            data_params=data_params,
            iden_constraints=iden_constraints
        )

        assert isinstance(setup, DataClass)


class TestCircuitMetrics:
    def test_circuit_metrics_initialization(self):
        """Test CircuitMetrics initialization"""
        metrics = CircuitMetrics(
            circuit_idx=0,
            accuracy=0.95,
            logit_similarity=0.98,
            bit_similarity=0.92,
            sparsity=(0.3, 0.4, 0.35),
            commutes=True,
            comm_gap=0.01,
            faithfulness=0.88,
            iia=0.75
        )

        assert metrics.circuit_idx == 0
        assert metrics.accuracy == 0.95
        assert metrics.logit_similarity == 0.98
        assert metrics.bit_similarity == 0.92
        assert metrics.sparsity == (0.3, 0.4, 0.35)
        assert metrics.commutes is True
        assert metrics.comm_gap == 0.01
        assert metrics.faithfulness == 0.88
        assert metrics.iia == 0.75

    def test_circuit_metrics_optional_fields(self):
        """Test CircuitMetrics with optional fields"""
        metrics = CircuitMetrics(
            circuit_idx=0,
            accuracy=0.95,
            logit_similarity=0.98,
            bit_similarity=0.92,
            sparsity=(0.3, 0.4, 0.35)
        )

        assert metrics.commutes is None
        assert metrics.comm_gap is None
        assert metrics.faithfulness is None
        assert metrics.iia is None


class TestGateMetrics:
    def test_gate_metrics_initialization(self):
        """Test GateMetrics initialization"""
        circuit_metrics = {
            0: CircuitMetrics(0, 0.95, 0.98, 0.92, (0.3, 0.4, 0.35)),
            1: CircuitMetrics(1, 0.93, 0.96, 0.90, (0.4, 0.5, 0.45))
        }

        metrics = GateMetrics(
            num_total_circuits=10,
            test_acc=0.94,
            per_circuit=circuit_metrics,
            faithful_circuits_idx=[0, 1, 5]
        )

        assert metrics.num_total_circuits == 10
        assert metrics.test_acc == 0.94
        assert len(metrics.per_circuit) == 2
        assert metrics.faithful_circuits_idx == [0, 1, 5]

    def test_gate_metrics_defaults(self):
        """Test GateMetrics default values"""
        metrics = GateMetrics(
            num_total_circuits=5,
            test_acc=0.90
        )

        assert metrics.per_circuit == {}
        assert metrics.faithful_circuits_idx == []


class TestMetrics:
    def test_metrics_initialization(self):
        """Test Metrics initialization"""
        gate_metrics = {
            "AND": GateMetrics(5, 0.95),
            "OR": GateMetrics(3, 0.92)
        }

        metrics = Metrics(
            avg_loss=0.05,
            val_acc=0.94,
            test_acc=0.93,
            per_gate=gate_metrics
        )

        assert metrics.avg_loss == 0.05
        assert metrics.val_acc == 0.94
        assert metrics.test_acc == 0.93
        assert len(metrics.per_gate) == 2

    def test_metrics_defaults(self):
        """Test Metrics default values"""
        metrics = Metrics()

        assert metrics.avg_loss is None
        assert metrics.val_acc is None
        assert metrics.test_acc is None
        assert metrics.per_gate == {}


class TestProfilingData:
    def test_profiling_data_initialization(self):
        """Test ProfilingData initialization"""
        profiling = ProfilingData(
            device="cuda",
            train_secs=120,
            circuit_secs=300,
            iden_secs=180
        )

        assert profiling.device == "cuda"
        assert profiling.train_secs == 120
        assert profiling.circuit_secs == 300
        assert profiling.iden_secs == 180


class TestTrialResult:
    def test_trial_result_initialization(self):
        """Test TrialResult initialization"""
        setup = TrialSetup(
            seed=42,
            model_params=ModelParams(),
            train_params=TrainParams(
                learning_rate=0.001, loss_target=0.01, acc_target=0.95,
                batch_size=32, epochs=50, val_frequency=5
            ),
            data_params=DataParams(
                n_samples_train=1000, n_samples_val=200, n_samples_test=300
            ),
            iden_constraints=IdentifiabilityConstraints()
        )

        metrics = Metrics(avg_loss=0.05, val_acc=0.94)
        profiling = ProfilingData("cpu", 60, 120, 90)

        result = TrialResult(
            setup=setup,
            status="COMPLETED",
            metrics=metrics,
            profiling=profiling
        )

        assert result.setup == setup
        assert result.status == "COMPLETED"
        assert result.metrics == metrics
        assert result.profiling == profiling

    def test_trial_result_defaults(self):
        """Test TrialResult default values"""
        setup = TrialSetup(
            seed=42,
            model_params=ModelParams(),
            train_params=TrainParams(
                learning_rate=0.001, loss_target=0.01, acc_target=0.95,
                batch_size=32, epochs=50, val_frequency=5
            ),
            data_params=DataParams(
                n_samples_train=1000, n_samples_val=200, n_samples_test=300
            ),
            iden_constraints=IdentifiabilityConstraints()
        )

        result = TrialResult(setup=setup)

        assert result.status == "UNKNOWN"
        assert isinstance(result.metrics, Metrics)
        assert result.profiling is None

    def test_trial_result_post_init(self):
        """Test TrialResult __post_init__ sets trial_id"""
        setup = TrialSetup(
            seed=42,
            model_params=ModelParams(),
            train_params=TrainParams(
                learning_rate=0.001, loss_target=0.01, acc_target=0.95,
                batch_size=32, epochs=50, val_frequency=5
            ),
            data_params=DataParams(
                n_samples_train=1000, n_samples_val=200, n_samples_test=300
            ),
            iden_constraints=IdentifiabilityConstraints()
        )

        result = TrialResult(setup=setup)

        assert result.trial_id == setup.get_id()


class TestDataset:
    def test_dataset_initialization(self):
        """Test Dataset initialization"""
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        y = torch.tensor([[0.0], [1.0]])

        dataset = Dataset(x=x, y=y)

        assert torch.equal(dataset.x, x)
        assert torch.equal(dataset.y, y)


class TestTrialData:
    def test_trial_data_initialization(self):
        """Test TrialData initialization"""
        train_x = torch.tensor([[1.0, 2.0]])
        train_y = torch.tensor([[0.0]])
        train_dataset = Dataset(x=train_x, y=train_y)

        val_x = torch.tensor([[3.0, 4.0]])
        val_y = torch.tensor([[1.0]])
        val_dataset = Dataset(x=val_x, y=val_y)

        test_x = torch.tensor([[5.0, 6.0]])
        test_y = torch.tensor([[0.0]])
        test_dataset = Dataset(x=test_x, y=test_y)

        trial_data = TrialData(
            train=train_dataset,
            val=val_dataset,
            test=test_dataset
        )

        assert trial_data.train == train_dataset
        assert trial_data.val == val_dataset
        assert trial_data.test == test_dataset


class TestIntegration:
    def test_complete_trial_setup_flow(self):
        """Integration test for complete trial setup flow"""
        # Create all components
        model_params = ModelParams(logic_gates=["AND", "OR"], width=4, depth=2)
        train_params = TrainParams(
            learning_rate=0.001, loss_target=0.01, acc_target=0.95,
            batch_size=32, epochs=50, val_frequency=5
        )
        data_params = DataParams(
            n_samples_train=1000, n_samples_val=200, n_samples_test=300,
            noise_std=0.1, skewed_distribution=True
        )
        iden_constraints = IdentifiabilityConstraints(
            min_sparsity=0.2, acc_threshold=0.98, require_commutation=True
        )

        # Create setup
        setup = TrialSetup(
            seed=42,
            model_params=model_params,
            train_params=train_params,
            data_params=data_params,
            iden_constraints=iden_constraints
        )

        # Create metrics
        circuit_metrics = CircuitMetrics(
            circuit_idx=0, accuracy=0.96, logit_similarity=0.99,
            bit_similarity=0.94, sparsity=(0.3, 0.4, 0.35),
            commutes=True, faithfulness=0.92, iia=0.88
        )
        gate_metrics = GateMetrics(
            num_total_circuits=5, test_acc=0.95,
            per_circuit={0: circuit_metrics}, faithful_circuits_idx=[0]
        )
        metrics = Metrics(
            avg_loss=0.02, val_acc=0.96, test_acc=0.95,
            per_gate={"AND": gate_metrics}
        )

        # Create profiling data
        profiling = ProfilingData("cuda", 180, 240, 120)

        # Create result
        result = TrialResult(
            setup=setup, status="COMPLETED",
            metrics=metrics, profiling=profiling
        )

        # Verify everything is connected properly
        assert result.trial_id == setup.get_id()
        assert result.setup.model_params.logic_gates == ["AND", "OR"]
        assert result.metrics.per_gate["AND"].per_circuit[0].commutes is True

        # Test serialization
        str_repr = str(setup)
        parsed = json.loads(str_repr)
        assert "trial_id" in parsed
        assert parsed["seed"] == 42

    def test_dataclass_id_consistency(self):
        """Test that DataClass IDs are consistent across different instances"""
        # Create identical setups
        setup1 = TrialSetup(
            seed=42,
            model_params=ModelParams(logic_gates=["AND"], width=3),
            train_params=TrainParams(
                learning_rate=0.001, loss_target=0.01, acc_target=0.95,
                batch_size=32, epochs=50, val_frequency=5
            ),
            data_params=DataParams(
                n_samples_train=1000, n_samples_val=200, n_samples_test=300
            ),
            iden_constraints=IdentifiabilityConstraints()
        )

        setup2 = TrialSetup(
            seed=42,
            model_params=ModelParams(logic_gates=["AND"], width=3),
            train_params=TrainParams(
                learning_rate=0.001, loss_target=0.01, acc_target=0.95,
                batch_size=32, epochs=50, val_frequency=5
            ),
            data_params=DataParams(
                n_samples_train=1000, n_samples_val=200, n_samples_test=300
            ),
            iden_constraints=IdentifiabilityConstraints()
        )

        # Should have same ID
        assert setup1.get_id() == setup2.get_id()

        # Create slightly different setup
        setup3 = TrialSetup(
            seed=43,  # Different seed
            model_params=ModelParams(logic_gates=["AND"], width=3),
            train_params=TrainParams(
                learning_rate=0.001, loss_target=0.01, acc_target=0.95,
                batch_size=32, epochs=50, val_frequency=5
            ),
            data_params=DataParams(
                n_samples_train=1000, n_samples_val=200, n_samples_test=300
            ),
            iden_constraints=IdentifiabilityConstraints()
        )

        # Should have different ID
        assert setup1.get_id() != setup3.get_id()

    def test_trial_data_with_datasets(self):
        """Test creating TrialData with actual Dataset objects"""
        # Create datasets with realistic data
        train_x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        train_y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])  # XOR pattern
        train_dataset = Dataset(x=train_x, y=train_y)

        val_x = torch.tensor([[0.5, 0.5], [1.5, 0.5]])
        val_y = torch.tensor([[0.5], [1.0]])
        val_dataset = Dataset(x=val_x, y=val_y)

        test_x = torch.tensor([[0.1, 0.1], [0.9, 0.9]])
        test_y = torch.tensor([[0.0], [0.0]])
        test_dataset = Dataset(x=test_x, y=test_y)

        trial_data = TrialData(
            train=train_dataset,
            val=val_dataset,
            test=test_dataset
        )

        # Verify structure
        assert trial_data.train.x.shape == (4, 2)
        assert trial_data.train.y.shape == (4, 1)
        assert trial_data.val.x.shape == (2, 2)
        assert trial_data.test.x.shape == (2, 2)

        # Verify data integrity
        assert torch.equal(trial_data.train.x[0], torch.tensor([0.0, 0.0]))
        assert torch.equal(trial_data.train.y[0], torch.tensor([0.0]))