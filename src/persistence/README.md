# Persistence Module

This module handles saving and loading experiment results. All changes to save/load logic
should be documented here.

## Folder Structure

```
runs/
  run_{timestamp}/
    config.json                       # Experiment configuration
    log.txt                           # Training logs
    {trial_id}/                       # One folder per trial
      setup.json                      # Trial setup parameters
      metrics.json                    # Training metrics, robustness, faithfulness
      circuits.json                   # Subcircuit masks and structure analysis
      tensors.pt                      # All tensor data (test, activations, weights)
      all_gates/
        model.pt                      # Trained MLP weights
        decomposed_model.pt           # SPD decomposition
        circuit_activations.png       # Visualization
      {gate_name}/                    # Per-gate (e.g., XOR/)
        full/
          decomposed_model.pt
          circuit_activations.png
          robustness_summary.png
        {subcircuit_idx}/             # Best subcircuits (e.g., 42/)
          circuit.png
          circuit_activations.png
          decomposed_model.pt
          robustness_summary.png
          robustness_detail/
```

## File Formats

### config.json
Experiment-level configuration:
```json
{
  "device": "cpu",
  "widths": [3],
  "depths": [2],
  "target_logic_gates": ["XOR"],
  ...
}
```

### setup.json
Trial-specific parameters:
```json
{
  "seed": 0,
  "data_params": {"n_samples_train": 2048, ...},
  "model_params": {"logic_gates": ["XOR"], "width": 3, "depth": 2},
  "train_params": {"learning_rate": 0.001, ...},
  "constraints": {"epsilon": 0.1},
  "spd_config": {...}
}
```

### metrics.json
Training and analysis results:
```json
{
  "status": "SUCCESSFUL_TRIAL",
  "trial_id": "abc123",
  "avg_loss": 0.001,
  "test_acc": 1.0,
  "per_gate_metrics": {
    "XOR": {
      "test_acc": 1.0,
      "subcircuit_metrics": [...]
    }
  },
  "per_gate_bests": {"XOR": [0, 5, 12]},
  "per_gate_bests_robust": {...},
  "per_gate_bests_faith": {...}
}
```

### circuits.json
Subcircuit data:
```json
{
  "subcircuits": [
    {"idx": 0, "node_masks": [[1,1], [1,0,1], [1]], "edge_masks": [...]},
    ...
  ],
  "subcircuit_structure_analysis": [
    {
      "idx": 0,
      "node_sparsity": 0.5,
      "in_patches": [{"layers": [1], "indices": [0], "axis": "neuron"}],
      ...
    }
  ],
  "decomposed_subcircuit_indices": {"XOR": [0, 5]}
}
```

### tensors.pt
PyTorch file containing:
- `test_x`: Test input data `[N, input_dim]`
- `test_y`: Ground truth outputs `[N, output_dim]`
- `test_y_pred`: Model predictions `[N, output_dim]`
- `activations`: List of per-layer activations `[N, layer_width]`
- `canonical_activations`: Dict for visualization
  - `"0_0"`, `"0_1"`, `"1_0"`, `"1_1"`: Activations for each binary input
- `layer_weights`: Weight matrices per layer

### model.pt
MLP configuration and weights:
```
{
  "hidden_sizes": [3, 3],
  "input_size": 2,
  "output_size": 1,
  "activation": "leaky_relu",
  "state_dict": {...}
}
```

## Usage

### Saving
```python
from src.persistence import save_results

result = run_experiment(cfg, logger)
save_results(result, run_dir)
```

### Loading
```python
from src.persistence import (
    load_config,
    load_trial_setup,
    load_trial_metrics,
    load_trial_circuits,
    load_tensors,
    load_experiment,
)

# Load individual files
config = load_config(run_dir)
setup = load_trial_setup(trial_dir)
metrics = load_trial_metrics(trial_dir)
circuits = load_trial_circuits(trial_dir)
tensors = load_tensors(trial_dir)

# Or load everything at once
experiment = load_experiment(run_dir)
```

## Design Decisions

1. **Multiple JSON files**: Split by concern (config vs metrics vs circuits) for
   easier inspection and smaller file sizes.

2. **Tensors in .pt**: Large numerical data stored efficiently in PyTorch format,
   not serialized to JSON.

3. **Parsable data**: All JSON data uses proper nested dicts/lists, not string
   representations. PatchShape serializes as `{"layers": [1], "indices": [0], "axis": "neuron"}`.

4. **Hierarchical structure**: trial -> gate -> subcircuit organization makes
   it easy to navigate results.
