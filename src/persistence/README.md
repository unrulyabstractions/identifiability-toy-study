# Persistence Module

This module handles saving and loading experiment results. All changes to save/load logic
should be documented here.

## Folder Structure

```
runs/
  run_{timestamp}/
    summary.json                      # Ranked results across all trials and gates
    explanation.md                    # How to read this folder
    config.json                       # ExperimentConfig
    trial_org.json                    # Maps sweep parameters to trial IDs
    gates.json                        # Gate name -> base gate function mapping
    circuits.json                     # Run-level subcircuit masks and structures
    training_data/
      train.pt                        # Training data tensors (x, y)
      val.pt                          # Validation data tensors
      test.pt                         # Test data tensors
      metadata.json                   # Data shapes and gate names
    profiling/
      profiling.json                  # Timing data (events, phase durations)
    subcircuits/
      summary.json                    # Overview of subcircuit rankings
      explanation.md                  # How to read this folder
      subcircuit_score_ranking.json   # Rankings by gate with faithfulness scores
      mask_idx_map.json               # (node_mask_idx, edge_mask_idx) mapping
      circuit_diagrams/
        node_masks/{idx}.png          # Node pattern diagrams
        edge_masks/{idx}.png          # Edge pattern diagrams
        ranked_node_masks/            # Diagrams sorted by ranking
        ranked_edge_masks/
        structural_faithfulness/
          summary.json                # Structural analysis overview
          samples.json                # Per-subcircuit detailed metrics
          structural_rankings.json    # Rankings by structural metrics
          explanation.md              # How to read this folder
    trials/
      {trial_id}/
        summary.json                  # Trial-level ranked results
        explanation.md                # How to read this folder
        config.json                   # Full effective config
        setup.json                    # TrialSetup parameters
        metrics.json                  # Metrics (training, per-gate, faithfulness)
        tensors.pt                    # Test data, activations, weights
        all_gates/
          model.pt                    # Trained MLP weights
        gates/
          {gate_name}/
            summary.json              # Gate-level metrics summary
            full/
              decomposed_model.pt
            {subcircuit_idx}/
              decomposed_model.pt
```

## Key Concepts

### Index Types

- **node_mask_idx**: Sequential index identifying unique node connectivity patterns (0, 1, 2...)
- **edge_mask_idx**: Sequential index identifying unique edge connectivity patterns (0, 1, 2...)
- **edge_variant_rank**: Optimization rank (0=best, 1=2nd best...) - used during optimization only
- **subcircuit_idx**: Flat index encoding (node_mask_idx, edge_mask_idx) pair

### Architecture

Architecture is defined in `base_trial.model_params`:
- `width`: Hidden layer width (all layers have same width)
- `depth`: Number of hidden layers

## File Formats

### config.json
Experiment-level configuration:
```json
{
  "device": "cpu",
  "base_trial": {
    "model_params": {"width": 3, "depth": 2, "logic_gates": ["XOR"]},
    "train_params": {"learning_rate": 0.001},
    "constraints": {"epsilon": 0.1}
  },
  "activations": ["leaky_relu"],
  "learning_rates": [0.001]
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
  "per_gate_bests_faith": {...}
}
```

### mask_idx_map.json
Mapping between index types:
```json
{
  "n_unique_edge_patterns": 1,
  "n_unique_node_patterns": 48,
  "subcircuits": [
    {
      "subcircuit_idx": 0,
      "node_mask_idx": 0,
      "edge_mask_idx": 0,
      "node_sparsity": 0.67,
      "edge_sparsity": 1.0
    }
  ]
}
```

### tensors.pt
PyTorch file containing:
- `test_x`: Test input data `[N, input_dim]`
- `test_y`: Ground truth outputs `[N, output_dim]`
- `test_y_pred`: Model predictions `[N, output_dim]`
- `activations`: List of per-layer activations `[N, layer_width]`
- `canonical_activations`: Dict for visualization
- `layer_weights`: Weight matrices per layer

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
    load_experiment,
)

# Load individual files
config = load_config(run_dir)
setup = load_trial_setup(trial_dir)
metrics = load_trial_metrics(trial_dir)

# Or load everything at once
experiment = load_experiment(run_dir)
```

## Design Decisions

1. **Explanation files**: Every folder with data has an `explanation.md` describing its contents.

2. **Multiple JSON files**: Split by concern (config vs metrics vs circuits) for
   easier inspection and smaller file sizes.

3. **Tensors in .pt**: Large numerical data stored efficiently in PyTorch format,
   not serialized to JSON.

4. **Parsable data**: All JSON data uses proper nested dicts/lists, not string
   representations.

5. **Hierarchical structure**: run -> subcircuits/trials -> gates -> subcircuits
   organization makes it easy to navigate results.

6. **Schema classes**: All data structures use dataclasses extending `SchemaClass`
   for consistent serialization and validation.

7. **Validation**: At the end of saving, all required folders are validated to have
   `explanation.md` files. Missing explanations cause an assertion error.
