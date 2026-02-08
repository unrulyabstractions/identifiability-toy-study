# Identifiability Toy Study

A research framework for studying mechanistic interpretability and circuit identifiability in neural networks using boolean logic gates as ground-truth circuits.

## Quick Start

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone --recurse-submodules https://github.com/unrulyabstractions/identifiability-toy-study.git
cd identifiability-toy-study
uv sync

# Run a quick experiment
uv run python main.py --n-experiments 1 --epochs 100 --loss-target 0.1
```

## Running Experiments

```bash
# Show all available options
uv run python main.py --help

# Full experiment with multiple logic gates
uv run python main.py \
  --target-logic-gates AND OR XOR \
  --n-experiments 5 \
  --epochs 1000 \
  --size 3 4 5 \
  --device mps
```

### Hardware Options

```bash
--device mps      # Apple Silicon Macs (default, fastest for M-series)
--device cuda:0   # NVIDIA GPUs / Google Colab
--device cpu      # CPU only (any system)
```

## Key Parameters

### Model Configuration
| Parameter | Description | Default |
|-----------|-------------|---------|
| `--target-logic-gates` | Logic gates to learn: AND, OR, XOR, NAND, NOR, IMP, NIMP, MAJORITY, PARITY, FULL_ADDER, EXACT_2 | AND |
| `--size` | Hidden layer sizes to test | 3 4 5 |
| `--depth` | Number of hidden layers | 1 2 |

### Training Configuration
| Parameter | Description | Default |
|-----------|-------------|---------|
| `--n-experiments` | Number of random seeds | 3 |
| `--epochs` | Maximum training epochs | 5000 |
| `--loss-target` | Target loss for convergence | 1e-4 |
| `--learning-rate` | Learning rate | 0.03 |
| `--batch-size` | Training batch size | 32 |

### Analysis Configuration
| Parameter | Description | Default |
|-----------|-------------|---------|
| `--epsilon` | Similarity threshold for identifiability | 0.1 |
| `--n-noise-samples` | Samples for noise robustness | 100 |
| `--n-ood-samples` | Samples for out-of-distribution | 50 |
| `--n-interventions` | Interventions per patch | 5 |

### Output Control
| Parameter | Description |
|-----------|-------------|
| `--no-viz` | Skip generating visualizations |
| `--no-spd-sweep` | Skip SPD parameter sweep |
| `--verbose` | Show detailed training progress |

## Understanding the Metrics

Results are saved to `runs/run_TIMESTAMP/` with detailed metrics for each gate.

### Observational Metrics (Robustness)
Measure how well subcircuits maintain behavior under distribution shift:
- **Noise Robustness**: Performance with added Gaussian noise
- **OOD Extension**: Performance on out-of-distribution inputs (values outside [0,1])

### Interventional Metrics (Faithfulness)
Measure causal faithfulness of identified subcircuits:
- **Mean Intervention Effect**: Average effect of patching clean activations
- **Bit Similarity**: How closely patched outputs match clean outputs

### Counterfactual Metrics
Measure what-if scenarios:
- **Necessity**: Does ablating the circuit break the computation?
- **Sufficiency**: Does the circuit alone produce correct outputs?

### Per-Gate Summary (summary.json)
Each gate folder contains a `summary.json` with:
- `subcircuit_indices`: Ranked list of best subcircuits by overall score
- `observational`: Score based on noise/OOD robustness
- `interventional`: Score based on causal intervention effects
- `counterfactual`: Score based on necessity/sufficiency

## Output Structure

```
runs/run_TIMESTAMP/
  config.json              # Experiment configuration
  trial_001/
    setup.json             # Trial parameters
    metrics.json           # All computed metrics
    circuits.json          # Identified subcircuits
    tensors.pt             # Saved activations/weights
    AND/                   # Per-gate results
      summary.json         # Gate-level metrics summary
      activations_*.png    # Circuit activation visualizations
      ...
    spd/                   # SPD decomposition results
      ...
```

## Attribution

This repository integrates code from:
- [MI-identifiability](https://github.com/MelouxM/MI-identifiability) by Maxime Meloux
- [spd](https://github.com/goodfire-ai/spd) by the Goodfire-AI team (git submodule)

External libraries (in `external/` folder, not tracked):
- [circuit-stability](https://github.com/alansun17904/circuit-stability) by Alan Sun
- [eap-ig-faithfulness](https://github.com/hannamw/eap-ig-faithfulness) by Hannah W.
- [SubspacePartition](https://github.com/huangxt39/SubspacePartition) by Xiaotian Huang et al.

All projects licensed under MIT. Modifications (C) 2025 Ian Rios-Sialer, MIT License.
