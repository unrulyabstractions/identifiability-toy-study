# Identifiability Toy Study

This repository integrates and extends code from the following projects:
- [MI-identifiability](https://github.com/MelouxM/MI-identifiability) by Maxime Méloux
- [spd](https://github.com/goodfire-ai/spd) by the Goodfire-AI team (git submodule)

**External libraries** (not tracked, placed in `external/` folder):
- [circuit-stability](https://github.com/alansun17904/circuit-stability) by Alan Sun
- [eap-ig-faithfulness](https://github.com/hannamw/eap-ig-faithfulness) by Hannah W. (EAP-IG)
- [SubspacePartition](https://github.com/huangxt39/SubspacePartition) by Xiaotian Huang et al.

All of the above projects are licensed under MIT (see their LICENSE files).
My modifications and additional contributions are © 2025 Ian Rios-Sialer, released under the MIT License.

## Installation

### Local Development Setup (Recommended)

Using [uv](https://docs.astral.sh/uv/) - a fast Python package manager:

1. **Install uv** (one-time setup):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   export PATH="$HOME/.local/bin:$PATH"
   ```

2. **Clone and setup**:
   ```bash
   git clone --recurse-submodules https://github.com/unrulyabstractions/identifiability-toy-study.git
   cd identifiability-toy-study
   uv sync
   ```

This creates a virtual environment and installs all dependencies automatically.

### Google Colab Setup

For running in Google Colab:

```python
# Install uv in Colab
!curl -LsSf https://astral.sh/uv/install.sh | sh
import os
os.environ['PATH'] = f"/root/.local/bin:{os.environ['PATH']}"

# Clone and install the project
!git clone --recurse-submodules https://github.com/unrulyabstractions/identifiability-toy-study.git
%cd identifiability-toy-study
!uv sync

# For GPU acceleration in Colab, use CUDA
# when running main.py: --device cuda:0
```

### Alternative: Standard pip install

If you prefer traditional pip (slower but familiar):

```bash
pip install git+https://github.com/unrulyabstractions/identifiability-toy-study.git
```

## Usage

### Running Experiments

**Basic usage** (if you used uv installation):
```bash
# Show all available options
uv run python main.py --help

# Quick test run
uv run python main.py --n-experiments 1 --epochs 100 --loss-target 0.1

# Full experiment with multiple logic gates
uv run python main.py \
  --target-logic-gates AND OR XOR \
  --n-experiments 5 \
  --epochs 1000 \
  --size 3 4 5 \
  --device mps
```

**For different hardware:**
```bash
# Apple Silicon Macs (default, fastest)
uv run python main.py --device mps

# NVIDIA GPUs or Google Colab
uv run python main.py --device cuda:0

# CPU only (any system)
uv run python main.py --device cpu
```

**If you used pip install:** Replace `uv run python` with just `python` in the commands above.

### Key Parameters

- `--target-logic-gates`: Choose from AND, OR, XOR, NAND, NOR, IMP, NIMP, MAJORITY, PARITY, FULL_ADDER, EXACT_2
- `--n-experiments`: Number of different random seeds to run
- `--epochs`: Maximum training epochs
- `--size`: Hidden layer sizes to test
- `--depth`: Number of hidden layers to test
- `--device`: Use 'cpu', 'mps' (Apple Silicon), or 'cuda:0' (NVIDIA) for GPU acceleration
- `--loss-target`: Target loss value for convergence
- `--verbose`: Show detailed training progress

Results are saved to `logs/run_TIMESTAMP/` with CSV output containing experimental data.

### Development Commands

For contributors (requires uv installation):

```bash
make run       # Run main experiment
make sanity    # Check device availability and imports
make lint      # Check code style
make format    # Fix code formatting
uv run jupyter lab  # Start Jupyter notebooks
```

### Updating Submodules

To update the spd submodule to the latest version:

```bash
git submodule update --remote src/spd
```
