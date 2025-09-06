# Identifiability Toy Study

This repository integrates and extends code from the following projects:
- [MI-identifiability](https://github.com/MelouxM/MI-identifiability) by Maxime Méloux 
- [spd](https://github.com/goodfire-ai/spd) by the Goodfire-AI team
- [circuit-stability](https://github.com/alansun17904/circuit-stability) by Alan Sun
- [eap-ig-faithfulness](https://github.com/hannamw/eap-ig-faithfulness) by Hannah W. 

All of the above projects are licensed under MIT (see their LICENSE files).  
My modifications and additional contributions are © 2025 Ian Rios-Sialer, released under the MIT License.

## Installation

### Local Development Setup

For local development with modern Python package management:

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   export PATH="$HOME/.local/bin:$PATH"
   ```

2. **Clone and setup the repository**:
   ```bash
   git clone https://github.com/unrulyabstractions/identifiability-toy-study.git
   cd identifiability-toy-study
   uv sync --dev
   ```

This will create a virtual environment at `.venv` and install all dependencies including development tools.

### Google Colab Setup

For running in Google Colab:

```python
# Install uv in Colab
!curl -LsSf https://astral.sh/uv/install.sh | sh
import os
os.environ['PATH'] = f"/root/.local/bin:{os.environ['PATH']}"

# Clone and install the project
!git clone https://github.com/unrulyabstractions/identifiability-toy-study.git
%cd identifiability-toy-study
!uv sync --dev

# For GPU acceleration in Colab, use CUDA
# when running main.py: --device cuda:0
```

### Alternative: Direct pip install

```bash
pip install git+https://github.com/unrulyabstractions/identifiability-toy-study.git
```

## Usage

### Running the Main Experiment

The main experiment script can be run with various parameters:

```bash
# Show all available options
uv run python main.py --help

# Run a quick test experiment
uv run python main.py --n-experiments 1 --epochs 100 --loss-target 0.1

# Run with specific logic gates and parameters
uv run python main.py \
  --target-logic-gates AND OR XOR \
  --n-experiments 5 \
  --epochs 1000 \
  --size 3 4 5 \
  --depth 2 3 \
  --learning-rate 0.001 0.01 \
  --device cpu

# For GPU acceleration
# Apple Silicon Macs (recommended)
uv run python main.py --device mps --epochs 2000

# NVIDIA GPUs (Linux/Windows) or Google Colab
uv run python main.py --device cuda:0 --epochs 2000
```

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

### Development Tools

The project includes several development tools:

```bash
# Run tests
uv run pytest

# Code formatting and linting
uv run ruff check --fix .
uv run black .

# Type checking
uv run mypy identifiability_toy_study/

# Start Jupyter Lab
uv run jupyter lab

# Quick sanity check (device availability, imports)
make sanity
```
