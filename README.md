# Neighbor-Aware Reinforcement Learning Fosters Cooperation in Spatial Public Goods Games

This repository contains the implementation for "Neighbor-Aware Reinforcement Learning Fosters Cooperation in Spatial Public Goods Games", a study exploring how reinforcement learning agents can learn cooperative strategies in spatial public goods games through neighbor-aware mechanisms.

## Overview

This project implements a spatial public goods game (SPGG) where agents use reinforcement learning to learn optimal strategies. The key innovation is the incorporation of neighbor influence and reputation mechanisms, allowing agents to consider their neighbors' behavior when making decisions. The framework supports two state representation methods:

- **Reputation-based states**: Agents observe the average reputation of their neighbors
- **Action-based states**: Agents observe the previous actions/strategies of their neighbors

## Project Structure

```
.
├── config/                    # Configuration files
│   └── default_config.yaml   # Default experiment configuration
├── src/                       # Source code
│   ├── model/                # SPGG model implementation
│   │   ├── __init__.py
│   │   └── spgg.py           # Main SPGG class
│   ├── experiments/          # Experiment running logic
│   │   ├── __init__.py
│   │   └── runner.py         # Experiment runner functions
│   ├── visualization/        # Plotting and visualization
│   │   ├── __init__.py
│   │   └── plotting.py       # Figure generation functions
│   └── config_loader.py      # Configuration loading utilities
├── scripts/                   # Executable scripts
│   ├── run_experiments.py    # Main experiment runner
│   └── plot_figures.py       # Figure generation script
├── data/                      # Data directory (created at runtime)
├── results/                   # Results directory (created at runtime)
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running Experiments

Run experiments using the default configuration:

```bash
python scripts/run_experiments.py
```

Run specific experiment types:

```bash
# Run experiments for Figures 2, 3, 4
python scripts/run_experiments.py --experiment-type figure_2_3_4

# Run experiments for Figures 6, 7, 8, 9
python scripts/run_experiments.py --experiment-type figure_6_7_8_9

# Run custom experiments
python scripts/run_experiments.py --experiment-type custom
```

Use a custom configuration file:

```bash
python scripts/run_experiments.py --config path/to/config.yaml
```

Control number of processes:

```bash
python scripts/run_experiments.py --num-processes 8
```

### Generating Figures

Generate all figures:

```bash
python scripts/plot_figures.py
```

Generate specific figures:

```bash
python scripts/plot_figures.py --figures 2 4 6
```

Specify data and output directories:

```bash
python scripts/plot_figures.py --data-dir ./results --output-dir ./figures
```

### Configuration

Edit `config/default_config.yaml` to customize experiment parameters:

- **Model parameters**: Lattice size, iterations, game parameters
- **RL parameters**: Learning rate, discount factor, exploration
- **Neighbor influence**: Influence factor (kappa), neighbor order (M)
- **Reputation**: Reputation update rules
- **Reward structure**: Payoff and reputation weights

### Programmatic Usage

```python
from src.model import SPGG
from src.config_loader import load_config, get_model_params

# Load configuration
config = load_config()

# Get model parameters
params = get_model_params(config, r=3.0, influence_factor=1.0)

# Create and run model
spgg = SPGG(**params)
spgg.folder = "my_results"
spgg.run("my_results/data.h5")
```

## Key Features

- **Modular Design**: Separated model, experiments, and visualization
- **Configuration Management**: YAML-based configuration system
- **Flexible State Representation**: Support for both reputation-based and action-based states
- **Parallel Execution**: Multi-process experiment running
- **Publication-Quality Plots**: Pre-configured matplotlib styles
- **Comprehensive Data Output**: HDF5 format with all tracked metrics

## Parameters

### Model Parameters
- `r`: Multiplication factor for public goods
- `L`: Lattice size (L × L)
- `iterations`: Number of simulation iterations

### Reinforcement Learning
- `alpha`: Learning rate
- `gamma`: Discount factor
- `epsilon`: Exploration rate (with decay)
- `state_representation`: State representation method
  - `'reputation'` (default): State based on average reputation of neighbors
  - `'action'`: State based on previous action/strategy

### Neighbor Influence
- `influence_factor` (κ): Strength of neighbor influence
- `use_second_order`: Use second-order neighbors (M=2) vs first-order (M=1)

### Reputation
- `rep_gain_C` (ΔR_C): Reputation gain for cooperation
- `delta_R_D`: Reputation loss for defection

### Reward Structure
- `reward_weight_payoff` (w_P): Weight for payoff component
- Reputation weight automatically set to (1 - w_P)

## Output

Experiments generate:
- HDF5 data files with all tracked metrics
- Strategy and reputation snapshots
- Q-value histories
- Cooperation rate evolution
- And more...

Figures are saved as PDF files suitable for publication.

## Citation

If you use this code, please cite the original paper:

"Neighbor-Aware Reinforcement Learning Fosters Cooperation in Spatial Public Goods Games"

Available at: https://www.sciencedirect.com/science/article/abs/pii/S0960077925008756
