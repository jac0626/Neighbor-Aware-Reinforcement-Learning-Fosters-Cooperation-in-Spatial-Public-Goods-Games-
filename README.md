# Spatial Public Goods Games (SPGG) with Reinforcement Learning

This repository contains a Python implementation of a Spatial Public Goods Game (SPGG) simulation where agents learn strategies (Cooperate or Defect) using Q-learning and Neighbor Influence (NI).

## Project Structure

The codebase has been refactored for better modularity and maintainability:

```
.
├── configs/                  # Configuration files
│   └── default_config.yaml
├── src/
│   ├── core/                 # Core simulation logic
│   │   ├── spgg_model.py     # Main SPGG class
│   │   ├── dqn_model.py      # Shared DQN and ReplayBuffer
│   │   └── state_strategies.py # State definitions (Reputation vs Action)
│   ├── utils/                # Utility functions
│   │   ├── math_utils.py     # Math helpers (e.g., overlap5)
│   │   └── data_utils.py     # HDF5 data management
│   └── visualization/        # Plotting tools
│       ├── plot_figures.py   # Plotting functions
│       └── plot_utils.py     # Style definitions
├── run_experiment.py         # Unified entry point
└── requirements.txt
```

## Installation

Ensure you have the required dependencies installed:

```bash
pip install numpy matplotlib scipy h5py pyyaml torch
```

## Usage

### Running a Single Simulation

You can run a simulation using the command line interface. By default, it uses the "reputation" state definition.

```bash
python run_experiment.py --r 3.0 --influence_factor 1.0 --state_type reputation --plot
```

**Arguments:**
-   `--r`: Synergy factor (default: 2.0).
-   `--influence_factor`: Strength of neighbor influence (default: 1.0).
-   `--state_type`: Type of state definition. Options: `reputation` (default) or `action`.
-   `--plot`: If set, generates plots immediately after the simulation.
-   `--plot`: If set, generates plots immediately after the simulation.
-   `--output_dir`: Directory to save results (default: `results`).
-   `--iterations`: Number of iterations (default: 1000).
-   `--dqn_lambda`: Mixing coefficient for DQN (0.0 = pure Q-table, 1.0 = pure DQN).

### Hybrid Dual-Brain Architecture

This project implements a "Hybrid Dual-Brain" architecture where agents make decisions using a combination of:
1.  **Q-Table (Fast System)**: Traditional tabular Q-learning based on discrete states (Reputation/Action).
2.  **Shared DQN (Slow System)**: A deep neural network shared across all agents, taking continuous state features (Self Reputation, Neighbor Avg Reputation, Neighbor Coop Rate, Payoff) as input.

The final Q-value is a weighted mix: $Q_{final} = (1 - \lambda) \cdot Q_{table} + \lambda \cdot Q_{dqn}$

To enable DQN and tune the mixing parameter $\lambda$:

```bash
python run_experiment.py --r 3.0 --dqn_lambda 0.1 0.3 0.5 --parallel
```

### Running a Parameter Sweep

You can specify multiple values for `r` and `influence_factor` to run a batch of experiments. Use `--parallel` to run them concurrently.

```bash
python run_experiment.py --r 3.0 4.0 --influence_factor 0.0 1.0 --parallel --plot
```

### Using a Configuration File

You can define all parameters in a YAML file (see `configs/default_config.yaml` for an example) and load it:

```bash
python run_experiment.py --config configs/default_config.yaml
```

### Running via GitHub Actions

You can run large-scale experiments in the cloud using the provided GitHub Actions workflow:
1.  Go to the **Actions** tab in your repository.
2.  Select **Run SPGG Experiments**.
3.  Click **Run workflow**.
4.  Configure `iterations` and `state_type`.
5.  The workflow automatically sweeps over `r`, `influence_factor`, and `dqn_lambda`.

## Key Components

-   **SPGG Class**: The core simulation engine (`src/core/spgg_model.py`).
-   **StateProvider**: Defines how agents perceive their state (`src/core/state_strategies.py`).
    -   `ReputationStateProvider`: State based on local reputation.
    -   `ActionStateProvider`: State based on current action.
-   **SimulationConfig**: Dataclass for managing all simulation parameters (`src/config.py`).

## Outputs

Results are saved in the specified output directory (e.g., `results/r3.0_inf1.0_reputation/`).
-   `data/experiment_data.h5`: Contains all time-series data and final states.
-   `plots/`: Contains generated figures (if `--plot` is used).
-   `plots/snapshots/`: Contains snapshots of the grid during simulation.
