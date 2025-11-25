---
description: Run SPGG experiments with various configurations
---

This workflow allows you to run Spatial Public Goods Game (SPGG) simulations. You can run a single instance, a parameter sweep, or use a configuration file.

# 1. Run a Single Simulation
Run a simulation with specific `r` (synergy factor) and `influence_factor` values.

// turbo
```bash
python run_experiment.py --r 3.0 --influence_factor 1.0 --state_type reputation --plot
```

# 2. Run a Parameter Sweep
Run multiple simulations in parallel by specifying lists of values for `r` and `influence_factor`.

// turbo
```bash
python run_experiment.py --r 3.0 4.0 --influence_factor 0.0 1.0 --parallel --plot
```

# 3. Run with Configuration File
Load parameters from a YAML configuration file.

// turbo
```bash
python run_experiment.py --config configs/default_config.yaml
```

# 4. View Results
Results are saved in the `results/` directory (or the directory specified by `--output_dir`).
You can check the generated plots in the `plots/` subdirectory of each experiment folder.

// turbo
```bash
ls -R results/
```
