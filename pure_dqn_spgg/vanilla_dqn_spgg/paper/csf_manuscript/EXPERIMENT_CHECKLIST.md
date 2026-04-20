# Paper Experiment Checklist

This checklist is for the manuscript theme:

**How information perception reshapes reinforcement-learning-driven evolution in spatial public goods games.**

The goal is not merely to show that reinforcement learning increases cooperation, but to establish that different state-perception structures change:

- the cooperation threshold
- the equilibrium cooperation level
- the formation of cooperative clusters
- the variability and path dependence near the transition region

All paper-grade quantitative experiments should use:

- lattice size: `100 x 100`, unless explicitly marked as finite-size robustness
- DQN initialization: `zero_last`
- greedy tie-break: `random`
- synchronous updates by default: `update_prob=1.0`
- equilibrium tail window: `5000`

## Core Bundle Set

These bundles are the minimum recommended set for a submission-grade paper.

### `main_unified`

- Purpose: Main quantitative phase diagram under the final initialization setting
- Methods: `fermi tabular_q dqn_self dqn_local dqn_vonn dqn_wide dqn_history`
- `r`: `3.0 3.2 3.4 3.6 3.8 4.0 4.2 4.4 4.6 5.0`
- Seeds: `2026-2035`
- Use in paper:
  - Main phase diagram
  - Main threshold table
  - Representative quantitative comparison table

### `transition_multistability`

- Purpose: Dense multi-seed sweep near the transition region
- Methods: `fermi dqn_local dqn_vonn dqn_history`
- `r`: `3.2 3.4 3.6 3.8`
- Seeds: `2026-2045`
- Raw results: upload enabled
- Use in paper:
  - Final-state distribution plots
  - Cluster statistics
  - Path-dependence and multistability discussion

### `history_h1`, `history_h2`, `history_h3`

- Purpose: Temporal-perception ablation
- Method: `dqn_history`
- `r`: `3.6 4.0 4.4`
- Seeds: `2026-2035`
- Use in paper:
  - History-length ablation figure
  - Discussion of temporal information and non-monotonic effects

### `snapshot_panels`

- Purpose: Manuscript-ready representative trajectories and spatial snapshots
- Methods: `fermi dqn_local dqn_vonn dqn_history`
- `r`: `3.6 4.0 4.4`
- Seed: `2027`
- `save_frames_interval=1000`
- Raw results: upload enabled
- Use in paper:
  - Representative time-series curves
  - Final spatial-pattern panels

## Full Bundle Set

These bundles are recommended if the target is a higher-confidence submission to `Chaos, Solitons & Fractals`.

### `structure_ablation`

- Purpose: Show that the effect comes from information structure, not just larger state dimension
- Methods: `dqn_self dqn_local dqn_vonn dqn_vonn_full dqn_wide dqn_history`
- `r`: `3.4 3.6 3.8 4.0 4.4`
- Seeds: `2026-2035`
- Use in paper:
  - Main or supplementary structure-ablation figure
  - Stronger support for the information-perception theme

### `size_L50`, `size_L150`

- Purpose: Finite-size robustness relative to the main `L=100` results
- Methods: `fermi dqn_local dqn_history`
- `r`: `3.6 4.0 4.4`
- Seeds: `2026-2030`
- Use in paper:
  - Supplementary finite-size robustness table or figure

### `async_spot`

- Purpose: Check that the main conclusions are not purely artifacts of fully synchronous updating
- Methods: `fermi dqn_local dqn_history`
- `r`: `3.6 4.0`
- Seeds: `2026-2030`
- `update_prob=0.5`
- Use in paper:
  - Supplementary robustness discussion

## Recommended Figure Mapping

- Figure 1: model sketch, not from workflow
- Figure 2: `main_unified`
- Figure 3: focused transition comparison from `main_unified`
- Figure 4: `snapshot_panels`
- Figure 5: `history_h1/h2/h3`
- Supplementary Figure S1: `structure_ablation`
- Supplementary Figure S2: `transition_multistability`
- Supplementary Figure S3: `size_L50/size_L150`
- Supplementary Figure S4: `async_spot`

## GitHub Workflow

The full paper experiment workflow is:

- [paper-full-experiments.yml](/Users/jiangchao/workspace/Neighbor-Aware-Reinforcement-Learning-Fosters-Cooperation-in-Spatial-Public-Goods-Games-/.github/workflows/paper-full-experiments.yml)

The workflow reads:

- [gha_paper_experiment_plan.json](/Users/jiangchao/workspace/Neighbor-Aware-Reinforcement-Learning-Fosters-Cooperation-in-Spatial-Public-Goods-Games-/pure_dqn_spgg/vanilla_dqn_spgg/gha_paper_experiment_plan.json)

Profiles:

- `core`: minimum paper-grade run
- `full`: higher-confidence submission run

## Submission Rule

For a stronger submission, do not mix legacy DQN initialization with final figures in the main text. The principal quantitative plots should be generated from the `zero_last + random` configuration only.
