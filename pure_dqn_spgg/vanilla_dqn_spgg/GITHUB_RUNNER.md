# GitHub Runner Experiments

This project now includes a GitHub Actions workflow for large experiment
replication on GitHub-hosted runners:

- Workflow: `.github/workflows/spgg-github-runner.yml`
- Experiment scripts:
  - `gha_resolve_preset.py`
  - `gha_build_matrix.py`
  - `gha_run_shard.py`
  - `gha_merge_shards.py`
  - `gha_experiment_presets.json`

## What It Does

The workflow is designed for large SPGG sweeps that would be too slow to run as
one serial job.

1. `prepare` builds an explicit shard matrix from `methods × r_values × seeds`.
2. `run-shard` executes each shard on its own GitHub-hosted runner.
3. `aggregate` downloads all shard summaries, merges them, and generates a
   phase diagram artifact.

The GitHub-hosted version is tuned for summary-first replication:

- more shards to reduce per-job runtime
- optional multi-process execution inside each shard
- capped BLAS/PyTorch threads per worker to avoid oversubscription

Each shard writes outputs to:

```text
pure_dqn_spgg/vanilla_dqn_spgg/results/github_actions/<output_name>-run<run_number>/shard-XX/
```

The aggregate summary is written to:

```text
pure_dqn_spgg/vanilla_dqn_spgg/results/github_actions/<output_name>-run<run_number>/aggregate/
```

## Experiment Selection

The workflow now supports an `experiment_preset` input. In the GitHub Actions
UI you can choose one of:

- `smoke`
- `submission_main`
- `submission_structure`
- `submission_robustness_sync`
- `submission_robustness_async`
- `custom`

If you choose `custom`, the workflow uses the raw values you type into
`methods`, `r_values`, `seeds`, `iterations`, and the other parameter fields.

If you choose any non-`custom` preset, those fields are automatically replaced
by the preset configuration from `gha_experiment_presets.json`.

## Recommended Presets

- `submission_main`: main paper-quality phase diagram
- `submission_structure`: structure and information-content ablation
- `submission_robustness_sync`: multi-seed robustness under synchronous updates
- `submission_robustness_async`: multi-seed robustness under asynchronous updates
- `smoke`: quick CI sanity check

For `submission_main`, the preset expands to:

- `methods`: `fermi tabular_q dqn_self dqn_local dqn_vonn dqn_wide dqn_history`
- `r_values`: `2.0 2.5 3.0 3.2 3.4 3.6 3.8 4.0 4.2 4.4 4.6 5.0`
- `seeds`: `2026 2027 2028 2029 2030 2031 2032 2033 2034 2035`
- `shards`: `96`
- `grid_size`: `100`
- `iterations`: `100000`
- `tail_length`: `5000`
- `workers_per_runner`: `2`
- `threads_per_worker`: `1`
- `save_frames_interval`: `0`
- `save_models`: `false`
- `upload_raw_results`: `false` for scouting, `true` only when you really need
  the raw HDF5 outputs

## Why Smaller Shards

GitHub-hosted jobs have a hard execution cap of 6 hours, and GitHub Actions
matrix jobs are capped at 256 jobs per workflow run. The current presets stay
within that matrix limit while pushing the per-shard job count down far enough
that long DQN sweeps are less likely to be cancelled.

If one preset still times out, do not increase `workers_per_runner` first.
Instead:

1. increase `shards`
2. keep `threads_per_worker=1`
3. keep `save_frames_interval=0`
4. keep `save_models=false` unless checkpoints are necessary

## Why `save_models=false`

For GitHub-hosted runners, artifact size matters. DQN checkpoints are usually
not needed for phase-diagram replication, so the workflow defaults to
checkpoint-free shard runs unless you explicitly enable model saving.

## Typical Use

1. Push the branch to GitHub.
2. Open `Actions`.
3. Run `SPGG GitHub Runner`.
4. Select `experiment_preset`.
5. If needed, switch to `custom` and edit the raw parameter fields.
6. For custom runs, prefer `workers_per_runner=2` and `threads_per_worker=1` on GitHub-hosted Linux runners.
7. Keep snapshot-heavy figure generation local unless the underlying parameter scan itself is too large.
8. Download the final `spgg-aggregate-*` artifact after the workflow completes.

## Keep Local

These are usually better left on your local machine:

- representative spatial snapshots
- long time-evolution curves for a few selected `r` values
- final publication figure polishing

GitHub-hosted runners are best used for summary-oriented sweeps that end in
`summary.csv`, not for artifact-heavy visualization runs.

## Outputs You Get

- Per-shard `summary.csv`
- Per-shard `shard_manifest.json`
- Aggregate `summary.csv`
- Aggregate `merged_manifest.json`
- Aggregate `phase_diagram.png`
