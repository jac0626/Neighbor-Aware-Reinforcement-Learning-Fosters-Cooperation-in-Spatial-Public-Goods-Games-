# Submission Experiments

This repository can now run a publication-oriented experiment matrix around the
question:

How do spatial range, directional structure, and temporal depth of agent state
perception affect cooperation in the spatial public goods game?

## Main Sweep

Use this as the primary phase-transition experiment in the manuscript.

```bash
python3 batch_runner.py \
  --output-root results/submission_main \
  --methods fermi tabular_q dqn_self dqn_local dqn_vonn dqn_wide dqn_history \
  --r-values 2.0 2.5 3.0 3.2 3.4 3.6 3.8 4.0 4.2 4.4 4.6 5.0 \
  --seeds 2026 2027 2028 2029 2030 2031 2032 2033 2034 2035 \
  --grid-size 100 \
  --iterations 100000 \
  --history-len 3 \
  --tail-length 5000 \
  --update-prob 1.0 \
  --workers 4
```

Recommended outputs:

- `summary.csv`: main quantitative table source
- `experiment_manifest.json`: exact run configuration
- `plot_phase_diagram.py --summary results/submission_main/summary.csv`

## Structure Ablation

Use this to isolate whether gains come from information content rather than
simply larger state dimensionality.

```bash
python3 batch_runner.py \
  --output-root results/submission_structure \
  --methods dqn_self dqn_local dqn_vonn dqn_vonn_full dqn_wide dqn_history \
  --r-values 3.2 3.4 3.6 3.8 4.0 4.2 4.4 \
  --seeds 2026 2027 2028 2029 2030 2031 2032 2033 2034 2035 \
  --grid-size 100 \
  --iterations 100000 \
  --history-len 3 \
  --tail-length 5000 \
  --update-prob 1.0 \
  --workers 4
```

Interpretation focus:

- `self -> local`: effect of adding local density
- `local -> vonn`: density vs directional topology
- `vonn -> vonn_full`: aggregate density vs explicit neighbor actions
- `local -> history`: instantaneous perception vs temporal memory
- `local -> wide`: short-range vs extended-range perception

## Robustness

Use this to show that conclusions are not artifacts of a single seed or one
update schedule.

```bash
python3 batch_runner.py \
  --output-root results/submission_robustness_sync \
  --methods dqn_self dqn_local dqn_vonn dqn_wide dqn_history \
  --r-values 3.4 3.6 4.0 4.4 \
  --seeds 2026 2027 2028 2029 2030 2031 2032 2033 2034 2035 2036 2037 2038 2039 2040 2041 2042 2043 2044 2045 \
  --grid-size 100 \
  --iterations 100000 \
  --history-len 3 \
  --tail-length 5000 \
  --update-prob 1.0 \
  --workers 4
```

```bash
python3 batch_runner.py \
  --output-root results/submission_robustness_async \
  --methods dqn_self dqn_local dqn_vonn dqn_wide dqn_history \
  --r-values 3.4 3.6 4.0 4.4 \
  --seeds 2026 2027 2028 2029 2030 2031 2032 2033 2034 2035 2036 2037 2038 2039 2040 2041 2042 2043 2044 2045 \
  --grid-size 100 \
  --iterations 100000 \
  --history-len 3 \
  --tail-length 5000 \
  --update-prob 0.5 \
  --workers 4
```

Key statistics to report:

- mean tail cooperation over the last 5000 steps
- standard deviation across seeds
- mean tail raw payoff
- representative spatial snapshots near the transition region
