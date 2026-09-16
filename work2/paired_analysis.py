"""Primary paired intervals fixed by FINAL_TEST_PROTOCOL.md."""
import numpy as np


def primary_mean_intervals(differences):
    """Rows are 30 independent seed blocks; columns are eight primary contrasts."""
    values = np.asarray(differences, dtype=np.float64)
    if values.shape != (30, 8) or not np.isfinite(values).all():
        raise ValueError('Final protocol requires 30 finite seed blocks and eight contrasts')
    rng = np.random.default_rng(20260915)
    indices = rng.integers(0, 30, size=(100000, 30))
    result = []
    # Every contrast uses the same bootstrap seed blocks, preserving dependence.
    for column in values.T:
        bootstrap_means = column[indices].mean(axis=1)
        bounds = np.quantile(bootstrap_means, [.025, .975, .003125, .996875])
        result.append({'n': 30, 'mean_difference': float(column.mean()),
                       'sd_difference': float(column.std(ddof=1)),
                       'minimum_difference': float(column.min()),
                       'maximum_difference': float(column.max()),
                       'interval_95': bounds[:2].tolist(),
                       'interval_99_375': bounds[2:].tolist()})
    return result
