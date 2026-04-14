from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve

_CROSS_KERNEL = np.array(
    [
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)

_WIDE_KERNEL = np.ones((5, 5), dtype=np.float32)

PAYOFF_RAW_MIN = -5.0
PAYOFF_RAW_MAX = 25.0


def overlap5(a: np.ndarray) -> np.ndarray:
    return convolve(a, _CROSS_KERNEL, mode="wrap")


def overlap_wide(a: np.ndarray) -> np.ndarray:
    return convolve(a, _WIDE_KERNEL, mode="wrap")


def von_neumann_neighbors_sum(data: np.ndarray) -> np.ndarray:
    kernel = np.array(
        [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ],
        dtype=np.float32,
    )
    return convolve(data, kernel, mode="wrap")


def normalize_payoff_fixed(payoff_raw: np.ndarray) -> np.ndarray:
    return np.clip(
        (payoff_raw - PAYOFF_RAW_MIN) / (PAYOFF_RAW_MAX - PAYOFF_RAW_MIN),
        0.0,
        1.0,
    ).astype(np.float32)
