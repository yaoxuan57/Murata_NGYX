"""Preprocessing helpers."""

from __future__ import annotations

import numpy as np


def smooth_target_series_1d(vec: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return np.asarray(vec, dtype=np.float32)
    w = int(window)
    y = np.asarray(vec, dtype=np.float64)
    n = len(y)
    if n == 0:
        return np.asarray(y, dtype=np.float32)
    csum = np.concatenate(([0.0], np.cumsum(y)))
    idx = np.arange(n, dtype=np.int64)
    start = np.maximum(0, idx - w + 1)
    counts = (idx - start + 1).astype(np.float64)
    out = (csum[idx + 1] - csum[start]) / counts
    return out.astype(np.float32)


def median_quantile_index(quantiles) -> int:
    arr = np.asarray(quantiles, dtype=np.float64)
    return int(np.argmin(np.abs(arr - 0.5)))
