"""Shared preprocessing helpers (inference + legacy windowing). Standalone copy."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def smooth_target_series_1d(vec: np.ndarray, window: int) -> np.ndarray:
    """Causal (trailing) moving average; window length W, min_periods=1 at the series start."""
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


def compute_timestep_window_start_indices(
    timestamps: pd.Series,
    span_len: int,
    *,
    nominal_seconds: Optional[float] = None,
    tolerance_seconds: float = 1.01,
    max_consecutive_gap_seconds: Optional[float] = None,
) -> np.ndarray:
    """Row indices ``i`` where window ``[i, i + span_len)`` passes timestamp rules."""
    ts = pd.Series(timestamps).reset_index(drop=True)
    if span_len <= 0:
        raise ValueError("span_len must be positive.")
    n = len(ts)
    if n < span_len:
        return np.array([], dtype=np.int64)
    if span_len == 1:
        return np.arange(n, dtype=np.int64)

    if nominal_seconds is None and max_consecutive_gap_seconds is None:
        return np.arange(n - span_len + 1, dtype=np.int64)

    values = ts.to_numpy(dtype="datetime64[ns]")
    diffs_ns = np.diff(values.astype("int64"))

    if nominal_seconds is None:
        uniform_ok = np.ones(diffs_ns.shape[0], dtype=bool)
    else:
        nominal_ns = int(round(float(nominal_seconds) * 1e9))
        tol_ns = int(round(float(tolerance_seconds) * 1e9))
        uniform_ok = np.abs(diffs_ns - nominal_ns) <= tol_ns

    if max_consecutive_gap_seconds is None:
        gap_ok = np.ones(diffs_ns.shape[0], dtype=bool)
    else:
        max_ns = int(round(float(max_consecutive_gap_seconds) * 1e9))
        gap_ok = diffs_ns <= max_ns

    step_ok = uniform_ok & gap_ok
    m = span_len - 1
    if step_ok.size < m:
        return np.array([], dtype=np.int64)
    conv = np.convolve(step_ok.astype(np.int32), np.ones(m, dtype=np.int32), mode="valid")
    return np.where(conv == m)[0].astype(np.int64)
