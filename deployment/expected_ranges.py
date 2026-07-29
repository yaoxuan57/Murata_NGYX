"""Training-range checks using percentile bands stored in model checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import numpy as np


@dataclass(frozen=True)
class TrainingValueRange:
    """Inclusive percentile band where training data fell."""

    low: float
    high: float
    low_label: str
    high_label: str
    note: str = "train_split_percentiles"


def training_range_from_checkpoint(
    training_value_range: Mapping[str, Any] | None,
    *,
    band: str = "p10_p90",
) -> Optional[TrainingValueRange]:
    """Build a band from ``training_value_range`` saved in a ``.pth`` checkpoint."""
    if not training_value_range:
        return None
    if band == "p05_p95":
        low_key, high_key = "p05", "p95"
    else:
        low_key, high_key = "p10", "p90"
    low = training_value_range.get(low_key)
    high = training_value_range.get(high_key)
    if low is None or high is None:
        return None
    smoothed = bool(training_value_range.get("smoothed"))
    value_column = str(training_value_range.get("value_column") or "target")
    note = f"checkpoint_{band}_{'smoothed' if smoothed else 'raw'}_{value_column}"
    return TrainingValueRange(float(low), float(high), low_key, high_key, note)


def check_context_against_training_range(
    sensor_desc: str,
    context_values: np.ndarray,
    *,
    training_value_range: Mapping[str, Any] | None = None,
    band: str = "p10_p90",
) -> Optional[Dict[str, Any]]:
    """
    Return range-check metadata when the checkpoint contains ``training_value_range``.

    Uses p10/p90 from the model ``.pth`` by default. The window is out of range when
    any smoothed context value falls below p10 or above p90. Inference still proceeds.
    """
    _ = sensor_desc  # kept for call-site compatibility
    band_obj = training_range_from_checkpoint(training_value_range, band=band)
    if band_obj is None:
        return None

    values = np.asarray(context_values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return None

    window_min = float(np.min(values))
    window_max = float(np.max(values))
    window_mean = float(np.mean(values))
    in_range = window_min >= band_obj.low and window_max <= band_obj.high

    result: Dict[str, Any] = {
        "in_training_range": in_range,
        "band": band,
        f"expected_{band_obj.low_label}": band_obj.low,
        f"expected_{band_obj.high_label}": band_obj.high,
        "window_min": window_min,
        "window_mean": window_mean,
        "window_max": window_max,
        "range_source": band_obj.note,
        "training_value_range": dict(training_value_range),
    }

    if not in_range:
        result["warning"] = (
            f"Model performance may not be good: input window is outside the expected "
            f"training range ({band_obj.low_label}-{band_obj.high_label}: "
            f"{band_obj.low:.2f} - {band_obj.high:.2f}). "
            f"Window mean={window_mean:.2f}, max={window_max:.2f}."
        )

    return result
