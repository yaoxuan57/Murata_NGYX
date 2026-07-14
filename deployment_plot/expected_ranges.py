"""Expected Acceleration RMS bands from train-60% chronological p10–p90 analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from sensors import normalize_sensor_desc


@dataclass(frozen=True)
class TrainingValueRange:
    """Inclusive p10–p90 band where most training data fell."""

    p10: float
    p90: float
    note: str = "train_60pct_chronological_p10_p90"


# Raw Acceleration RMS on chronological train 60% split (data_30mins_frequency CSVs).
TRAIN_P10_P90_BY_SENSOR: Dict[str, TrainingValueRange] = {
    "AHU 2-9 Blower DE A": TrainingValueRange(0.15, 3.50, "AHU_2_9_Blower_DE_A_merged.csv"),
    "AHU 2-9 Blower DE V": TrainingValueRange(2.79, 6.57),
    "AHU 2-9 Blower DE Vibration X": TrainingValueRange(3.19, 6.01),
    "AHU 2-9 Blower NDE A": TrainingValueRange(2.20, 4.79),
    "AHU 2-9 Blower NDE H": TrainingValueRange(2.06, 7.58),
    "AHU 2-9 Blower NDE V": TrainingValueRange(1.78, 6.21),
    "AHU 2-9 motor DE H": TrainingValueRange(10.41, 15.47),
    "AHU 2-9 motor NDE H": TrainingValueRange(4.68, 7.00),
}


def training_range_for_sensor(sensor_desc: str) -> Optional[TrainingValueRange]:
    return TRAIN_P10_P90_BY_SENSOR.get(normalize_sensor_desc(sensor_desc))


def check_context_against_training_range(
    sensor_desc: str,
    context_values: np.ndarray,
) -> Optional[Dict[str, Any]]:
    """
    Return range-check metadata if a known band exists for *sensor_desc*.

    The window is out of range when any smoothed context value falls below p10
    or above p90. Inference still proceeds; callers should surface the warning.
    """
    band = training_range_for_sensor(sensor_desc)
    if band is None:
        return None

    values = np.asarray(context_values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return None

    window_min = float(np.min(values))
    window_max = float(np.max(values))
    window_mean = float(np.mean(values))
    in_range = window_min >= band.p10 and window_max <= band.p90

    result: Dict[str, Any] = {
        "in_training_range": in_range,
        "expected_p10": band.p10,
        "expected_p90": band.p90,
        "window_min": window_min,
        "window_mean": window_mean,
        "window_max": window_max,
        "range_source": band.note,
    }

    if not in_range:
        result["warning"] = (
            f"Model performance may not be good: input window is outside the expected "
            f"training range (p10-p90: {band.p10:.2f} - {band.p90:.2f}). "
            f"Window mean={window_mean:.2f}, max={window_max:.2f}."
        )

    return result
