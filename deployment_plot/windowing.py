"""Smooth Acceleration RMS and build gap-valid sliding windows for deployment."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401

from io_utils import parse_timestamp_series
from model_utils import compute_timestep_window_start_indices, smooth_target_series_1d


VALUE_COLUMN = "Acceleration RMS"


@dataclass
class WindowBuildResult:
    sensor_slug: str
    input_csv: str
    output_dir: str
    n_rows: int = 0
    n_windows: int = 0
    window_size: int = 288
    smooth_window: int = 200
    max_gap_seconds: float = 3600.0
    stride: int = 1
    files_written: Dict[str, str] = field(default_factory=dict)


def build_rms_windows_from_csv(
    csv_path: str,
    output_dir: str,
    *,
    value_column: str = VALUE_COLUMN,
    smooth_window: int = 200,
    window_size: int = 288,
    max_gap_seconds: float = 3600.0,
    stride: int = 1,
) -> WindowBuildResult:
    """
    Read a single-sensor CSV, smooth *value_column*, keep TIMESTAMP + RMS only,
    then extract sliding windows where every consecutive pair is <= *max_gap_seconds*.
    """
    csv_path = os.path.abspath(csv_path)
    slug = Path(csv_path).stem
    out_dir = os.path.abspath(output_dir)
    os.makedirs(out_dir, exist_ok=True)

    result = WindowBuildResult(
        sensor_slug=slug,
        input_csv=csv_path,
        output_dir=out_dir,
        window_size=window_size,
        smooth_window=smooth_window,
        max_gap_seconds=max_gap_seconds,
        stride=stride,
    )

    df = pd.read_csv(csv_path)
    if value_column not in df.columns:
        raise ValueError(f"{csv_path}: missing column {value_column!r}")
    if "TIMESTAMP" not in df.columns:
        raise ValueError(f"{csv_path}: missing column 'TIMESTAMP'")

    ts = parse_timestamp_series(df["TIMESTAMP"], name="TIMESTAMP", strict=False)
    valid = ts.notna()
    df = df.loc[valid].copy()
    ts = ts.loc[valid]
    df["_ts_parsed"] = ts.values
    df = df.sort_values("_ts_parsed", kind="mergesort").reset_index(drop=True)
    ts = parse_timestamp_series(df["TIMESTAMP"], name="TIMESTAMP")
    df = df.drop(columns=["_ts_parsed"])

    rms_raw = pd.to_numeric(df[value_column], errors="coerce").to_numpy(dtype=np.float64)
    if np.isnan(rms_raw).any():
        n_bad = int(np.isnan(rms_raw).sum())
        raise ValueError(f"{csv_path}: {n_bad} non-numeric {value_column} value(s).")

    rms_smooth = smooth_target_series_1d(rms_raw, smooth_window)
    if len(rms_smooth) != len(df):
        raise ValueError(f"{csv_path}: smoothed length mismatch.")

    smoothed = pd.DataFrame(
        {
            "TIMESTAMP": df["TIMESTAMP"].astype(str).values,
            value_column: rms_smooth.astype(np.float32),
        }
    )
    result.n_rows = int(len(smoothed))

    smoothed_path = os.path.join(out_dir, "smoothed_acceleration_rms.csv")
    smoothed.to_csv(smoothed_path, index=False)
    result.files_written["smoothed_csv"] = smoothed_path

    starts = compute_timestep_window_start_indices(
        ts,
        window_size,
        max_consecutive_gap_seconds=max_gap_seconds,
    )
    if stride > 1 and starts.size:
        starts = starts[::stride]

    if starts.size == 0:
        windows = np.zeros((0, window_size), dtype=np.float32)
    else:
        rms = rms_smooth.astype(np.float32)
        windows = np.stack([rms[s : s + window_size] for s in starts], axis=0)

    npz_path = os.path.join(out_dir, "windows.npz")
    np.savez(
        npz_path,
        windows=windows,
        start_indices=starts.astype(np.int64),
        window_size=np.int32(window_size),
        smooth_window=np.int32(smooth_window),
        max_gap_seconds=np.float64(max_gap_seconds),
    )
    result.files_written["windows_npz"] = npz_path
    result.n_windows = int(windows.shape[0])

    # Human-readable: one row per window, columns step_1 .. step_288 (Acceleration RMS only).
    if result.n_windows > 0:
        cols = {f"step_{i + 1}": windows[:, i] for i in range(window_size)}
        win_df = pd.DataFrame(cols)
        win_df.insert(0, "window_index", np.arange(result.n_windows, dtype=np.int64))
        win_df.insert(1, "start_row", starts.astype(np.int64))
        win_df.insert(2, "window_start_timestamp", ts.iloc[starts].astype(str).to_numpy())
        windows_csv_path = os.path.join(out_dir, "windows_acceleration_rms.csv")
        win_df.to_csv(windows_csv_path, index=False)
        result.files_written["windows_csv"] = windows_csv_path

    manifest_path = os.path.join(out_dir, "windows_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fp:
        json.dump(
            {
                "sensor_slug": slug,
                "input_csv": csv_path,
                "n_rows": result.n_rows,
                "n_windows": result.n_windows,
                "window_size": window_size,
                "smooth_window": smooth_window,
                "max_gap_seconds": max_gap_seconds,
                "stride": stride,
                "value_column": value_column,
                "files_written": result.files_written,
            },
            fp,
            indent=2,
        )
    result.files_written["manifest"] = manifest_path
    return result


def build_rms_windows_for_directory(
    sensor_csv_dir: str,
    windows_root: str,
    *,
    smooth_window: int = 200,
    window_size: int = 288,
    max_gap_seconds: float = 3600.0,
    stride: int = 1,
    glob_pattern: str = "AHU_*.csv",
) -> List[WindowBuildResult]:
    """Process every per-sensor CSV in *sensor_csv_dir*."""
    sensor_dir = Path(sensor_csv_dir)
    windows_root = Path(windows_root)
    results: List[WindowBuildResult] = []

    for csv_path in sorted(sensor_dir.glob(glob_pattern)):
        if csv_path.name == "preprocess_manifest.json":
            continue
        out_dir = windows_root / csv_path.stem
        results.append(
            build_rms_windows_from_csv(
                str(csv_path),
                str(out_dir),
                smooth_window=smooth_window,
                window_size=window_size,
                max_gap_seconds=max_gap_seconds,
                stride=stride,
            )
        )

    summary_path = windows_root / "windows_summary.json"
    windows_root.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump(
            [
                {
                    "sensor_slug": r.sensor_slug,
                    "n_rows": r.n_rows,
                    "n_windows": r.n_windows,
                    "output_dir": r.output_dir,
                }
                for r in results
            ],
            fp,
            indent=2,
        )

    return results
