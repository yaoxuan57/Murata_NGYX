"""Filter multi-sensor vibration CSVs by SENSOR_DESC and write per-sensor files."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from deployment.io_utils import parse_timestamp_series, read_vibration_export_csv
from deployment.sensors import (
    AHU_2_9_SENSOR_DESCS,
    normalize_sensor_desc,
    resolve_sensor_list,
    sensor_desc_to_slug,
)


@dataclass
class PreprocessResult:
    input_path: str
    output_dir: str
    sensors: List[str] = field(default_factory=list)
    files_written: Dict[str, str] = field(default_factory=dict)
    row_counts: Dict[str, int] = field(default_factory=dict)
    timestamp_min: Dict[str, str] = field(default_factory=dict)
    timestamp_max: Dict[str, str] = field(default_factory=dict)
    rows_dropped_unparsed_ts: int = 0
    rows_dropped_duplicates: int = 0
    max_rows_per_sensor: Optional[int] = None


def filter_by_sensor_desc(
    df: pd.DataFrame,
    sensor_descs: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Keep only rows whose ``SENSOR_DESC`` is in the requested list."""
    targets = set(resolve_sensor_list(sensor_descs))
    df = df.copy()
    df["SENSOR_DESC"] = df["SENSOR_DESC"].map(normalize_sensor_desc)
    out = df[df["SENSOR_DESC"].isin(targets)].copy()
    if out.empty:
        allowed = ", ".join(repr(s) for s in (sensor_descs or AHU_2_9_SENSOR_DESCS))
        raise ValueError(f"No rows matched SENSOR_DESC filter. Expected one of: {allowed}")
    return out


def split_and_write_sensor_csvs(
    df: pd.DataFrame,
    output_dir: str,
    sensor_descs: Optional[List[str]] = None,
    *,
    sort_chronological: bool = True,
    drop_duplicate_timestamps: bool = True,
    max_rows_per_sensor: Optional[int] = None,
) -> PreprocessResult:
    """Write one CSV per sensor (training-ready: single SENSOR_DESC, sorted TIMESTAMP)."""
    targets = resolve_sensor_list(sensor_descs)
    os.makedirs(output_dir, exist_ok=True)

    result = PreprocessResult(
        input_path="",
        output_dir=os.path.abspath(output_dir),
        sensors=targets,
    )

    parsed_ts = parse_timestamp_series(df["TIMESTAMP"], name="TIMESTAMP", strict=False)
    bad = parsed_ts.isna()
    result.rows_dropped_unparsed_ts = int(bad.sum())
    if bad.any():
        df = df.loc[~bad].copy()
        parsed_ts = parsed_ts.loc[~bad]

    df = df.copy()
    df["_ts_parsed"] = parsed_ts.values
    df["SENSOR_DESC"] = df["SENSOR_DESC"].map(normalize_sensor_desc)

    for sensor in targets:
        part = df[df["SENSOR_DESC"] == sensor].copy()
        if part.empty:
            result.row_counts[sensor] = 0
            continue

        if sort_chronological:
            part = part.sort_values("_ts_parsed", kind="mergesort")

        if drop_duplicate_timestamps:
            n_before = len(part)
            part = part.drop_duplicates(subset=["TIMESTAMP"], keep="last")
            result.rows_dropped_duplicates += n_before - len(part)

        if max_rows_per_sensor is not None and max_rows_per_sensor > 0:
            part = part.head(max_rows_per_sensor)

        part = part.drop(columns=["_ts_parsed"])
        slug = sensor_desc_to_slug(sensor)
        out_path = os.path.join(output_dir, f"{slug}.csv")
        part.to_csv(out_path, index=False)

        result.files_written[sensor] = out_path
        result.row_counts[sensor] = int(len(part))
        ts_sorted = parse_timestamp_series(part["TIMESTAMP"])
        result.timestamp_min[sensor] = ts_sorted.min().isoformat()
        result.timestamp_max[sensor] = ts_sorted.max().isoformat()

    manifest_path = os.path.join(output_dir, "preprocess_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fp:
        json.dump(
            {
                "sensors": targets,
                "files_written": result.files_written,
                "row_counts": result.row_counts,
                "timestamp_min": result.timestamp_min,
                "timestamp_max": result.timestamp_max,
                "rows_dropped_unparsed_ts": result.rows_dropped_unparsed_ts,
                "rows_dropped_duplicates": result.rows_dropped_duplicates,
                "max_rows_per_sensor": max_rows_per_sensor,
            },
            fp,
            indent=2,
        )
    result.files_written["__manifest__"] = manifest_path
    return result


def run_preprocess(
    input_csv: str,
    output_dir: str,
    sensor_descs: Optional[List[str]] = None,
    *,
    max_rows_per_sensor: Optional[int] = None,
) -> PreprocessResult:
    """End-to-end: read export CSV → filter SENSOR_DESC → write per-sensor CSVs."""
    df = read_vibration_export_csv(input_csv)
    df = filter_by_sensor_desc(df, sensor_descs=sensor_descs)
    result = split_and_write_sensor_csvs(
        df,
        output_dir,
        sensor_descs=sensor_descs,
        max_rows_per_sensor=max_rows_per_sensor,
    )
    result.input_path = os.path.abspath(input_csv)
    result.max_rows_per_sensor = max_rows_per_sensor
    return result
