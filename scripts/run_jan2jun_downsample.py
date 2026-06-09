"""Split data/Jan2Jun.csv by sensor and downsample to 30 min (notebook-equivalent)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

TIME_COL = "TIMESTAMP"
TARGET_STEP_SECONDS = 30 * 60
MULTI = REPO / "data" / "Jan2Jun.csv"
SPLIT_DIR = REPO / "data" / "jan2jun_by_sensor"
OUT_DIR = REPO / "data_30mins_frequency"
METHOD = "pick_last_in_bin"


def sensor_desc_to_stem(desc: str) -> str:
    slug = str(desc).strip().replace(" ", "_")
    slug = re.sub(r"[^\w\-]+", "_", slug)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug.replace("2-9", "2_9")


def parse_timestamp_series(series: pd.Series) -> pd.Series:
    raw = series.astype(str).str.strip()
    parsed = pd.to_datetime(raw, dayfirst=True, format="mixed", errors="coerce")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        mask = parsed.isna()
        if not mask.any():
            break
        parsed.loc[mask] = pd.to_datetime(raw.loc[mask], format=fmt, errors="coerce")
    if int(parsed.isna().sum()):
        raise ValueError(f"Failed to parse {int(parsed.isna().sum())} timestamps.")
    return parsed


def load_sensor_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    out = df.copy()
    out[TIME_COL] = parse_timestamp_series(out[TIME_COL])
    return out.sort_values(TIME_COL, kind="mergesort").drop_duplicates(TIME_COL, keep="last").reset_index(drop=True)


def downsample_pick_row_per_bin(df: pd.DataFrame, *, which: str = "last") -> pd.DataFrame:
    work = df.sort_values(TIME_COL, kind="mergesort").copy()
    bins = work[TIME_COL].dt.floor("30min")
    pick_idx = work.groupby(bins, sort=True)[TIME_COL].idxmax() if which == "last" else work.groupby(bins, sort=True)[TIME_COL].idxmin()
    return df.loc[pick_idx].sort_values(TIME_COL, kind="mergesort").reset_index(drop=True)


def split_multi_sensor(src: Path, out_dir: Path) -> list[tuple[str, Path, int]]:
    raw = pd.read_csv(src, low_memory=False)
    if "SENSOR_DESC" in raw.columns:
        sensor_col = "SENSOR_DESC"
    elif "SENSOR_NAME" in raw.columns:
        raw = raw.copy()
        raw["SENSOR_DESC"] = raw["SENSOR_NAME"].astype(str).str.strip()
        sensor_col = "SENSOR_DESC"
    else:
        raise ValueError("Need SENSOR_DESC or SENSOR_NAME")
    if "DATA12" in raw.columns and "Acceleration RMS" not in raw.columns:
        raw["Acceleration RMS"] = pd.to_numeric(raw["DATA12"], errors="coerce")
    out_dir.mkdir(parents=True, exist_ok=True)
    stems = []
    for desc, grp in raw.groupby(sensor_col, sort=True):
        stem = sensor_desc_to_stem(desc)
        out = out_dir / f"{stem}.csv"
        grp.sort_values(TIME_COL, kind="mergesort").to_csv(out, index=False)
        stems.append((desc, out, len(grp)))
    return stems


def main() -> None:
    if not MULTI.is_file():
        raise SystemExit(f"Missing {MULTI}")
    print(f"Split {MULTI.name} ...")
    split_info = split_multi_sensor(MULTI, SPLIT_DIR)
    for desc, path, n in split_info:
        print(f"  {n:6,} rows  {path.name}  ({desc})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    for path in sorted(SPLIT_DIR.glob("*.csv")):
        df = load_sensor_csv(path)
        n_before = len(df)
        out = downsample_pick_row_per_bin(df, which="last")
        n_after = len(out)
        out_path = OUT_DIR / f"{path.stem}_30_min.csv"
        out.to_csv(out_path, index=False)
        med = out[TIME_COL].diff().dt.total_seconds().dropna().median() if n_after > 1 else float("nan")
        summary.append((path.name, n_before, n_after, med, out_path.name))
        print(f"OK {path.name}: {n_before:,} -> {n_after:,} (median dt {med:.0f} s) -> {out_path.name}")

    print(f"\nWrote {len(summary)} files under {OUT_DIR}")


if __name__ == "__main__":
    main()
