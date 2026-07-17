"""Split each jun_by_sensor/*.csv into chronological train/val/test (50/25/25)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FINETUNE_DIR = Path(__file__).resolve().parent
JUN_DIR = FINETUNE_DIR / "jun_by_sensor"
DEFAULT_SUMMARY = JUN_DIR / "split_summary.json"

TIME_COL = "TIMESTAMP"
VALUE_COL = "Acceleration RMS"
SENSOR_COL = "SENSOR_DESC"
SENSOR_NAME_COL = "SENSOR_NAME"


def parse_ts(series: pd.Series) -> pd.Series:
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


def chrono_row_counts(
    n: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[int, int, int]:
    if n < 3:
        raise ValueError(f"Need at least 3 rows to split; got {n}.")
    weights = np.asarray([train_ratio, val_ratio, test_ratio], dtype=np.float64)
    if np.any(weights < 0) or float(weights.sum()) <= 0:
        raise ValueError("train/val/test ratios must be non-negative and sum > 0.")
    weights = weights / weights.sum()
    n_tr = int(np.floor(n * weights[0]))
    n_va = int(np.floor(n * weights[1]))
    n_te = n - n_tr - n_va
    if min(n_tr, n_va, n_te) < 1:
        raise ValueError(
            f"Split too small after ratios (train={n_tr}, val={n_va}, test={n_te} from n={n})."
        )
    return n_tr, n_va, n_te


def load_sensor_frame(path: Path, sensor_name: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if VALUE_COL not in df.columns and "DATA12" in df.columns:
        df[VALUE_COL] = pd.to_numeric(df["DATA12"], errors="coerce")
    if SENSOR_COL not in df.columns and SENSOR_NAME_COL in df.columns:
        df[SENSOR_COL] = df[SENSOR_NAME_COL].astype(str).str.strip()

    df[TIME_COL] = parse_ts(df[TIME_COL])
    if sensor_name and SENSOR_COL in df.columns:
        df = df[df[SENSOR_COL].astype(str).str.strip() == sensor_name]
    return (
        df.sort_values(TIME_COL, kind="mergesort")
        .drop_duplicates(TIME_COL, keep="last")
        .reset_index(drop=True)
    )


def split_jun_csv(
    source: Path,
    out_dir: Path,
    *,
    sensor_name: str | None = None,
    train_ratio: float = 0.50,
    val_ratio: float = 0.25,
    test_ratio: float = 0.25,
) -> dict:
    df = load_sensor_frame(source, sensor_name=sensor_name)
    n = len(df)
    n_tr, n_va, n_te = chrono_row_counts(n, train_ratio, val_ratio, test_ratio)

    train = df.iloc[:n_tr].copy()
    val = df.iloc[n_tr : n_tr + n_va].copy()
    test = df.iloc[n_tr + n_va :].copy()

    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(out_dir / "train.csv", index=False)
    val.to_csv(out_dir / "val.csv", index=False)
    test.to_csv(out_dir / "test.csv", index=False)

    manifest = {
        "mode": "jun_only_chronological_50_25_25",
        "source": str(source.resolve()),
        "sensor": sensor_name,
        "ratios": {
            "train": float(train_ratio),
            "val": float(val_ratio),
            "test": float(test_ratio),
        },
        "rows": {"train": len(train), "val": len(val), "test": len(test), "total": n},
        "time": {
            "train": [str(train[TIME_COL].iloc[0]), str(train[TIME_COL].iloc[-1])],
            "val": [str(val[TIME_COL].iloc[0]), str(val[TIME_COL].iloc[-1])],
            "test": [str(test[TIME_COL].iloc[0]), str(test[TIME_COL].iloc[-1])],
        },
        "note": "Split from jun_by_sensor CSV only; train is earliest, test is latest.",
        "outputs": {
            "train": str((out_dir / "train.csv").resolve()),
            "val": str((out_dir / "val.csv").resolve()),
            "test": str((out_dir / "test.csv").resolve()),
        },
    }
    (out_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY,
        help="split_summary.json from split_jun_by_sensor.py",
    )
    parser.add_argument("--train-ratio", type=float, default=0.50)
    parser.add_argument("--val-ratio", type=float, default=0.25)
    parser.add_argument("--test-ratio", type=float, default=0.25)
    args = parser.parse_args()

    if not args.summary.is_file():
        raise SystemExit(f"Summary not found: {args.summary}")

    payload = json.loads(args.summary.read_text(encoding="utf-8"))
    sensors = payload.get("sensors") or []
    if not sensors:
        raise SystemExit(f"No sensors listed in {args.summary}")

    manifests = []
    for row in sensors:
        stem = row["stem"]
        sensor_name = row.get("sensor")
        source = JUN_DIR / f"{stem}.csv"
        out_dir = FINETUNE_DIR / f"data_{stem}" / "splits"
        print(f"\n=== {stem} ===")
        manifest = split_jun_csv(
            source,
            out_dir,
            sensor_name=sensor_name,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
        )
        manifests.append(manifest)
        print(
            f"  train: {manifest['rows']['train']} rows  "
            f"{manifest['time']['train'][0]} -> {manifest['time']['train'][1]}"
        )
        print(
            f"  val:   {manifest['rows']['val']} rows  "
            f"{manifest['time']['val'][0]} -> {manifest['time']['val'][1]}"
        )
        print(
            f"  test:  {manifest['rows']['test']} rows  "
            f"{manifest['time']['test'][0]} -> {manifest['time']['test'][1]}"
        )

    summary_path = FINETUNE_DIR / "jun_finetune_splits_summary.json"
    summary_path.write_text(json.dumps(manifests, indent=2), encoding="utf-8")
    print(f"\nWrote {len(manifests)} sensor split(s)")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
