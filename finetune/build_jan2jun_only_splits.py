"""Build train/val/test from Jan–Jun (or any) sensor CSV with chronological 70/15/15.

Order by TIMESTAMP, then:
  first 70% → train
  next  15% → val
  last  15% → test   (remainder so rows always sum to N)

Test is always after train/val — never earlier on the timeline.

Example:
  python finetune/build_jan2jun_only_splits.py
  python finetune/build_jan2jun_only_splits.py \\
    --source finetune/data_AHU_2_9_Blower_DE_A/AHU_2_9_Blower_DE_A_jan2_jun_30_min.csv \\
    --out-dir finetune/data_AHU_2_9_Blower_DE_A/splits
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "finetune" / "data_AHU_2_9_Blower_DE_A"
DEFAULT_OUT = DATA_DIR / "splits"
DEFAULT_SOURCE = DATA_DIR / "AHU_2_9_Blower_DE_A_jan2_jun_30_min.csv"
TIME_COL = "TIMESTAMP"
VALUE_COL = "Acceleration RMS"
SENSOR_COL = "SENSOR_DESC"
DEFAULT_SENSOR = "AHU 2-9 Blower DE A"
TR, VA, TE = 0.70, 0.15, 0.15


def parse_ts(s: pd.Series) -> pd.Series:
    raw = s.astype(str).str.strip()
    p = pd.to_datetime(raw, dayfirst=True, format="mixed", errors="coerce")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        m = p.isna()
        if not m.any():
            break
        p.loc[m] = pd.to_datetime(raw.loc[m], format=fmt, errors="coerce")
    return p


def chrono_row_counts(n: int, train_ratio: float, val_ratio: float, test_ratio: float) -> tuple[int, int, int]:
    """First train%, then val%, remainder = test (always last)."""
    if n < 3:
        raise ValueError(f"Need at least 3 rows to split; got {n}.")
    w = np.asarray([train_ratio, val_ratio, test_ratio], dtype=np.float64)
    if np.any(w < 0) or float(w.sum()) <= 0:
        raise ValueError("train/val/test ratios must be non-negative and sum > 0.")
    w = w / w.sum()
    n_tr = int(np.floor(n * w[0]))
    n_va = int(np.floor(n * w[1]))
    # Remainder → test so train time < val time < test time and all rows used.
    n_te = n - n_tr - n_va
    if min(n_tr, n_va, n_te) < 1:
        raise ValueError(
            f"Split too small after ratios (train={n_tr}, val={n_va}, test={n_te} from n={n})."
        )
    return n_tr, n_va, n_te


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--sensor",
        type=str,
        default=DEFAULT_SENSOR,
        help="If SENSOR_DESC/SENSOR_NAME exists, keep only this sensor. "
        "Pass empty string to keep all rows.",
    )
    p.add_argument("--train-ratio", type=float, default=TR)
    p.add_argument("--val-ratio", type=float, default=VA)
    p.add_argument("--test-ratio", type=float, default=TE)
    args = p.parse_args()

    if not args.source.is_file():
        raise FileNotFoundError(f"Source CSV not found: {args.source}")

    df = pd.read_csv(args.source, low_memory=False)
    if VALUE_COL not in df.columns and "DATA12" in df.columns:
        df[VALUE_COL] = pd.to_numeric(df["DATA12"], errors="coerce")
    if SENSOR_COL not in df.columns and "SENSOR_NAME" in df.columns:
        df[SENSOR_COL] = df["SENSOR_NAME"].astype(str).str.strip()

    df[TIME_COL] = parse_ts(df[TIME_COL])
    sensor = (args.sensor or "").strip()
    if sensor and SENSOR_COL in df.columns:
        df = df[df[SENSOR_COL].astype(str).str.strip() == sensor]
    df = df.sort_values(TIME_COL).drop_duplicates(TIME_COL, keep="last").reset_index(drop=True)

    n = len(df)
    n_tr, n_va, n_te = chrono_row_counts(n, args.train_ratio, args.val_ratio, args.test_ratio)
    train = df.iloc[:n_tr]
    val = df.iloc[n_tr : n_tr + n_va]
    test = df.iloc[n_tr + n_va :]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(args.out_dir / "train.csv", index=False)
    val.to_csv(args.out_dir / "val.csv", index=False)
    test.to_csv(args.out_dir / "test.csv", index=False)

    manifest = {
        "mode": "chronological_70_15_15",
        "source": str(args.source.resolve()),
        "sensor_filter": sensor or None,
        "ratios": {
            "train": float(args.train_ratio),
            "val": float(args.val_ratio),
            "test": float(args.test_ratio),
        },
        "rows": {"train": len(train), "val": len(val), "test": len(test), "total": n},
        "time": {
            "train": [str(train[TIME_COL].iloc[0]), str(train[TIME_COL].iloc[-1])],
            "val": [str(val[TIME_COL].iloc[0]), str(val[TIME_COL].iloc[-1])],
            "test": [str(test[TIME_COL].iloc[0]), str(test[TIME_COL].iloc[-1])],
        },
        "note": "train is earlier than val, val earlier than test (no test before train).",
    }
    (args.out_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"\nWrote {args.out_dir / 'train.csv'}")
    print(f"Wrote {args.out_dir / 'val.csv'}")
    print(f"Wrote {args.out_dir / 'test.csv'}")


if __name__ == "__main__":
    main()
