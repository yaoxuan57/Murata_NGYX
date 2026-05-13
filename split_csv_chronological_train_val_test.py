#!/usr/bin/env python3
"""Chronological row split of one CSV into train / val / test files (for transformer / sweep scripts)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from forecast_sweep_common import parse_timestamp_series  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(
        description="Sort by TIMESTAMP and split rows into train/val/test CSVs (default 60/20/20)."
    )
    p.add_argument("--input", "-i", type=Path, required=True, help="Source CSV path.")
    p.add_argument(
        "--out-dir",
        "-o",
        type=Path,
        required=True,
        help="Directory to write train.csv, val.csv, test.csv (created if missing).",
    )
    p.add_argument("--train-ratio", type=float, default=0.6)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--test-ratio", type=float, default=0.2)
    p.add_argument("--timestamp-column", type=str, default="TIMESTAMP")
    args = p.parse_args()

    inp = args.input.expanduser().resolve()
    if not inp.is_file():
        raise SystemExit(f"Input not found: {inp}")

    rsum = float(args.train_ratio + args.val_ratio + args.test_ratio)
    if abs(rsum - 1.0) > 1e-6:
        raise SystemExit(f"Ratios must sum to 1.0, got {rsum}")

    df = pd.read_csv(inp)
    if args.timestamp_column not in df.columns:
        raise SystemExit(f"Missing column {args.timestamp_column!r}. Columns: {list(df.columns)}")

    df = df.copy()
    df[args.timestamp_column] = parse_timestamp_series(df[args.timestamp_column], str(inp))
    df = df.sort_values(args.timestamp_column).reset_index(drop=True)

    n = len(df)
    if n < 3:
        raise SystemExit(f"Need at least 3 rows to split, got {n}")

    n_train = int(np.floor(n * args.train_ratio))
    n_val = int(np.floor(n * args.val_ratio))
    n_test = n - n_train - n_val
    if n_train < 1 or n_val < 1 or n_test < 1:
        raise SystemExit(
            f"Split produced empty part: n={n}, n_train={n_train}, n_val={n_val}, n_test={n_test}. "
            "Adjust ratios or use more rows."
        )

    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train : n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val :].copy()

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train.csv"
    val_path = out_dir / "val.csv"
    test_path = out_dir / "test.csv"
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    manifest = {
        "source_csv": str(inp),
        "out_dir": str(out_dir),
        "timestamp_column": args.timestamp_column,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "total_rows": n,
        "train_rows": n_train,
        "val_rows": n_val,
        "test_rows": n_test,
        "train_time_range": [
            str(train_df[args.timestamp_column].iloc[0]),
            str(train_df[args.timestamp_column].iloc[-1]),
        ],
        "val_time_range": [
            str(val_df[args.timestamp_column].iloc[0]),
            str(val_df[args.timestamp_column].iloc[-1]),
        ],
        "test_time_range": [
            str(test_df[args.timestamp_column].iloc[0]),
            str(test_df[args.timestamp_column].iloc[-1]),
        ],
        "files": {
            "train": str(train_path),
            "val": str(val_path),
            "test": str(test_path),
        },
    }
    manifest_path = out_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote {train_path} ({n_train} rows)")
    print(f"Wrote {val_path} ({n_val} rows)")
    print(f"Wrote {test_path} ({n_test} rows)")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
