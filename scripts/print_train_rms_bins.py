#!/usr/bin/env python3
"""Train RMS distribution tables (Option B) for all 30-min sensor CSVs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from forecast_sweep_common import parse_timestamp_series  # noqa: E402

from scripts.analyze_window_rms_distribution import analyze_split  # noqa: E402

# Contiguous bins covering [0, inf); percentages sum to 100%.
FULL_BINS: list[tuple[float, float, str]] = [
    (0.0, 0.5, "0 – 0.5"),
    (0.5, 1.0, "0.5 – 1.0"),
    (1.0, 1.5, "1.0 – 1.5"),
    (1.5, 2.0, "1.5 – 2.0"),
    (2.0, 2.5, "2.0 – 2.5"),
    (2.5, 3.0, "2.5 – 3.0"),
    (3.0, 3.5, "3.0 – 3.5"),
    (3.5, 4.0, "3.5 – 4.0"),
    (4.0, 5.0, "4.0 – 5.0"),
    (5.0, 6.0, "5.0 – 6.0"),
    (6.0, 8.0, "6.0 – 8.0"),
    (8.0, 10.0, "8.0 – 10.0"),
    (10.0, 15.0, "10.0 – 15.0"),
    (15.0, float("inf"), ">= 15.0"),
]


def train_covered(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    df["TIMESTAMP"] = parse_timestamp_series(df["TIMESTAMP"], str(path))
    df = df.sort_values("TIMESTAMP").reset_index(drop=True)
    n_tr = int(np.floor(len(df) * 0.6))
    train = df.iloc[:n_tr]
    span = 96
    wk = {
        "span_len": span,
        "nominal_seconds": 1800.0,
        "tolerance_seconds": 60.0,
        "max_consecutive_gap_seconds": 1860.0,
    }
    return analyze_split(
        "train", train, input_len=48, pred_len=48, smooth_window=48, wk=wk
    )["covered_unique"]


def bin_table(cov: np.ndarray) -> list[tuple[str, float, int]]:
    n = cov.size
    rows: list[tuple[str, float, int]] = []
    for lo, hi, lab in FULL_BINS:
        if np.isinf(hi):
            mask = cov >= lo
        else:
            mask = (cov >= lo) & (cov < hi)
        cnt = int(mask.sum())
        pct = 100.0 * cnt / n if n else 0.0
        rows.append((lab, pct, cnt))
    return rows


def print_sensor(label: str, cov: np.ndarray, *, show_zero: bool) -> None:
    print("=" * 60)
    print(label)
    n = cov.size
    if n == 0:
        print("(no train rows in any window)")
        return
    print(f"n = {n:,} unique rows  |  median = {np.median(cov):.3f}  |  mean = {cov.mean():.3f}")
    print()
    print(f"{'RMS range (smoothed)':<18} {'%':>8}  {'count':>8}")
    print("-" * 38)
    total_pct = 0.0
    for lab, pct, cnt in bin_table(cov):
        if not show_zero and pct < 0.05:
            continue
        print(f"{lab:<18} {pct:7.1f}%  {cnt:8,}")
        total_pct += pct
    print("-" * 38)
    print(f"{'TOTAL':<18} {total_pct:7.1f}%")
    if abs(total_pct - 100.0) > 0.15:
        print(f"(warning: bins sum {total_pct:.2f}%)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--show-zero",
        action="store_true",
        help="Print bins with 0%% (default: hide bins < 0.05%%).",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Write long-format CSV (sensor, range, pct, count).",
    )
    args = p.parse_args()

    data_dir = REPO_ROOT / "data_30mins_frequency"
    long_rows: list[dict] = []

    for path in sorted(data_dir.glob("*_30_min.csv")):
        stem = path.stem.replace("_30_min", "").replace("AHU_2_9_", "")
        cov = train_covered(path)
        print_sensor(stem, cov, show_zero=args.show_zero)
        for lab, pct, cnt in bin_table(cov):
            long_rows.append(
                {"sensor": stem, "range": lab, "pct": pct, "count": cnt}
            )

    if args.csv:
        pd.DataFrame(long_rows).to_csv(args.csv, index=False)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
