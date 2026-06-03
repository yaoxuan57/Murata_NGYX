#!/usr/bin/env python3
"""RMS distribution in training windows (matches run_transformer_tuning.sbatch defaults)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from forecast_sweep_common import (  # noqa: E402
    compute_timestep_window_start_indices,
    parse_timestamp_series,
    row_indices_covered_by_windows,
    smooth_target_series_1d,
)

VALUE_COLUMN = "Acceleration RMS"


DEFAULT_EDGES = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 8.0, 100.0]


def histogram_pct(arr: np.ndarray, edges: list[float]) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return np.zeros(len(edges) - 1), edges
    hist, _ = np.histogram(arr, bins=edges)
    pct = 100.0 * hist / hist.sum()
    return pct, hist


def pct_table(arr: np.ndarray, title: str, edges: list[float]) -> None:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        print(f"\n{title}: (no data)")
        return
    labels = []
    for a, b in zip(edges[:-1], edges[1:]):
        labels.append(f">={a:g}" if b >= 100 else f"{a:g}-{b:g}")
    hist, _ = np.histogram(arr, bins=edges)
    pct = 100.0 * hist / hist.sum()
    print(f"\n{title} (n={arr.size:,})")
    print(
        f"  min={arr.min():.4f} max={arr.max():.4f} "
        f"mean={arr.mean():.4f} median={np.median(arr):.4f}"
    )
    for lab, p, c in zip(labels, pct, hist):
        if c > 0:
            print(f"  {lab:>10}: {p:5.2f}%  ({int(c):,})")


def analyze_split(
    name: str,
    part: pd.DataFrame,
    *,
    input_len: int,
    pred_len: int,
    smooth_window: int,
    wk: dict,
) -> dict:
    span = input_len + pred_len
    raw = pd.to_numeric(part[VALUE_COLUMN], errors="coerce").to_numpy(dtype=np.float64)
    smooth = smooth_target_series_1d(raw.astype(np.float32), smooth_window)
    starts = compute_timestep_window_start_indices(part["TIMESTAMP"], **wk)
    n_sliding = max(0, len(smooth) - span + 1)
    row_idx = row_indices_covered_by_windows(starts, span, len(smooth))
    covered_smooth = smooth[row_idx]
    chunks = [smooth[int(i) : int(i) + span] for i in starts]
    all_pts = np.concatenate(chunks) if chunks else np.array([], dtype=np.float64)
    return {
        "rows": len(part),
        "n_windows": len(starts),
        "n_sliding": n_sliding,
        "pct_windows_kept": 100.0 * len(starts) / max(n_sliding, 1),
        "covered_unique": covered_smooth,
        "all_window_pts": all_pts,
    }


def run_one(
    input_path: Path,
    *,
    train_ratio: float,
    val_ratio: float,
    input_len: int,
    pred_len: int,
    smooth_window: int,
    uniform_step_seconds: float,
    uniform_tol_seconds: float,
    max_gap_seconds: float,
    no_require_uniform: bool,
    verbose: bool,
    edges: list[float],
) -> dict:
    df = pd.read_csv(input_path)
    df["TIMESTAMP"] = parse_timestamp_series(df["TIMESTAMP"], str(input_path))
    df = df.sort_values("TIMESTAMP").reset_index(drop=True)
    n = len(df)
    n_tr = int(np.floor(n * train_ratio))
    n_va = int(np.floor(n * val_ratio))
    n_te = n - n_tr - n_va

    splits = {
        "train": df.iloc[:n_tr].copy(),
        "val": df.iloc[n_tr : n_tr + n_va].copy(),
        "test": df.iloc[n_tr + n_va :].copy(),
    }

    span = input_len + pred_len
    wk = {
        "span_len": span,
        "nominal_seconds": None if no_require_uniform else uniform_step_seconds,
        "tolerance_seconds": uniform_tol_seconds,
        "max_consecutive_gap_seconds": max_gap_seconds,
    }

    if verbose:
        print(f"Source: {input_path}")
        print(f"Rows: total={n:,} train={n_tr:,} val={n_va:,} test={n_te:,}")
        uni = (
            "uniform OFF"
            if no_require_uniform
            else f"uniform {uniform_step_seconds:g}s +/-{uniform_tol_seconds:g}s"
        )
        print(
            f"Window: INPUT_LEN={input_len} PRED_LEN={pred_len} span={span} | "
            f"causal smooth w={smooth_window} | {uni} | max_gap<={max_gap_seconds}s"
        )

    out: dict = {"sensor": input_path.stem.replace("_30_min", ""), "path": str(input_path)}
    for name, part in splits.items():
        r = analyze_split(
            name,
            part,
            input_len=input_len,
            pred_len=pred_len,
            smooth_window=smooth_window,
            wk=wk,
        )
        cov = r["covered_unique"]
        pct, _ = histogram_pct(cov, edges)
        out[f"{name}_rows"] = r["rows"]
        out[f"{name}_windows"] = r["n_windows"]
        out[f"{name}_pct_kept"] = r["pct_windows_kept"]
        if cov.size:
            out[f"{name}_median_rms"] = float(np.median(cov))
            out[f"{name}_mean_rms"] = float(cov.mean())
        else:
            out[f"{name}_median_rms"] = float("nan")
            out[f"{name}_mean_rms"] = float("nan")
        for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            key = f"{name}_pct_{a:g}_{b:g}"
            out[key] = float(pct[i]) if cov.size else 0.0

        if verbose:
            print(
                f"\n[{name}] rows={r['rows']:,}  valid_windows={r['n_windows']:,}  "
                f"({r['pct_windows_kept']:.1f}% of {r['n_sliding']:,} sliding starts)"
            )
            pct_table(
                cov,
                f"{name}: UNIQUE rows in >=1 window (smoothed RMS — model target)",
                edges,
            )
            pct_table(
                r["all_window_pts"],
                f"{name}: ALL points in all windows (smoothed, counts overlap)",
                edges,
            )
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "data" / "AHU_2_9_Blower_DE_A.csv",
    )
    p.add_argument(
        "--all-30min",
        action="store_true",
        help="Run Option B on every data_30mins_frequency/*_30_min.csv (sbatch defaults).",
    )
    p.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="With --all-30min, write one row per sensor (default: data_30mins_frequency/_window_rms_summary.csv).",
    )
    p.add_argument("--train-ratio", type=float, default=0.6)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--input-len", type=int, default=48)
    p.add_argument("--pred-len", type=int, default=48)
    p.add_argument("--smooth-window", type=int, default=48)
    p.add_argument("--uniform-step-seconds", type=float, default=1800.0)
    p.add_argument("--uniform-tol-seconds", type=float, default=60.0)
    p.add_argument("--max-gap-seconds", type=float, default=1860.0)
    p.add_argument(
        "--no-require-uniform",
        action="store_true",
        help="Only apply max-gap filter (like REQUIRE_UNIFORM_TIMESTEP=0).",
    )
    args = p.parse_args()
    edges = DEFAULT_EDGES

    if args.all_30min:
        data_dir = REPO_ROOT / "data_30mins_frequency"
        paths = sorted(data_dir.glob("*_30_min.csv"))
        if not paths:
            raise SystemExit(f"No *_30_min.csv under {data_dir}")
        rows = []
        for path in paths:
            rows.append(
                run_one(
                    path,
                    train_ratio=args.train_ratio,
                    val_ratio=args.val_ratio,
                    input_len=args.input_len,
                    pred_len=args.pred_len,
                    smooth_window=args.smooth_window,
                    uniform_step_seconds=args.uniform_step_seconds,
                    uniform_tol_seconds=args.uniform_tol_seconds,
                    max_gap_seconds=args.max_gap_seconds,
                    no_require_uniform=args.no_require_uniform,
                    verbose=False,
                    edges=edges,
                )
            )
        summary = pd.DataFrame(rows)
        out_csv = args.summary_csv or (data_dir / "_window_rms_summary.csv")
        summary.to_csv(out_csv, index=False)
        print(
            "Option B: 30 min CSV + uniform 1800±60s, max_gap 1860s, smooth 48, windows 48+48\n"
        )
        cols = [
            "sensor",
            "train_rows",
            "train_windows",
            "train_pct_kept",
            "train_median_rms",
            "train_pct_0_0.5",
            "train_pct_0.5_1",
            "train_pct_1_1.5",
            "train_pct_1.5_2",
            "train_pct_2_2.5",
            "val_windows",
            "test_windows",
        ]
        cols = [c for c in cols if c in summary.columns]
        print(summary[cols].to_string(index=False))
        print(f"\nWrote {out_csv}")
        return

    run_one(
        args.input,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        input_len=args.input_len,
        pred_len=args.pred_len,
        smooth_window=args.smooth_window,
        uniform_step_seconds=args.uniform_step_seconds,
        uniform_tol_seconds=args.uniform_tol_seconds,
        max_gap_seconds=args.max_gap_seconds,
        no_require_uniform=args.no_require_uniform,
        verbose=True,
        edges=edges,
    )


if __name__ == "__main__":
    main()
