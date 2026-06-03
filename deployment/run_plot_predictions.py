#!/usr/bin/env python3
"""Plot forecasts from an existing predictions.json (no re-inference)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from _bootstrap import DEPLOY_ROOT

from plot_predictions import plot_all_sensors_from_json, plot_combined_overview


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot median + quantiles from predictions JSON.")
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=DEPLOY_ROOT / "output" / "predictions.json",
        help="Predictions JSON from run_inference.py",
    )
    parser.add_argument(
        "-o",
        "--plots-dir",
        type=Path,
        default=DEPLOY_ROOT / "output" / "plots",
        help="Directory for PNG outputs",
    )
    parser.add_argument(
        "--data-csv",
        type=Path,
        default=DEPLOY_ROOT / "data" / "vibration_May.csv",
        help="Same CSV used for run_inference.py (history only; input/forecast from JSON).",
    )
    parser.add_argument(
        "-c",
        "--models-dir",
        type=Path,
        default=DEPLOY_ROOT / "models",
        help="Model directory for context rebuild (per-sensor .pth).",
    )
    parser.add_argument("--smooth-window", type=int, default=48)
    parser.add_argument("--max-gap-minutes", type=float, default=600.0)
    parser.add_argument(
        "--history-before",
        type=int,
        default=100,
        help="Extra points to plot before the 48-step model input (from CSV; 0=off).",
    )
    args = parser.parse_args()
    if not args.input.is_file():
        raise SystemExit(f"Not found: {args.input}")

    data_csv = args.data_csv if args.data_csv.is_file() else None
    print(f"Reading {args.input}")
    n_ok, n_skip, reasons = plot_all_sensors_from_json(
        args.input,
        args.plots_dir,
        input_csv=data_csv,
        models_dir=args.models_dir if args.models_dir.is_dir() else None,
        smooth_window=args.smooth_window,
        max_gap_seconds=float(args.max_gap_minutes) * 60.0,
        history_before=args.history_before,
    )
    plot_combined_overview(args.input, args.plots_dir / "_all_sensors_overview.png")
    print(f"Done: plotted {n_ok}, skipped {n_skip}")
    for r in reasons:
        print(f"  - {r}")
    if n_ok == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
