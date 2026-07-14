#!/usr/bin/env python3
"""Plot input window + forecast vs actual from predictions JSON and/or rolling CSV windows."""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from _bootstrap import DEPLOY_ROOT

from plot_predictions import (
    plot_all_sensors_forecast_vs_actual,
    plot_rolling_windows_from_csv,
)


def main() -> None:
    default_models = DEPLOY_ROOT.parent / "deployment" / "models"
    parser = argparse.ArgumentParser(
        description=(
            "Plot model input window, predicted forecast, and actual values. "
            "Latest JSON forecasts go to output/plots/latest/; optional rolling "
            "backtest plots go to output/plots/rolling/<sensor>/."
        ),
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=DEPLOY_ROOT / "output" / "predictions_June.json",
        help="Predictions JSON from run_inference.py (latest-window plots)",
    )
    parser.add_argument(
        "-o",
        "--plots-dir",
        type=Path,
        default=DEPLOY_ROOT / "output" / "plots",
        help="Root directory for plot outputs",
    )
    parser.add_argument(
        "--plot-format",
        type=str,
        default="html",
        choices=("html", "png"),
        help="Interactive HTML (zoom/pan) or static PNG (default: html).",
    )
    parser.add_argument(
        "--data-csv",
        type=Path,
        default=DEPLOY_ROOT / "data" / "Jun.csv",
        help="CSV with vibration history (actuals + rolling inference).",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=default_models,
        help="Directory of .pth checkpoints (for rolling-window plots).",
    )
    parser.add_argument("--smooth-window", type=int, default=48)
    parser.add_argument(
        "--history-before",
        type=int,
        default=0,
        help="Extra smoothed points shown before the 48-pt input window (0 = input only).",
    )
    parser.add_argument(
        "--rolling-windows",
        type=int,
        default=5,
        help="Rolling backtest plots per sensor (0=off). Evenly spaced across the CSV.",
    )
    parser.add_argument(
        "--skip-latest-json",
        action="store_true",
        help="Only generate rolling plots, not plots from the JSON file.",
    )
    parser.add_argument(
        "--match-tolerance-minutes",
        type=float,
        default=20.0,
        help="Unused for rolling; kept for latest JSON plot alignment.",
    )
    parser.add_argument("--device", type=str, default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()

    if not args.data_csv.is_file():
        raise SystemExit(f"Data CSV not found: {args.data_csv}")

    total_ok = total_skip = 0
    all_reasons: list[str] = []

    print(f"Data CSV:    {args.data_csv}")
    print(f"Plots root:  {args.plots_dir}")
    print(f"History before input: {args.history_before} points")

    if not args.skip_latest_json:
        if not args.input.is_file():
            raise SystemExit(f"Predictions JSON not found: {args.input}")
        print(f"\n=== Latest forecast (from JSON) -> {args.plots_dir / 'latest'} ===")
        print(f"Predictions: {args.input}")
        n_ok, n_skip, reasons = plot_all_sensors_forecast_vs_actual(
            args.input,
            args.plots_dir,
            input_csv=args.data_csv,
            smooth_window=args.smooth_window,
            history_before=args.history_before,
            tolerance_minutes=args.match_tolerance_minutes,
            plot_format=args.plot_format,
        )
        total_ok += n_ok
        total_skip += n_skip
        all_reasons.extend(reasons)
        print(f"Latest: plotted {n_ok}, skipped {n_skip}")

    if args.rolling_windows > 0:
        if not args.models_dir.is_dir():
            raise SystemExit(f"Models directory not found: {args.models_dir}")
        print(
            f"\n=== Rolling windows ({args.rolling_windows} per sensor) "
            f"-> {args.plots_dir / 'rolling'} ==="
        )
        n_ok, n_skip, reasons = plot_rolling_windows_from_csv(
            args.data_csv,
            args.models_dir,
            args.plots_dir,
            rolling_windows=args.rolling_windows,
            history_before=args.history_before,
            smooth_window=args.smooth_window,
            device=args.device,
            plot_format=args.plot_format,
        )
        total_ok += n_ok
        total_skip += n_skip
        all_reasons.extend(reasons)
        print(f"Rolling: plotted {n_ok}, skipped {n_skip}")

    print(f"\nDone: total plotted {total_ok}, skipped {total_skip}")
    for r in all_reasons[:30]:
        print(f"  - {r}")
    if len(all_reasons) > 30:
        print(f"  ... and {len(all_reasons) - 30} more")
    if total_ok == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
