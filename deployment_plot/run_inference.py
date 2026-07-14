#!/usr/bin/env python3
"""
Run vibration forecasts: rolling 48-point input windows -> 48-step forecasts.

For each sensor and window:
  - validate input (point count, interval gaps, training range)
  - run the transformer forecast
  - write JSON results
  - plot input + forecast band + actual holdout

  cd deployment_plot
  python run_inference.py

  python run_inference.py -i data/Jun.csv -o output/predictions_June.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from _bootstrap import DEPLOY_ROOT

from inference import (  # noqa: E402
    forecast_rolling_windows,
    write_predictions_json,
)
from io_utils import read_vibration_export_csv  # noqa: E402
from plot_predictions import plot_rolling_inference_results  # noqa: E402
from sensors import build_vibration_sensor_model_map  # noqa: E402


def parse_args() -> argparse.Namespace:
    default_input = DEPLOY_ROOT / "data" / "Jun.csv"
    default_output = DEPLOY_ROOT / "output" / "predictions_June.json"
    default_models = DEPLOY_ROOT.parent / "deployment" / "models"
    default_plots = DEPLOY_ROOT / "output" / "plots"

    parser = argparse.ArgumentParser(
        description=(
            "Load vibration CSV, run rolling 48->48 forecasts per sensor with "
            "validation warnings, write JSON, and plot actual vs predicted."
        ),
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=default_input,
        help=f"Vibration CSV (default: {default_input})",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=default_models,
        help=f"Directory of .pth checkpoints (default: {default_models})",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=default_output,
        help=f"Predictions JSON path (default: {default_output})",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=default_plots,
        help=f"Plot output root (default: {default_plots})",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating plots.",
    )
    parser.add_argument(
        "--plot-format",
        type=str,
        default="html",
        choices=("html", "png"),
        help="Interactive HTML (zoom/pan) or static PNG (default: html).",
    )
    parser.add_argument(
        "--windows-per-sensor",
        type=int,
        default=5,
        help=(
            "Rolling windows per sensor (evenly spaced). "
            "Use 0 for a single latest-window forecast only."
        ),
    )
    parser.add_argument(
        "--history-before",
        type=int,
        default=0,
        help="Extra smoothed history points shown before the 48-pt input on plots (0 = input window only).",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=48,
        help="Causal (trailing) MA window on Acceleration RMS (match training TARGET_SMOOTHING_WINDOW).",
    )
    parser.add_argument(
        "--max-interval-warning-hours",
        type=float,
        default=2.0,
        help="Warn (but still predict) when any consecutive gap in the 48-point window exceeds this many hours.",
    )
    parser.add_argument(
        "--forecast-step-minutes",
        type=float,
        default=30.0,
        help="Minutes between each forecast timestamp in JSON output (30-min cadence).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=("cpu", "cuda"),
    )
    parser.add_argument(
        "--fail-if-any-sensor-fails",
        action="store_true",
        help="Exit code 1 if any sensor has zero successful windows.",
    )
    return parser.parse_args()


def _print_window_warnings(sensor_id: str, sensor_name: str, body: dict) -> None:
    wi = body.get("window_index", "?")
    if body.get("success") is False:
        msg = body.get("warning") or body.get("error", "failed")
        n = body.get("n_points")
        req = body.get("required_points")
        extra = f" ({n}/{req} points)" if n is not None and req is not None else ""
        print(f"    window {wi}: SKIPPED — {msg}{extra}")
        return

    print(f"    window {wi}: OK")
    ic = body.get("interval_check")
    if ic:
        print(f"      WARNING: {ic['warning']}")
    rc = body.get("range_check")
    if rc and not rc.get("in_training_range"):
        print(f"      WARNING: {rc['warning']}")


def main() -> None:
    args = parse_args()

    if not args.input.is_file():
        raise SystemExit(f"Input CSV not found: {args.input}")
    if not args.models_dir.is_dir():
        raise SystemExit(f"Models directory not found: {args.models_dir}")

    device = args.device
    if device == "cuda" and not __import__("torch").cuda.is_available():
        print("CUDA not available; using CPU.")
        device = "cpu"

    vib_df = read_vibration_export_csv(str(args.input))
    vibration_sensor_model_map = build_vibration_sensor_model_map(args.models_dir)

    print(
        f"Running rolling inference: {args.windows_per_sensor} window(s) per sensor "
        f"({48}-pt input -> {48}-pt forecast)"
    )
    vib_results = forecast_rolling_windows(
        vib_df,
        sensor_model_map=vibration_sensor_model_map,
        rolling_windows=args.windows_per_sensor,
        smooth_window=args.smooth_window,
        max_interval_warning_hours=float(args.max_interval_warning_hours),
        forecast_step_minutes=float(args.forecast_step_minutes),
        device=device,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_predictions_json(vib_results, str(args.output))

    total_windows = 0
    total_ok = 0
    sensors_failed = 0

    print(f"\nWrote {len(vib_results)} sensor(s) -> {args.output}")
    for sensor_id, block in vib_results.items():
        sensor_name = block.get("sensorName", "?")
        windows = block.get("windows")

        if windows is None:
            sensors_failed += 1
            msg = block.get("warning") or block.get("error", "failed")
            n = block.get("n_points")
            req = block.get("required_points")
            extra = f" ({n}/{req} points)" if n is not None and req is not None else ""
            print(f"  - {sensor_id} ({sensor_name}): FAILED — {msg}{extra}")
            continue

        ok = int(block.get("n_succeeded", 0))
        n_win = len(windows)
        total_windows += n_win
        total_ok += ok
        if ok == 0:
            sensors_failed += 1
        print(f"  - {sensor_id} ({sensor_name}): {ok}/{n_win} window(s) succeeded")
        for w in windows:
            _print_window_warnings(sensor_id, sensor_name, w)

    print(f"\nSummary: {total_ok}/{total_windows} windows succeeded across {len(vib_results)} sensors")

    if not args.no_plot:
        args.plots_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Plots -> {args.plots_dir / 'rolling'} ===")
        n_plot, n_skip, reasons = plot_rolling_inference_results(
            vib_results,
            args.plots_dir,
            input_csv=args.input,
            smooth_window=args.smooth_window,
            history_before=args.history_before,
            plot_format=args.plot_format,
        )
        print(f"Plotted {n_plot} window(s); skipped {n_skip}.")
        for r in reasons:
            print(f"  - {r}")

    if sensors_failed and args.fail_if_any_sensor_fails:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
