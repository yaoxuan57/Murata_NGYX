#!/usr/bin/env python3
"""
Run transformer inference from a multi-sensor vibration CSV.

One sensor (--sensor) or all default AHU 2-9 sensors (--all-sensors).
Writes a single JSON file with one top-level key per sensor.

Standalone: run from this folder (no repo root required):
  cd deployment
  python run_inference.py --all-sensors -c models
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from _bootstrap import DEPLOY_ROOT

from inference import (  # noqa: E402
    run_inference_all_sensors,
    run_inference_payload,
    write_predictions_json,
)
from sensors import AHU_2_9_SENSOR_DESCS  # noqa: E402


def parse_args() -> argparse.Namespace:
    default_input = DEPLOY_ROOT / "data" / "vibration_May.csv"
    default_output = DEPLOY_ROOT / "output" / "predictions.json"
    default_models = DEPLOY_ROOT / "models"

    parser = argparse.ArgumentParser(
        description=(
            "Filter SENSOR_DESC from a multi-sensor CSV, take input_len latest rows "
            "(48 for current models), validate <=10 h gaps, causal smooth RMS (48), "
            "predict pred_len steps (48)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "All sensors (one JSON, 8 keys; per-sensor .pth in a folder):\n"
            "  python run_inference.py --all-sensors -c models\n\n"
            "Routing: models/<stem>.pth where stem is SENSOR_DESC with\n"
            "  spaces to underscores and 2-9 to 2_9 (e.g. AHU_2_9_Blower_DE_A.pth).\n\n"
            "Single sensor:\n"
            '  python run_inference.py -s "AHU 2-9 Blower DE A" -c path/to/model.pth\n\n'
            "Sensors in --all-sensors mode:\n  "
            + "\n  ".join(AHU_2_9_SENSOR_DESCS)
        ),
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=default_input,
        help=f"Multi-sensor CSV (default: {default_input})",
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=Path,
        default=default_models,
        help=(
            f"Path to one .pth file (shared by all sensors) or a directory of per-sensor "
            f".pth files (default: {default_models}). "
            "With --all-sensors, use a directory: models/AHU_2_9_<name>.pth."
        ),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--sensor",
        "-s",
        type=str,
        help='Single SENSOR_DESC, e.g. "AHU 2-9 Blower DE A".',
    )
    mode.add_argument(
        "--all-sensors",
        action="store_true",
        help=f"Run all {len(AHU_2_9_SENSOR_DESCS)} default AHU 2-9 sensors into one JSON.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=default_output,
        help=f"Predictions JSON path (default: {default_output})",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=48,
        help="Causal (trailing) MA window on Acceleration RMS (match training TARGET_SMOOTHING_WINDOW).",
    )
    parser.add_argument(
        "--max-gap-minutes",
        type=float,
        default=600.0,
        help="Max allowed gap between consecutive context timestamps in minutes (default 600 = 10 h).",
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
        help="Exit code 1 if any sensor failed (default: still write JSON with per-sensor errors).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="After writing JSON, save per-sensor forecast PNGs (median + quantile band) under output/plots/.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help="Plot output directory (default: <output>/plots).",
    )
    parser.add_argument(
        "--plot-history-before",
        type=int,
        default=100,
        help="When using --plot: extra CSV points shown before the model input window (0=off).",
    )
    return parser.parse_args()


def _summarize_payload(payload: dict) -> tuple[int, int]:
    ok = fail = 0
    for body in payload.values():
        if body.get("success") is False:
            fail += 1
        else:
            ok += 1
    return ok, fail


def main() -> None:
    args = parse_args()

    if not args.input.is_file():
        raise SystemExit(f"Input CSV not found: {args.input}")
    if not args.checkpoint.exists():
        raise SystemExit(f"Checkpoint path not found: {args.checkpoint}")
    if args.all_sensors and args.checkpoint.is_file():
        print(
            "Note: --checkpoint is a single file; all sensors will share that model. "
            "For per-sensor models, pass models/ as a directory.",
            file=sys.stderr,
        )

    device = args.device
    if device == "cuda" and not __import__("torch").cuda.is_available():
        print("CUDA not available; using CPU.")
        device = "cpu"

    kw = dict(
        smooth_window=args.smooth_window,
        max_gap_seconds=float(args.max_gap_minutes) * 60.0,
        forecast_step_minutes=float(args.forecast_step_minutes),
        device=device,
    )

    if args.all_sensors:
        payload = run_inference_all_sensors(
            str(args.input),
            str(args.checkpoint),
            **kw,
        )
    else:
        payload = run_inference_payload(
            str(args.input),
            str(args.checkpoint),
            args.sensor,
            **kw,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_predictions_json(payload, str(args.output))

    ok, fail = _summarize_payload(payload)
    print(f"\nWrote {len(payload)} sensor(s) -> {args.output}")
    print(f"  succeeded: {ok}  failed: {fail}")

    if args.all_sensors:
        for name, body in payload.items():
            if body.get("success") is False:
                print(f"  - {name}: {body['error']}")
            else:
                q = body.get("forecast_quantiles")
                extra = f" quantiles={q}" if q else ""
                print(f"  - {name}: OK{extra}")

    if args.plot:
        try:
            from plot_predictions import plot_all_sensors_from_json, plot_combined_overview
        except ImportError as exc:
            raise SystemExit(
                "Plotting requires matplotlib: pip install matplotlib"
            ) from exc
        plots_dir = args.plots_dir or (args.output.parent / "plots")
        print(f"\nPlots -> {plots_dir}")
        n_ok, n_skip, reasons = plot_all_sensors_from_json(
            args.output,
            plots_dir,
            input_csv=args.input if args.input.is_file() else None,
            models_dir=args.checkpoint if args.checkpoint.is_dir() else None,
            smooth_window=args.smooth_window,
            max_gap_seconds=float(args.max_gap_minutes) * 60.0,
            history_before=args.plot_history_before,
        )
        plot_combined_overview(args.output, plots_dir / "_all_sensors_overview.png")
        print(f"  plotted: {n_ok}  skipped: {n_skip}")
        for r in reasons:
            print(f"    - {r}")

    if fail and (args.fail_if_any_sensor_fails or (not args.all_sensors)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
