#!/usr/bin/env python3
"""
Run inference for AHU 2-9 Blower DE A only (single model, single sensor).

  cd deployment_blower_de_a
  copy your trained .pth to models/AHU_2_9_Blower_DE_A.pth
  python run_inference.py --plot
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from _bootstrap import DEPLOY_ROOT

from config import (
    CHECKPOINT_STEM,
    DEFAULT_CHECKPOINT,
    DEFAULT_INPUT_CSV,
    DEFAULT_OUTPUT_JSON,
    DEFAULT_PLOT_PATH,
    SENSOR_DESC,
)
from inference import run_inference, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Inference for {SENSOR_DESC} (DLinear or transformer .pth).",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help=f"Multi-sensor vibration CSV (default: {DEFAULT_INPUT_CSV})",
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"Trained .pth (default: models/{CHECKPOINT_STEM}.pth)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help=f"Predictions JSON (default: {DEFAULT_OUTPUT_JSON})",
    )
    parser.add_argument("--smooth-window", type=int, default=48)
    parser.add_argument("--max-gap-minutes", type=float, default=600.0)
    parser.add_argument("--forecast-step-minutes", type=float, default=30.0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=DEFAULT_PLOT_PATH,
    )
    parser.add_argument("--plot-history-before", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.is_file():
        raise SystemExit(f"Input CSV not found: {args.input}")

    device = args.device
    if device == "cuda" and not __import__("torch").cuda.is_available():
        print("CUDA not available; using CPU.", file=sys.stderr)
        device = "cpu"

    payload = run_inference(
        args.input,
        args.checkpoint,
        smooth_window=args.smooth_window,
        max_gap_seconds=float(args.max_gap_minutes) * 60.0,
        forecast_step_minutes=float(args.forecast_step_minutes),
        device=device,
    )

    write_json(payload, args.output)

    if payload.get("success") is False:
        print(f"FAILED: {payload.get('error')}")
        raise SystemExit(1)

    print(f"OK: {SENSOR_DESC}")
    print(f"  model:   {payload.get('model_type')}  checkpoint: {payload.get('checkpoint')}")
    print(f"  context: {payload.get('input_len')} rows  forecast: {payload.get('pred_len')} steps")
    if payload.get("forecast_quantiles"):
        print(f"  quantiles: {payload['forecast_quantiles']}")
    print(f"Wrote {args.output}")

    if args.plot:
        try:
            from plot_predictions import plot_forecast
        except ImportError as exc:
            raise SystemExit("Plotting requires matplotlib: pip install matplotlib") from exc
        plot_forecast(
            payload,
            args.plot_path,
            input_csv=args.input,
            history_before=args.plot_history_before,
            smooth_window=args.smooth_window,
        )
        print(f"Plot -> {args.plot_path}")


if __name__ == "__main__":
    main()
