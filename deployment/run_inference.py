#!/usr/bin/env python3
"""
Run transformer inference from a multi-sensor vibration CSV.

One sensor (--sensor) or all default AHU 2-9 sensors (--all-sensors).
Writes a single JSON file with one top-level key per sensor.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deployment.inference import (  # noqa: E402
    run_inference_all_sensors,
    run_inference_payload,
    write_predictions_json,
)
from deployment.sensors import AHU_2_9_SENSOR_DESCS  # noqa: E402


def parse_args() -> argparse.Namespace:
    default_input = (
        REPO_ROOT / "deployment" / "data" / "Vibration sensors _ 2022 to 2026.csv"
    )
    default_output = REPO_ROOT / "deployment" / "output" / "predictions.json"
    default_models = REPO_ROOT / "deployment" / "models"

    parser = argparse.ArgumentParser(
        description=(
            "Filter SENSOR_DESC from a multi-sensor CSV, take 288 latest rows, "
            "validate <=60 min gaps, smooth RMS (200), predict 288 steps."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "All sensors (one JSON, 8 keys; per-sensor .pth in a folder):\n"
            "  python deployment/run_inference.py --all-sensors -c deployment/models\n\n"
            "Routing: deployment/models/<stem>.pth where stem is SENSOR_DESC with\n"
            "  spaces to underscores and 2-9 to 2_9 (e.g. AHU_2_9_Blower_DE_A.pth).\n\n"
            "Single sensor:\n"
            '  python deployment/run_inference.py -s "AHU 2-9 Blower DE A" -c path/to/model.pth\n\n'
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
            "With --all-sensors, use a directory: deployment/models/AHU_2_9_<name>.pth."
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
    parser.add_argument("--smooth-window", type=int, default=200)
    parser.add_argument("--max-gap-minutes", type=float, default=60.0)
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
            "For per-sensor models, pass deployment/models/ as a directory.",
            file=sys.stderr,
        )

    device = args.device
    if device == "cuda" and not __import__("torch").cuda.is_available():
        print("CUDA not available; using CPU.")
        device = "cpu"

    kw = dict(
        smooth_window=args.smooth_window,
        max_gap_seconds=float(args.max_gap_minutes) * 60.0,
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

    if fail and (args.fail_if_any_sensor_fails or (not args.all_sensors)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
