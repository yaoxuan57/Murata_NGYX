#!/usr/bin/env python3
"""Rename legacy .pth + sidecar JSON bundles to deployment batch-upload naming."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from model_meta import deployment_metadata_filename, deployment_model_filename  # noqa: E402


def rename_models_in_dir(models_dir: Path) -> int:
    renamed = 0
    for json_path in sorted(models_dir.glob("*.json")):
        if json_path.name.endswith(".metadata.json"):
            continue
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        model_type = payload.get("modelType", "rms_forecast")
        sensor_id = payload.get("sensorId", "")
        sensor_name = payload.get("sensorName", "")
        if not sensor_id or not sensor_name:
            print(f"SKIP (no sensor info): {json_path.name}")
            continue

        pth_old = models_dir / str(
            payload.get("checkpointFile") or json_path.with_suffix(".pth").name
        )
        if not pth_old.is_file():
            pth_old = json_path.with_suffix(".pth")
        if not pth_old.is_file():
            print(f"SKIP (no pth): {json_path.name}")
            continue

        new_pth_name = deployment_model_filename(model_type, sensor_id, sensor_name)
        new_meta_name = deployment_metadata_filename(model_type, sensor_id, sensor_name)
        new_pth = models_dir / new_pth_name
        new_meta = models_dir / new_meta_name

        payload["checkpointFile"] = new_pth_name
        if new_pth.resolve() != pth_old.resolve():
            if new_pth.exists():
                new_pth.unlink()
            pth_old.rename(new_pth)
        new_meta.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        if json_path.resolve() != new_meta.resolve():
            json_path.unlink()
        print(f"{pth_old.name} -> {new_pth_name}")
        print(f"{json_path.name} -> {new_meta_name}")
        renamed += 1
    return renamed


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rename model bundles to modelType__sensorID__sensorName.pth + "
            ".metadata.json for batch upload."
        ),
    )
    parser.add_argument(
        "models_dir",
        type=Path,
        nargs="?",
        default=_SCRIPT_DIR / "models",
        help="Directory containing legacy or mixed model bundles (default: ./models).",
    )
    args = parser.parse_args()
    if not args.models_dir.is_dir():
        raise SystemExit(f"Models directory not found: {args.models_dir}")
    count = rename_models_in_dir(args.models_dir.resolve())
    print(f"Renamed {count} bundle(s) in {args.models_dir}")


if __name__ == "__main__":
    main()
