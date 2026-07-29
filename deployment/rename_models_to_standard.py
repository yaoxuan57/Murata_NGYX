#!/usr/bin/env python3
"""One-off: rename legacy deployment/models bundles to standard naming."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from model_meta import deployment_metadata_filename, deployment_model_filename  # noqa: E402


def main() -> None:
    models_dir = Path(__file__).resolve().parent / "models"
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


if __name__ == "__main__":
    main()
