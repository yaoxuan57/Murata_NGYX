"""Target SENSOR_DESC values for AHU 2-9 deployment."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Union

# Canonical names as they appear in the source CSV (``SENSOR_DESC`` column).
AHU_2_9_SENSOR_DESCS: List[str] = [
    "AHU 2-9 Blower DE A",
    "AHU 2-9 Blower DE V",
    "AHU 2-9 Blower DE Vibration X",
    "AHU 2-9 Blower NDE A",
    "AHU 2-9 Blower NDE H",
    "AHU 2-9 Blower NDE V",
    "AHU 2-9 motor DE H",
    "AHU 2-9 motor NDE H",
]

# ``SENSOR_CODE`` values from vibration exports (hex strings; NDE A/V may appear as 9100/910 in CSV).
AHU_2_9_SENSOR_IDS: Dict[str, str] = {
    "AHU 2-9 Blower DE A": "91B8",
    "AHU 2-9 Blower DE V": "916E",
    "AHU 2-9 Blower DE Vibration X": "8894",
    "AHU 2-9 Blower NDE A": "91E2",
    "AHU 2-9 Blower NDE H": "91ED",
    "AHU 2-9 Blower NDE V": "91E1",
    "AHU 2-9 motor DE H": "91F0",
    "AHU 2-9 motor NDE H": "91AC",
}

# (sensor_desc, checkpoint filename under deployment/models/, model version tag)
DEPLOYMENT_TRANSFORMER_MODELS: List[tuple[str, str, str]] = [
    ("AHU 2-9 Blower DE A", "AHU_2_9_Blower_DE_A_v3.pth", "v3"),
    ("AHU 2-9 Blower DE V", "AHU_2_9_Blower_DE_V_v2.pth", "v2"),
    ("AHU 2-9 Blower DE Vibration X", "AHU_2_9_Blower_DE_Vibration_X_v2.pth", "v2"),
    ("AHU 2-9 Blower NDE A", "AHU_2_9_Blower_NDE_A_v2.pth", "v2"),
    ("AHU 2-9 Blower NDE H", "AHU_2_9_Blower_NDE_H_v2.pth", "v2"),
    ("AHU 2-9 Blower NDE V", "AHU_2_9_Blower_NDE_V_v2.pth", "v2"),
    ("AHU 2-9 motor DE H", "AHU_2_9_motor_DE_H_v2.pth", "v2"),
    ("AHU 2-9 motor NDE H", "AHU_2_9_motor_NDE_H_v2.pth", "v2"),
]

# Alternate spellings (underscores / spacing) → canonical ``SENSOR_DESC``.
SENSOR_DESC_ALIASES: Dict[str, str] = {
    "AHU 2-9_Blower DE A": "AHU 2-9 Blower DE A",
    "AHU_2-9 Blower DE A": "AHU 2-9 Blower DE A",
    "AHU_2-9 Blower DE V": "AHU 2-9 Blower DE V",
    "AHU_2-9 Blower DE Vibration X": "AHU 2-9 Blower DE Vibration X",
    "AHU_2-9 Blower NDE A": "AHU 2-9 Blower NDE A",
    "AHU_2-9 Blower NDE H": "AHU 2-9 Blower NDE H",
    "AHU_2-9 Blower NDE V": "AHU 2-9 Blower NDE V",
    "AHU_2-9 motor DE H": "AHU 2-9 motor DE H",
    "AHU_2-9 motor NDE H": "AHU 2-9 motor NDE H",
}


def normalize_sensor_desc(value: str) -> str:
    text = str(value).strip()
    return SENSOR_DESC_ALIASES.get(text, text)


def sensor_desc_to_slug(sensor_desc: str) -> str:
    """Filesystem-safe stem from SENSOR_DESC (spaces → underscores)."""
    slug = sensor_desc.strip()
    slug = slug.replace(" ", "_")
    slug = re.sub(r"[^\w\-]+", "_", slug)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug


def checkpoint_stem_for_sensor(sensor_desc: str) -> str:
    """
    ``.pth`` filename stem under ``deployment/models/``.

    On disk, files use ``AHU_2_9_...`` (underscore), not ``AHU_2-9_...``.
    """
    return sensor_desc_to_slug(normalize_sensor_desc(sensor_desc)).replace("2-9", "2_9")


def resolve_sensor_checkpoint(
    checkpoint: Union[str, Path],
    sensor_desc: str,
) -> Path:
    """
    Resolve which ``.pth`` to load for a sensor.

    - If *checkpoint* is a **file** → use it for every sensor (shared model).
    - If *checkpoint* is a **directory** → ``<dir>/<checkpoint_stem_for_sensor>.pth``.
    """
    base = Path(checkpoint)
    canon = normalize_sensor_desc(sensor_desc)
    if base.is_file():
        return base.resolve()
    if base.is_dir():
        stem = checkpoint_stem_for_sensor(canon)
        candidates = [
            base / f"{stem}_v3.pth",
            base / f"{stem}_v2.pth",
            base / f"{stem}.pth",
        ]
        for path in candidates:
            if path.is_file():
                return path.resolve()
        raise FileNotFoundError(
            f"No checkpoint for sensor {canon!r}: tried "
            + ", ".join(str(p) for p in candidates)
            + f". Stems: {[checkpoint_stem_for_sensor(s) for s in AHU_2_9_SENSOR_DESCS]}."
        )
    raise FileNotFoundError(f"Checkpoint path not found: {checkpoint}")


def build_vibration_sensor_model_map(models_dir: Union[str, Path]) -> Dict[str, dict]:
    """Build colleague-facing ``sensor_model_map`` from ``deployment/models/*.pth``."""
    models_dir = Path(models_dir)
    out: Dict[str, dict] = {}
    for sensor_name, filename, version in DEPLOYMENT_TRANSFORMER_MODELS:
        sensor_id = AHU_2_9_SENSOR_IDS[sensor_name]
        slug = sensor_desc_to_slug(sensor_name).replace("2-9", "2_9").lower()
        out[sensor_id] = {
            "sensorId": sensor_id,
            "sensorName": sensor_name,
            "models": {
                "transformer": {
                    "modelId": f"{slug}-transformer",
                    "modelName": "vib_transformer",
                    "version": version,
                    "filePath": str(models_dir / filename),
                },
            },
        }
    return out


def resolve_sensor_list(requested: Iterable[str] | None = None) -> List[str]:
    if requested is None:
        return list(AHU_2_9_SENSOR_DESCS)
    out: List[str] = []
    for item in requested:
        canon = normalize_sensor_desc(item)
        if canon not in AHU_2_9_SENSOR_DESCS:
            allowed = ", ".join(repr(s) for s in AHU_2_9_SENSOR_DESCS)
            raise ValueError(f"Unknown SENSOR_DESC {item!r}. Allowed: {allowed}")
        if canon not in out:
            out.append(canon)
    return out
