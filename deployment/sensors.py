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
        path = base / f"{checkpoint_stem_for_sensor(canon)}.pth"
        if not path.is_file():
            raise FileNotFoundError(
                f"No checkpoint for sensor {canon!r}: expected {path}. "
                f"Stems: {[checkpoint_stem_for_sensor(s) for s in AHU_2_9_SENSOR_DESCS]}."
            )
        return path.resolve()
    raise FileNotFoundError(f"Checkpoint path not found: {checkpoint}")


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
