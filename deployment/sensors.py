"""Sensor helpers and colleague-owned model path routing for vibration deployment."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import pandas as pd

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

# Colleague mapping: sensorId -> absolute or relative model file path.
SensorPathMap = Dict[str, str]


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
    """``.pth`` filename stem under ``deployment/models/``."""
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


def normalize_sensor_code(value: object) -> str:
    """Normalize a sensor code without guessing its numeric base.

    Canonical IDs such as ``91B8`` remain unchanged. Numeric values are reduced
    to a stable decimal representation; registry-aware aliases later handle
    Excel conversions such as ``91E2`` being read as ``9100``.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip().upper()
    if not text or text in {"NAN", "NONE", "<NA>"}:
        return ""
    if re.fullmatch(r"[0-9A-F]+", text) and re.search(r"[A-F]", text):
        return text
    try:
        numeric = float(text)
    except ValueError:
        return text
    if not math.isfinite(numeric):
        return ""
    if numeric.is_integer():
        return str(int(numeric))
    return format(numeric, ".15g").upper()


def _sensor_id_aliases(sensor_id: str) -> set[str]:
    """Aliases for a model ID, including Excel/scientific-notation damage."""
    canonical = normalize_sensor_code(sensor_id)
    aliases = {canonical}
    # Hex IDs such as 91E2 can be interpreted by Excel as 91 × 10² = 9100.
    if re.fullmatch(r"\d+E[+\-]?\d+", canonical):
        try:
            aliases.add(normalize_sensor_code(float(canonical)))
        except ValueError:
            pass
    return {alias for alias in aliases if alias}


def _version_from_path(path: Path) -> str:
    match = re.search(r"_v(\d+)(?:\.pth)?$", path.name, flags=re.IGNORECASE)
    return f"v{match.group(1)}" if match else ""


def csv_sensor_code_column(vib_df: pd.DataFrame) -> str:
    if "SENSOR_CODE" in vib_df.columns:
        return "SENSOR_CODE"
    if "STN_CODE" in vib_df.columns:
        return "STN_CODE"
    raise ValueError(
        "Input CSV needs SENSOR_CODE (or legacy STN_CODE) for model selection."
    )


def unique_sensor_codes(vib_df: pd.DataFrame) -> Tuple[str, set[str]]:
    """Return (code_column, normalized unique codes present in the CSV)."""
    code_column = csv_sensor_code_column(vib_df)
    codes = {
        normalize_sensor_code(value)
        for value in vib_df[code_column].dropna().unique().tolist()
    }
    codes.discard("")
    return code_column, codes


def sensor_names_by_code(vib_df: pd.DataFrame) -> Dict[str, str]:
    """Map each CSV sensor code to its most common ``SENSOR_DESC``."""
    code_column = csv_sensor_code_column(vib_df)
    if "SENSOR_DESC" not in vib_df.columns:
        raise ValueError("Input CSV needs SENSOR_DESC to resolve sensor names.")

    work = vib_df[[code_column, "SENSOR_DESC"]].copy()
    work["_code"] = work[code_column].map(normalize_sensor_code)
    work["SENSOR_DESC"] = work["SENSOR_DESC"].map(normalize_sensor_desc)
    work = work[work["_code"] != ""]
    if work.empty:
        return {}

    counts = (
        work.groupby(["_code", "SENSOR_DESC"], sort=False)
        .size()
        .reset_index(name="n")
        .sort_values(["_code", "n"], ascending=[True, False])
    )
    out: Dict[str, str] = {}
    for code, group in counts.groupby("_code", sort=False):
        out[str(code)] = str(group.iloc[0]["SENSOR_DESC"])
    return out


def load_sensor_path_map(
    source: Union[str, Path, Mapping[str, Any]],
) -> SensorPathMap:
    """
    Load colleague mapping ``{sensorId: modelFilePath}``.

    *source* may be an in-memory dict or a JSON file path containing that dict.
    Values may be plain path strings, or objects with a ``filePath`` / ``path`` key.
    """
    if isinstance(source, Mapping):
        raw: Mapping[str, Any] = source
    else:
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"Model path map not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(
                f"Model path map must be a JSON object (sensorId -> path), got "
                f"{type(payload).__name__} in {path}"
            )
        raw = payload

    out: SensorPathMap = {}
    for key, value in raw.items():
        sensor_id = normalize_sensor_code(key)
        if not sensor_id:
            continue
        if isinstance(value, Mapping):
            path_text = str(
                value.get("filePath") or value.get("path") or value.get("checkpointFile") or ""
            ).strip()
        else:
            path_text = str(value).strip()
        if not path_text:
            raise ValueError(f"Empty model path for sensorId={sensor_id!r}")
        out[sensor_id] = path_text
    if not out:
        raise ValueError("Model path map is empty.")
    return out


def resolve_model_file_path(
    path_text: str,
    *,
    base_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Resolve absolute paths as-is; resolve relative paths against *base_dir*."""
    path = Path(path_text)
    if not path.is_absolute():
        root = Path(base_dir) if base_dir is not None else Path.cwd()
        path = root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")
    return path


def _optional_sidecar_meta(checkpoint_path: Path) -> Dict[str, Any]:
    """If a sibling ``.json`` exists beside the ``.pth``, load light metadata."""
    meta_path = checkpoint_path.with_suffix(".json")
    if not meta_path.is_file():
        return {}
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    keep = {
        "modelId": payload.get("modelId") or payload.get("modelName"),
        "modelName": payload.get("modelName"),
        "version": payload.get("version"),
        "metadataFile": str(meta_path.resolve()),
    }
    return {k: v for k, v in keep.items() if v is not None}


def _alias_index(path_map: Mapping[str, str]) -> Dict[str, str]:
    """Map every alias of a mapped sensorId back to the canonical map key."""
    alias_to_id: Dict[str, str] = {}
    ambiguous: set[str] = set()
    for sensor_id in path_map:
        for alias in _sensor_id_aliases(sensor_id):
            previous = alias_to_id.get(alias)
            if previous is not None and previous != sensor_id:
                ambiguous.add(alias)
            else:
                alias_to_id[alias] = sensor_id
    for alias in ambiguous:
        alias_to_id.pop(alias, None)
    return alias_to_id


def build_sensor_model_map_from_path_dict(
    path_map: Mapping[str, str],
    vib_df: pd.DataFrame,
    *,
    base_dir: Optional[Union[str, Path]] = None,
) -> Tuple[Dict[str, dict], Dict[str, Any]]:
    """
    Route CSV sensor codes through a colleague path dictionary.

    Flow:
      CSV SENSOR_CODE → look up in *path_map* → load that ``.pth``
      sensorName comes from the CSV ``SENSOR_DESC`` for that code.
    """
    normalized_map = {
        normalize_sensor_code(sensor_id): str(path).strip()
        for sensor_id, path in path_map.items()
        if normalize_sensor_code(sensor_id) and str(path).strip()
    }
    if not normalized_map:
        raise ValueError("Model path map is empty after normalization.")

    code_column, input_codes = unique_sensor_codes(vib_df)
    names_by_code = sensor_names_by_code(vib_df)
    alias_to_id = _alias_index(normalized_map)

    matched_ids = sorted(
        {
            alias_to_id[code]
            for code in input_codes
            if code in alias_to_id
        }
    )
    unmatched_codes = sorted(code for code in input_codes if code not in alias_to_id)

    selected: Dict[str, dict] = {}
    missing_files: List[str] = []
    missing_names: List[str] = []

    for sensor_id in matched_ids:
        # Prefer a CSV code that aliases to this sensor_id for name lookup.
        name = ""
        for code, desc in names_by_code.items():
            if alias_to_id.get(code) == sensor_id:
                name = desc
                break
        if not name:
            missing_names.append(sensor_id)
            continue

        try:
            checkpoint_path = resolve_model_file_path(
                normalized_map[sensor_id],
                base_dir=base_dir,
            )
        except FileNotFoundError as exc:
            missing_files.append(str(exc))
            continue

        model_meta: Dict[str, Any] = {
            "filePath": str(checkpoint_path),
            "version": _version_from_path(checkpoint_path),
        }
        model_meta.update(_optional_sidecar_meta(checkpoint_path))
        if not model_meta.get("modelName"):
            model_meta["modelName"] = f"{sensor_id}_rms_forecast"
        if not model_meta.get("modelId"):
            model_meta["modelId"] = model_meta["modelName"]

        selected[sensor_id] = {
            "sensorId": sensor_id,
            "sensorName": normalize_sensor_desc(name),
            "models": {"transformer": model_meta},
        }

    routing = {
        "codeColumn": code_column,
        "uniqueInputCodes": sorted(input_codes),
        "matchedSensorIds": sorted(selected),
        "unmatchedInputCodes": unmatched_codes,
        "availableModelSensorIds": sorted(normalized_map),
        "missingModelFiles": missing_files,
        "missingSensorNames": missing_names,
    }
    return selected, routing


def resolve_sensor_list(requested: Iterable[str] | None = None) -> List[str]:
    if requested is None:
        return list(AHU_2_9_SENSOR_DESCS)
    out: List[str] = []
    for item in requested:
        canon = normalize_sensor_desc(item)
        if canon not in out:
            out.append(canon)
    return out
