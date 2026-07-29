"""Write colleague-facing model metadata JSON next to each ``.pth`` checkpoint."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent

# Authoritative hex ids when the mapping CSV has Excel-mangled SENSOR_CODE values.
SENSOR_ID_BY_NAME: Dict[str, str] = {
    "AHU 2-9 Blower DE A": "91B8",
    "AHU 2-9 Blower DE V": "916E",
    "AHU 2-9 Blower DE Vibration X": "8894",
    "AHU 2-9 Blower NDE A": "91E2",
    "AHU 2-9 Blower NDE H": "91ED",
    "AHU 2-9 Blower NDE V": "91E1",
    "AHU 2-9 motor DE H": "91F0",
    "AHU 2-9 motor NDE H": "91AC",
    "AHU 4-4 Blower DE Vibration X": "88B3",
}


def default_sensor_mapping_path() -> Path:
    candidates = [
        SCRIPT_DIR / "data" / "sensor_id_name_mapping.csv",
        SCRIPT_DIR / "data_30mins_frequency" / "sensor_id_name_mapping.csv",
        SCRIPT_DIR.parent / "data" / "sensor_id_name_mapping.csv",
        SCRIPT_DIR.parent / "data_30mins_frequency" / "sensor_id_name_mapping.csv",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]


def _normalize_sensor_code(raw: object) -> str:
    text = str(raw).strip().upper()
    if not text or text.lower() in ("nan", "none"):
        return ""
    if re.fullmatch(r"[\d.]+E[+\-]?\d+", text, flags=re.IGNORECASE):
        return f"{int(float(text)):X}"
    if re.fullmatch(r"\d+\.0+", text):
        return f"{int(float(text)):X}"
    return text


def _sensor_name_column(df: pd.DataFrame) -> str:
    for col in ("SENSOR_NAME", "SENSOR_DESC"):
        if col in df.columns:
            return col
    raise ValueError(
        "CSV needs SENSOR_NAME or SENSOR_DESC to build sensor_id_name_mapping.csv. "
        f"Columns: {list(df.columns)}"
    )


def _sensor_code_column(df: pd.DataFrame) -> str:
    for col in ("SENSOR_CODE", "STN_CODE"):
        if col in df.columns:
            return col
    raise ValueError(
        "CSV needs SENSOR_CODE or STN_CODE to build sensor_id_name_mapping.csv. "
        f"Columns: {list(df.columns)}"
    )


def build_sensor_mapping_dataframe(source_csv: Path | str) -> pd.DataFrame:
    """
    Build a clean ``SENSOR_CODE`` / ``SENSOR_NAME`` table from a vibration export CSV.

    One row per unique sensor name; code is taken from the export as-is (normalized hex).
    """
    path = Path(source_csv)
    if not path.is_file():
        raise FileNotFoundError(f"Source CSV not found: {path}")

    df = pd.read_csv(path, low_memory=False)
    name_col = _sensor_name_column(df)
    code_col = _sensor_code_column(df)

    work = df[[code_col, name_col]].copy()
    work["SENSOR_CODE"] = work[code_col].map(_normalize_sensor_code)
    work["SENSOR_NAME"] = work[name_col].astype(str).str.strip()
    work = work[
        (work["SENSOR_CODE"] != "")
        & (work["SENSOR_NAME"] != "")
        & (work["SENSOR_NAME"].str.lower() != "nan")
    ]
    if work.empty:
        raise ValueError(f"No sensor code/name pairs found in {path}")

    return (
        work[["SENSOR_CODE", "SENSOR_NAME"]]
        .drop_duplicates(subset=["SENSOR_NAME"], keep="first")
        .sort_values("SENSOR_NAME", kind="mergesort")
        .reset_index(drop=True)
    )


def write_sensor_mapping_csv(
    source_csv: Path | str,
    out_path: Path | str | None = None,
) -> Path:
    """Write ``sensor_id_name_mapping.csv`` from unique sensors in *source_csv*."""
    mapping_df = build_sensor_mapping_dataframe(source_csv)
    out = Path(out_path) if out_path is not None else default_sensor_mapping_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    mapping_df.to_csv(out, index=False)
    return out


def load_sensor_mapping(mapping_csv: Path | str | None = None) -> Dict[str, Dict[str, str]]:
    """Return ``sensor_name -> {sensorId, sensorName}``."""
    path = Path(mapping_csv) if mapping_csv else default_sensor_mapping_path()
    if not path.is_file():
        return {
            name: {"sensorId": sid, "sensorName": name}
            for name, sid in SENSOR_ID_BY_NAME.items()
        }

    df = pd.read_csv(path, dtype=str)
    name_col = "SENSOR_NAME" if "SENSOR_NAME" in df.columns else "SENSOR_DESC"
    if name_col not in df.columns:
        raise ValueError(f"{path} missing SENSOR_NAME or SENSOR_DESC column")

    out: Dict[str, Dict[str, str]] = {}
    code_col = "SENSOR_CODE" if "SENSOR_CODE" in df.columns else None
    for _, row in df.drop_duplicates(subset=[name_col]).iterrows():
        name = str(row[name_col]).strip()
        if not name:
            continue
        sid = SENSOR_ID_BY_NAME.get(name, "")
        if code_col:
            sid = sid or _normalize_sensor_code(row[code_col])
        out[name] = {"sensorId": sid, "sensorName": name}
    for name, sid in SENSOR_ID_BY_NAME.items():
        out.setdefault(name, {"sensorId": sid, "sensorName": name})
    return out


def csv_stem_to_sensor_name(stem: str) -> str:
    """``AHU_2_9_Blower_DE_A_30_min`` / ``data_AHU_4_4_...`` -> ``AHU 2-9 Blower DE A``."""
    text = Path(stem).stem if "." in stem else stem
    text = re.sub(r"^data_", "", text, flags=re.IGNORECASE)
    text = re.sub(r"_30_min$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"_r[\d._]+$", "", text)
    text = re.sub(r"(?i)(AHU)[_ ]?(\d+)[_-](\d+)", r"\1 \2-\3", text)
    text = text.replace("_", " ")
    return re.sub(r"\s+", " ", text).strip()


def infer_machine_id(sensor_name: str) -> str:
    match = re.search(r"AHU\s*(\d+)\s*-\s*(\d+)", sensor_name, flags=re.IGNORECASE)
    if match:
        return f"AHU{match.group(1)}-{match.group(2)}"
    return ""


def resolve_sensor_info(
    data_source_path: str | Path | None,
    mapping: Mapping[str, Dict[str, str]],
) -> Dict[str, str]:
    if data_source_path is None:
        return {"sensorId": "", "sensorName": "", "machineId": ""}

    path = Path(data_source_path)
    stem = path.stem
    # Split CSVs: .../<sensor_or_data_stem>/train.csv
    # Finetune:   .../data_<STEM>/splits/{train,val,test}.csv
    if stem in ("train", "val", "test"):
        parent = path.parent
        if parent.name.lower() == "splits":
            stem = parent.parent.name
        else:
            stem = parent.name

    candidate = csv_stem_to_sensor_name(stem)
    if candidate in mapping:
        info = mapping[candidate]
        return {
            "sensorId": info["sensorId"],
            "sensorName": info["sensorName"],
            "machineId": infer_machine_id(info["sensorName"]),
        }

    cand_lower = candidate.lower()
    matches = [
        (name, info)
        for name, info in mapping.items()
        if name.lower() in cand_lower or cand_lower in name.lower()
    ]
    if matches:
        name, info = max(matches, key=lambda item: len(item[0]))
        return {
            "sensorId": info["sensorId"],
            "sensorName": info["sensorName"],
            "machineId": infer_machine_id(info["sensorName"]),
        }

    return {
        "sensorId": "",
        "sensorName": candidate,
        "machineId": infer_machine_id(candidate),
    }


def sensor_name_for_filename(sensor_name: str) -> str:
    """``AHU 2-9 Blower DE A`` -> ``AHU_2-9_Blower_DE_A`` (spaces -> underscores)."""
    return re.sub(r"\s+", "_", str(sensor_name).strip())


def deployment_bundle_basename(
    model_type: str,
    sensor_id: str,
    sensor_name: str,
) -> str:
    """``modelType__sensorID__sensorName`` for batch-upload pairing."""
    mt = str(model_type or "rms_forecast").strip()
    sid = str(sensor_id or "").strip().upper()
    sname = sensor_name_for_filename(sensor_name)
    if not sid or not sname:
        raise ValueError(f"deployment bundle requires sensorId and sensorName, got {sid!r} {sname!r}")
    return f"{mt}__{sid}__{sname}"


def deployment_model_filename(
    model_type: str,
    sensor_id: str,
    sensor_name: str,
    *,
    extension: str = ".pth",
) -> str:
    ext = extension if extension.startswith(".") else f".{extension}"
    return f"{deployment_bundle_basename(model_type, sensor_id, sensor_name)}{ext}"


def deployment_metadata_filename(
    model_type: str,
    sensor_id: str,
    sensor_name: str,
) -> str:
    return f"{deployment_bundle_basename(model_type, sensor_id, sensor_name)}.metadata.json"


def parse_deployment_filename(filename: str) -> Dict[str, str] | None:
    """
    Parse ``modelType__sensorID__sensorName`` from a model or metadata filename.

    Supports ``.pth``, ``.joblib``, and ``.metadata.json`` suffixes.
    """
    name = Path(filename).name
    for suffix in (".metadata.json", ".pth", ".joblib"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    parts = name.split("__")
    if len(parts) < 3:
        return None
    model_type, sensor_id, sensor_name_part = parts[0], parts[1], "__".join(parts[2:])
    if not model_type or not sensor_id or not sensor_name_part:
        return None
    return {
        "modelType": model_type,
        "sensorId": sensor_id.upper(),
        "sensorName": sensor_name_part.replace("_", " "),
        "sensorNameFile": sensor_name_part,
        "basename": name,
    }


def discover_deployment_model_pairs(models_dir: str | Path) -> Dict[str, Dict[str, Any]]:
    """
    Pair batch-uploaded model + metadata files by ``modelType__sensorID__sensorName``.

    Returns ``sensorId -> {modelPath, metadataPath, modelType, sensorName, basename}``.
    """
    root = Path(models_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Models directory not found: {root}")

    models: Dict[str, Dict[str, Any]] = {}
    metas: Dict[str, Dict[str, Any]] = {}

    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        parsed = parse_deployment_filename(path.name)
        if parsed is None:
            continue
        key = parsed["basename"]
        if path.name.endswith(".metadata.json"):
            metas[key] = {**parsed, "metadataPath": str(path.resolve())}
        elif path.suffix.lower() in {".pth", ".joblib"}:
            models[key] = {**parsed, "modelPath": str(path.resolve())}

    paired: Dict[str, Dict[str, Any]] = {}
    for key, model in models.items():
        meta = metas.get(key)
        sensor_id = model["sensorId"]
        entry = {
            **model,
            "metadataPath": meta["metadataPath"] if meta else None,
            "basename": key,
        }
        if sensor_id in paired:
            raise ValueError(f"Duplicate deployment model for sensorId={sensor_id!r} in {root}")
        paired[sensor_id] = entry
    return paired


def build_model_name(sensor_id: str, sensor_name: str = "", version: str = "v1") -> str:
    """Colleague convention: ``{sensorId}_rms_forecast`` (e.g. ``88B3_rms_forecast``)."""
    sid = str(sensor_id or "").strip().upper()
    if sid:
        return f"{sid}_rms_forecast"
    slug = (sensor_name or "transformer").replace(" ", "_").replace("2-9", "2_9")
    slug = re.sub(r"[^\w\-]+", "_", slug)
    slug = re.sub(r"_+", "_", slug).strip("_")
    version = str(version).strip() or "v1"
    if not version.startswith("v"):
        version = f"v{version}"
    return f"{slug}_{version}"


def forecast_metrics_block(metrics_payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Forecast metrics from ``best_metrics.json`` (no classification placeholders)."""
    raw = dict(metrics_payload.get("metrics") or {})
    block: Dict[str, Any] = {}
    has_quantiles = bool(metrics_payload.get("forecast_quantiles"))

    if metrics_payload.get("best_val_window_rmse") is not None:
        block["valWindowRmse"] = round(float(metrics_payload["best_val_window_rmse"]), 6)
    if metrics_payload.get("best_val_loss") is not None:
        val_loss = round(float(metrics_payload["best_val_loss"]), 6)
        if has_quantiles:
            block["valQuantileLoss"] = val_loss
        else:
            block["valTrajectoryLoss"] = val_loss
    if metrics_payload.get("baseline_rmse") is not None:
        block["baselineRmse"] = round(float(metrics_payload["baseline_rmse"]), 6)

    key_map = {
        "rmse": "testRmse",
        "mae": "testMae",
        "mape": "testMape",
        "r2": "testR2",
        "mse": "testMse",
    }
    for src, dst in key_map.items():
        if src in raw and raw[src] is not None:
            block[dst] = round(float(raw[src]), 6)

    if "testMape" in block:
        block["headlineMetric"] = "testMape"
        block["headlineValue"] = block["testMape"]
        block["headlineUnit"] = "percent"

    return block


def build_model_meta_payload(
    *,
    checkpoint_path: str | Path,
    best_config: Mapping[str, Any],
    metrics_payload: Mapping[str, Any],
    data_source_path: str | Path | None,
    model_version: str = "v1",
    mapping_csv: str | Path | None = None,
    trained_at: datetime | None = None,
) -> Dict[str, Any]:
    mapping = load_sensor_mapping(mapping_csv)
    sensor = resolve_sensor_info(data_source_path, mapping)

    feature_cols = list(best_config.get("feature_columns") or [])
    if not feature_cols:
        vc = best_config.get("value_column")
        if vc:
            feature_cols = [str(vc)]

    model_config = dict(best_config.get("model_config") or {})
    sensor_name = sensor["sensorName"]
    version = str(model_version).strip() or "v1"

    out_dir = str(best_config.get("output_dir") or checkpoint_path)
    is_finetune = "finetune" in out_dir.replace("\\", "/").lower()
    kind = "Finetuned transformer" if is_finetune else "Transformer"

    payload: Dict[str, Any] = {
        "modelName": build_model_name(sensor["sensorId"], sensor_name, version),
        "modelType": "rms_forecast",
        "sensorId": sensor["sensorId"],
        "sensorName": sensor_name,
        "machineId": sensor["machineId"],
        "trainedAt": (trained_at or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "description": (
            f"{kind} forecasting for {best_config.get('value_column', 'target')} "
            f"(input_len={best_config.get('best_input_len')}, pred_len={best_config.get('best_pred_len')})"
        ),
        "inputFeatures": feature_cols,
        "metrics": forecast_metrics_block(metrics_payload),
        "checkpointFile": Path(checkpoint_path).name,
        "inputLen": best_config.get("best_input_len"),
        "predLen": best_config.get("best_pred_len"),
        "valueColumn": best_config.get("value_column"),
        "forecastQuantiles": model_config.get("forecast_quantiles")
        or metrics_payload.get("forecast_quantiles"),
    }
    return payload


def save_model_meta_json(
    *,
    checkpoint_path: str | Path,
    best_config: Mapping[str, Any],
    metrics_payload: Mapping[str, Any],
    data_source_path: str | Path | None = None,
    model_version: str = "v1",
    mapping_csv: str | Path | None = None,
    meta_path: str | Path | None = None,
) -> Path:
    """Write ``modelType__sensorID__sensorName.metadata.json`` beside the ``.pth`` file."""
    checkpoint_path = Path(checkpoint_path)
    if data_source_path is None:
        data_source_path = (
            best_config.get("train_csv")
            or best_config.get("single_csv")
            or best_config.get("train_val_csv")
        )

    payload = build_model_meta_payload(
        checkpoint_path=checkpoint_path,
        best_config=best_config,
        metrics_payload=metrics_payload,
        data_source_path=data_source_path,
        model_version=model_version,
        mapping_csv=mapping_csv,
    )

    sensor_id = str(payload.get("sensorId") or "").strip()
    sensor_name = str(payload.get("sensorName") or "").strip()
    model_type = str(payload.get("modelType") or "rms_forecast").strip()

    if meta_path is None and sensor_id and sensor_name:
        std_meta_name = deployment_metadata_filename(model_type, sensor_id, sensor_name)
        std_model_name = deployment_model_filename(model_type, sensor_id, sensor_name)
        meta_path = checkpoint_path.parent / std_meta_name
        payload["checkpointFile"] = std_model_name
        std_checkpoint = checkpoint_path.parent / std_model_name
        if checkpoint_path.resolve() != std_checkpoint.resolve():
            if std_checkpoint.exists():
                std_checkpoint.unlink()
            checkpoint_path.rename(std_checkpoint)
    elif meta_path is None:
        meta_path = checkpoint_path.with_name(f"{checkpoint_path.stem}_meta.json")
    else:
        meta_path = Path(meta_path)

    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)
        fp.write("\n")
    return meta_path
