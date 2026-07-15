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


def load_sensor_mapping(mapping_csv: Path | str | None = None) -> Dict[str, Dict[str, str]]:
    """Return ``sensor_name -> {sensorId, sensorName}``."""
    path = Path(mapping_csv) if mapping_csv else default_sensor_mapping_path()
    if not path.is_file():
        return {
            name: {"sensorId": sid, "sensorName": name}
            for name, sid in SENSOR_ID_BY_NAME.items()
        }

    df = pd.read_csv(path, dtype=str)
    if "SENSOR_NAME" not in df.columns:
        raise ValueError(f"{path} missing SENSOR_NAME column")

    out: Dict[str, Dict[str, str]] = {}
    code_col = "SENSOR_CODE" if "SENSOR_CODE" in df.columns else None
    for _, row in df.drop_duplicates(subset=["SENSOR_NAME"]).iterrows():
        name = str(row["SENSOR_NAME"]).strip()
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
    for name, info in mapping.items():
        if name.lower() in cand_lower or cand_lower in name.lower():
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
    """Write ``<checkpoint_stem>_meta.json`` beside the ``.pth`` file."""
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

    if meta_path is None:
        meta_path = checkpoint_path.with_name(f"{checkpoint_path.stem}_meta.json")
    else:
        meta_path = Path(meta_path)

    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)
        fp.write("\n")
    return meta_path
