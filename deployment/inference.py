"""End-to-end inference: raw multi-sensor CSV -> JSON predictions only."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

import _bootstrap  # noqa: F401

from io_utils import parse_timestamp_series, prepare_vibration_dataframe, read_vibration_export_csv
from model_utils import median_quantile_index, smooth_target_series_1d
from sensors import (
    AHU_2_9_SENSOR_DESCS,
    normalize_sensor_desc,
    resolve_sensor_checkpoint,
    resolve_sensor_list,
)
from transformer_model import make_model
from windowing import VALUE_COLUMN

# Multi-sensor table: CSV path or in-memory DataFrame (TIMESTAMP, SENSOR_DESC, Acceleration RMS, …).
VibrationInput = Union[str, pd.DataFrame]


def _load_vibration_frame(input: VibrationInput) -> Tuple[pd.DataFrame, str]:
    """Return prepared frame and a label for error messages."""
    if isinstance(input, pd.DataFrame):
        return prepare_vibration_dataframe(input), "input DataFrame"
    return read_vibration_export_csv(input), input


class InferenceValidationError(Exception):
    """Input window failed timestamp continuity or row-count checks."""

    def __init__(self, sensor_desc: str, message: str):
        self.sensor_desc = sensor_desc
        self.message = message
        super().__init__(message)


def load_sensor_context_rows(
    input: VibrationInput,
    sensor_desc: str,
    context_len: int,
    value_column: str = VALUE_COLUMN,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Filter one sensor and return exactly *context_len* latest rows + parsed timestamps.

    *input* may be a CSV path or a multi-sensor DataFrame (same columns as the export CSV).
    """
    canon = normalize_sensor_desc(sensor_desc)
    resolve_sensor_list([canon])

    df, source = _load_vibration_frame(input)
    df["SENSOR_DESC"] = df["SENSOR_DESC"].map(normalize_sensor_desc)
    part = df[df["SENSOR_DESC"] == canon].copy()
    if part.empty:
        raise InferenceValidationError(
            canon,
            f"No rows found for SENSOR_DESC={canon!r} in {source}.",
        )

    ts = parse_timestamp_series(part["TIMESTAMP"], name="TIMESTAMP", strict=False)
    valid = ts.notna()
    part = part.loc[valid].copy()
    part["_ts"] = ts.loc[valid].values
    part = part.sort_values("_ts", kind="mergesort").reset_index(drop=True)

    if value_column not in part.columns:
        raise InferenceValidationError(
            canon,
            f"Missing column {value_column!r} in filtered data.",
        )

    part = part.drop_duplicates(subset=["TIMESTAMP"], keep="last").reset_index(drop=True)

    if len(part) < context_len:
        raise InferenceValidationError(
            canon,
            f"Need at least {context_len} rows for this sensor after sort/dedup; found {len(part)}.",
        )

    part = part.tail(context_len).reset_index(drop=True)
    ts_out = parse_timestamp_series(part["TIMESTAMP"], name="TIMESTAMP")
    return part.drop(columns=["_ts"]), ts_out


def load_sensor_tail_rows(
    input: VibrationInput,
    sensor_desc: str,
    n_rows: int,
    value_column: str = VALUE_COLUMN,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Like :func:`load_sensor_context_rows` but returns the latest *n_rows* rows."""
    if n_rows <= 0:
        raise ValueError("n_rows must be positive.")
    return load_sensor_context_rows(input, sensor_desc, context_len=n_rows, value_column=value_column)


def validate_timestamp_continuity(
    ts: pd.Series,
    max_gap_seconds: float,
) -> Optional[str]:
    """Return an error message if any consecutive gap exceeds *max_gap_seconds*, else None."""
    if len(ts) < 2:
        return "Need at least 2 timestamps to check continuity."

    diffs_sec = ts.diff().iloc[1:].dt.total_seconds()
    over = diffs_sec > float(max_gap_seconds)
    if not over.any():
        return None

    first_i = int(over.argmax())
    gap_s = float(diffs_sec.iloc[first_i])
    t0 = pd.Timestamp(ts.iloc[first_i])
    t1 = pd.Timestamp(ts.iloc[first_i + 1])
    max_min = max_gap_seconds / 60.0
    gap_min = gap_s / 60.0
    return (
        f"Timestamp continuity check failed: consecutive points at rows {first_i} and {first_i + 1} "
        f"({t0.isoformat()} -> {t1.isoformat()}) are {gap_s:.0f}s apart ({gap_min:.1f} min), "
        f"which exceeds the maximum allowed {max_gap_seconds:.0f}s ({max_min:.0f} min)."
    )


def build_smoothed_single_context(
    df: pd.DataFrame,
    ts: pd.Series,
    *,
    smooth_window: int = 48,
    value_column: str = VALUE_COLUMN,
) -> np.ndarray:
    """Causal (trailing) smooth on exactly len(df) RMS values; shape (context_len,)."""
    rms_raw = pd.to_numeric(df[value_column], errors="coerce").to_numpy(dtype=np.float64)
    if np.isnan(rms_raw).any():
        raise InferenceValidationError(
            str(df["SENSOR_DESC"].iloc[0]) if "SENSOR_DESC" in df.columns else "unknown",
            f"Non-numeric values in {value_column!r}.",
        )
    if len(rms_raw) != len(ts):
        raise ValueError("Timestamp and value length mismatch.")

    return smooth_target_series_1d(rms_raw, smooth_window).astype(np.float32)


def forecast_timestamps_after_context(
    ts: pd.Series,
    pred_len: int,
    *,
    step_minutes: float = 30.0,
) -> List[str]:
    """Extrapolate *pred_len* future timestamps: last context time + step_minutes per step."""
    last = pd.Timestamp(ts.iloc[-1])
    step = pd.Timedelta(minutes=step_minutes)
    stamps: List[str] = []
    for _ in range(pred_len):
        last = last + step
        stamps.append(last.isoformat())
    return stamps


def load_checkpoint(checkpoint_path: str, device: str) -> Tuple[torch.nn.Module, dict, SimpleNamespace]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    required = {"model_state_dict", "train_mean", "train_std", "input_len", "pred_len", "model_config"}
    missing = required - set(ckpt.keys())
    if missing:
        raise ValueError(f"Checkpoint missing keys: {sorted(missing)}")

    mc = dict(ckpt["model_config"])
    fq = mc.get("forecast_quantiles")
    args = SimpleNamespace(
        d_model=int(mc["d_model"]),
        nhead=int(mc["nhead"]),
        num_layers=int(mc["num_layers"]),
        dim_feedforward=int(mc["dim_feedforward"]),
        dropout=float(mc.get("dropout", 0.1)),
        input_dim=int(mc.get("input_dim", 1)),
        forecast_quantiles=list(fq) if fq else None,
    )

    input_len = int(ckpt["input_len"])
    pred_len = int(ckpt["pred_len"])
    model = make_model(input_len, pred_len, args, device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt, args


def _quantile_key(q: float) -> str:
    return f"{float(q):g}"


def predict_single_context(
    model: torch.nn.Module,
    context_raw: np.ndarray,
    *,
    train_mean: float,
    train_std: float,
    forecast_quantiles: Optional[List[float]],
    device: str,
) -> Dict[str, Any]:
    """
    Return forecast dict in raw Acceleration RMS units.

    Point model: ``{"predicted": [pred_len floats]}``
    Quantile model: also ``forecast_quantiles``, ``quantiles`` (per level), ``predicted`` (median).
    """
    context_raw = np.asarray(context_raw, dtype=np.float32).reshape(-1)
    norm = (context_raw - train_mean) / train_std
    x = torch.tensor(norm, dtype=torch.float32, device=device).view(1, 1, -1)
    last_val = float(norm[-1])

    with torch.no_grad():
        pred_delta = model(x)

    quantile_mode = forecast_quantiles is not None and len(forecast_quantiles) >= 2
    if quantile_mode:
        pd_np = pred_delta.cpu().numpy()[0]
        pred_abs_norm = pd_np + last_val
        pred_raw = (pred_abs_norm * train_std + train_mean).astype(np.float32)
        fq = [float(q) for q in forecast_quantiles]
        mi = median_quantile_index(fq)
        return {
            "forecast_quantiles": fq,
            "predicted": [float(v) for v in pred_raw[mi]],
            "quantiles": {
                _quantile_key(q): [float(v) for v in pred_raw[qi]]
                for qi, q in enumerate(fq)
            },
        }

    pd_np = pred_delta.cpu().numpy()[0]
    pred_abs_norm = pd_np + last_val
    pred_raw = (pred_abs_norm * train_std + train_mean).astype(np.float32)
    return {"predicted": [float(v) for v in pred_raw]}


def failure_payload(sensor_desc: str, message: str) -> Dict[str, Any]:
    return {
        sensor_desc: {
            "success": False,
            "error": message,
        }
    }


def _inference_body_for_sensor(
    input: VibrationInput,
    sensor_desc: str,
    model: torch.nn.Module,
    ckpt: dict,
    args: SimpleNamespace,
    *,
    smooth_window: int,
    max_gap_seconds: float,
    forecast_step_minutes: float,
    device: str,
) -> Dict[str, Any]:
    """Run inference for one sensor; raises InferenceValidationError on failure."""
    canon = normalize_sensor_desc(sensor_desc)
    input_len = int(ckpt["input_len"])
    pred_len = int(ckpt["pred_len"])

    df, ts = load_sensor_context_rows(input, canon, context_len=input_len)

    gap_err = validate_timestamp_continuity(ts, max_gap_seconds)
    if gap_err is not None:
        raise InferenceValidationError(canon, gap_err)

    context_smooth = build_smoothed_single_context(
        df,
        ts,
        smooth_window=smooth_window,
    )
    if context_smooth.shape[0] != input_len:
        raise InferenceValidationError(
            canon,
            f"Internal error: context length {context_smooth.shape[0]} != expected {input_len}.",
        )

    forecast = predict_single_context(
        model,
        context_smooth,
        train_mean=float(ckpt["train_mean"]),
        train_std=float(ckpt["train_std"]),
        forecast_quantiles=args.forecast_quantiles,
        device=device,
    )

    pred_list = forecast["predicted"]
    if len(pred_list) != pred_len:
        raise InferenceValidationError(
            canon,
            f"Model returned {len(pred_list)} steps; expected {pred_len}.",
        )

    ctx_ts = parse_timestamp_series(df["TIMESTAMP"], name="TIMESTAMP")
    body: Dict[str, Any] = {
        "context_timestamps": [pd.Timestamp(t).isoformat() for t in ctx_ts.tolist()],
        "context_values": [float(v) for v in context_smooth],
        "timestamps": forecast_timestamps_after_context(
            ts,
            pred_len,
            step_minutes=forecast_step_minutes,
        ),
        "predicted": pred_list,
    }
    if "quantiles" in forecast:
        body["forecast_quantiles"] = forecast["forecast_quantiles"]
        body["quantiles"] = forecast["quantiles"]
    return body


def run_inference_payload(
    input: VibrationInput,
    checkpoint: str,
    sensor_desc: str,
    *,
    smooth_window: int = 48,
    max_gap_seconds: float = 36000.0,
    forecast_step_minutes: float = 30.0,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Take exactly ``input_len`` rows from the checkpoint (48 for current models) as context.

    *input* is a CSV path or a multi-sensor DataFrame with ``TIMESTAMP``, ``SENSOR_DESC``,
    and ``Acceleration RMS`` (or the column used at training time).

    *checkpoint* may be a single ``.pth`` file or a directory (per-sensor routing).

    Returns a one-key dict: ``{ "<SENSOR_DESC>": { ... } }`` (success or failure body).
    """
    canon = normalize_sensor_desc(sensor_desc)
    try:
        ckpt_path = resolve_sensor_checkpoint(checkpoint, canon)
        model, ckpt, args = load_checkpoint(str(ckpt_path), device)
        body = _inference_body_for_sensor(
            input,
            canon,
            model,
            ckpt,
            args,
            smooth_window=smooth_window,
            max_gap_seconds=max_gap_seconds,
            forecast_step_minutes=forecast_step_minutes,
            device=device,
        )
        body["checkpoint"] = str(ckpt_path)
        return {canon: body}
    except FileNotFoundError as exc:
        return failure_payload(canon, str(exc))
    except InferenceValidationError as exc:
        return failure_payload(exc.sensor_desc, exc.message)


def run_inference_all_sensors(
    input: VibrationInput,
    checkpoint: str,
    sensor_descs: Optional[List[str]] = None,
    *,
    smooth_window: int = 48,
    max_gap_seconds: float = 36000.0,
    forecast_step_minutes: float = 30.0,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Run inference for every listed sensor; merge into one JSON object (multiple top-level keys).

    *input* is a CSV path or a multi-sensor DataFrame (same schema as the vibration export).

    *checkpoint* should be ``models/`` (one ``<stem>.pth`` per sensor).
    If it is a single ``.pth`` file, that same weights file is used for all sensors.

    Failed sensors are included with ``"success": false`` and ``"error"``; others get forecasts.
    """
    targets = resolve_sensor_list(sensor_descs) if sensor_descs else list(AHU_2_9_SENSOR_DESCS)
    ckpt_base = Path(checkpoint)
    if ckpt_base.is_file():
        print(f"Using one shared checkpoint for all sensors: {ckpt_base.resolve()}")

    merged: Dict[str, Any] = {}
    for sensor in targets:
        canon = normalize_sensor_desc(sensor)
        try:
            ckpt_path = resolve_sensor_checkpoint(checkpoint, canon)
            print(f"Running inference: {canon}  ->  {ckpt_path.name}")
            model, ckpt, args = load_checkpoint(str(ckpt_path), device)
            body = _inference_body_for_sensor(
                input,
                canon,
                model,
                ckpt,
                args,
                smooth_window=smooth_window,
                max_gap_seconds=max_gap_seconds,
                forecast_step_minutes=forecast_step_minutes,
                device=device,
            )
            body["checkpoint"] = str(ckpt_path)
            merged[canon] = body
            print(f"  OK: {canon}")
        except FileNotFoundError as exc:
            merged[canon] = failure_payload(canon, str(exc))[canon]
            print(f"  FAILED (checkpoint): {exc}")
        except InferenceValidationError as exc:
            merged[canon] = failure_payload(exc.sensor_desc, exc.message)[exc.sensor_desc]
            print(f"  FAILED: {exc.message}")
    return merged


def write_predictions_json(payload: Dict[str, Any], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)
