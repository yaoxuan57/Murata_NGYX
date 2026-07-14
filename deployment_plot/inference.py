"""End-to-end inference: raw multi-sensor CSV -> JSON predictions only."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

import _bootstrap  # noqa: F401

from expected_ranges import check_context_against_training_range
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

# Colleague-facing map: sensorId -> { sensorId, sensorName, models: { transformer: { filePath, ... } } }
SensorModelMap = Dict[str, Dict[str, Any]]


def _load_vibration_frame(input: VibrationInput) -> Tuple[pd.DataFrame, str]:
    """Return prepared frame and a label for error messages."""
    if isinstance(input, pd.DataFrame):
        return prepare_vibration_dataframe(input), "input DataFrame"
    return read_vibration_export_csv(input), input


class InferenceValidationError(Exception):
    """Input window failed validation checks."""

    def __init__(self, sensor_desc: str, message: str):
        self.sensor_desc = sensor_desc
        self.message = message
        super().__init__(message)


class InsufficientContextError(InferenceValidationError):
    """Fewer than ``input_len`` points were supplied for a sensor."""

    def __init__(self, sensor_desc: str, n_points: int, required_points: int):
        self.n_points = n_points
        self.required_points = required_points
        super().__init__(
            sensor_desc,
            f"less than {required_points} points",
        )


def load_sensor_prepared_rows(
    input: VibrationInput,
    sensor_desc: str,
    value_column: str = VALUE_COLUMN,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Filter one sensor, sort by time, dedupe timestamps — no row-count limit."""
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
    ts_out = parse_timestamp_series(part["TIMESTAMP"], name="TIMESTAMP")
    return part.drop(columns=["_ts"]), ts_out


def select_context_window(
    df: pd.DataFrame,
    ts: pd.Series,
    context_len: int,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Use all rows when *context_len* are provided; otherwise keep the latest *context_len*."""
    if len(df) > context_len:
        df = df.tail(context_len).reset_index(drop=True)
        ts = parse_timestamp_series(df["TIMESTAMP"], name="TIMESTAMP")
    return df, ts


def check_interval_warnings(
    ts: pd.Series,
    max_interval_hours: float,
) -> Optional[Dict[str, Any]]:
    """
    Return interval-check metadata when any consecutive gap exceeds *max_interval_hours*.

    *n_intervals_violating* counts gap pairs (for 48 points, at most 47 intervals).
    """
    if len(ts) < 2:
        return None

    diffs_hours = ts.diff().iloc[1:].dt.total_seconds() / 3600.0
    over = diffs_hours > float(max_interval_hours)
    n_viol = int(over.sum())
    if n_viol == 0:
        return None

    return {
        "max_interval_hours": float(max_interval_hours),
        "n_intervals_violating": n_viol,
        "warning": (
            f"{n_viol} interval(s) between consecutive points exceed "
            f"{max_interval_hours:g} hours."
        ),
    }


def load_sensor_context_rows(
    input: VibrationInput,
    sensor_desc: str,
    context_len: int,
    value_column: str = VALUE_COLUMN,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Filter one sensor and return exactly *context_len* latest rows + parsed timestamps.

    *input* may be a CSV path or a multi-sensor DataFrame (same columns as the export CSV).
    """
    part, ts = load_sensor_prepared_rows(input, sensor_desc, value_column=value_column)
    canon = normalize_sensor_desc(sensor_desc)

    if len(part) < context_len:
        raise InferenceValidationError(
            canon,
            f"Need at least {context_len} rows for this sensor after sort/dedup; found {len(part)}.",
        )

    return select_context_window(part, ts, context_len)


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
    max_interval_warning_hours: float,
    forecast_step_minutes: float,
    device: str,
) -> Dict[str, Any]:
    """Run inference for one sensor; raises on hard validation failures."""
    canon = normalize_sensor_desc(sensor_desc)
    input_len = int(ckpt["input_len"])
    pred_len = int(ckpt["pred_len"])

    df, ts = load_sensor_prepared_rows(input, canon)
    n_points = len(df)
    if n_points < input_len:
        raise InsufficientContextError(canon, n_points=n_points, required_points=input_len)

    df, ts = select_context_window(df, ts, input_len)

    interval_check = check_interval_warnings(ts, max_interval_warning_hours)
    if interval_check is not None:
        warnings.warn(str(interval_check["warning"]), UserWarning, stacklevel=2)

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

    range_check = check_context_against_training_range(canon, context_smooth)
    if range_check is not None and not range_check["in_training_range"]:
        warnings.warn(str(range_check["warning"]), UserWarning, stacklevel=2)

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
    if interval_check is not None:
        body["interval_check"] = interval_check
    if range_check is not None:
        body["range_check"] = range_check
    return body


def _insufficient_payload(sensor_desc: str, exc: InsufficientContextError) -> Dict[str, Any]:
    warnings.warn(exc.message, UserWarning, stacklevel=2)
    return {
        sensor_desc: {
            "success": False,
            "predicted": None,
            "warning": exc.message,
            "n_points": exc.n_points,
            "required_points": exc.required_points,
        }
    }


def run_inference_payload(
    input: VibrationInput,
    checkpoint: str,
    sensor_desc: str,
    *,
    smooth_window: int = 48,
    max_interval_warning_hours: float = 2.0,
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
            max_interval_warning_hours=max_interval_warning_hours,
            forecast_step_minutes=forecast_step_minutes,
            device=device,
        )
        body["checkpoint"] = str(ckpt_path)
        body["success"] = True
        return {canon: body}
    except FileNotFoundError as exc:
        return failure_payload(canon, str(exc))
    except InsufficientContextError as exc:
        return _insufficient_payload(exc.sensor_desc, exc)
    except InferenceValidationError as exc:
        return failure_payload(exc.sensor_desc, exc.message)


def run_inference_all_sensors(
    input: VibrationInput,
    checkpoint: str,
    sensor_descs: Optional[List[str]] = None,
    *,
    smooth_window: int = 48,
    max_interval_warning_hours: float = 2.0,
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
                max_interval_warning_hours=max_interval_warning_hours,
                forecast_step_minutes=forecast_step_minutes,
                device=device,
            )
            body["checkpoint"] = str(ckpt_path)
            body["success"] = True
            merged[canon] = body
            print(f"  OK: {canon}")
        except FileNotFoundError as exc:
            merged[canon] = failure_payload(canon, str(exc))[canon]
            print(f"  FAILED (checkpoint): {exc}")
        except InsufficientContextError as exc:
            merged[canon] = _insufficient_payload(exc.sensor_desc, exc)[exc.sensor_desc]
            print(f"  WARNING: {exc.message} ({exc.n_points}/{exc.required_points} points)")
        except InferenceValidationError as exc:
            merged[canon] = failure_payload(exc.sensor_desc, exc.message)[exc.sensor_desc]
            print(f"  FAILED: {exc.message}")
    return merged


def write_predictions_json(payload: Dict[str, Any], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)


def _resolve_transformer_checkpoint(sensor_entry: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Return (checkpoint_path, model_meta) from a sensor_model_map entry."""
    models = sensor_entry.get("models") or {}
    if "transformer" not in models:
        raise ValueError(
            f"No transformer model for sensor {sensor_entry.get('sensorId')!r}; "
            f"available: {sorted(models)}"
        )
    meta = dict(models["transformer"])
    path = meta.get("filePath")
    if not path:
        raise ValueError(
            f"transformer model missing filePath for sensor {sensor_entry.get('sensorId')!r}"
        )
    return str(path), meta


def _forecast_failure(sensor_id: str, sensor_name: Optional[str], message: str) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "sensorId": sensor_id,
        "success": False,
        "error": message,
    }
    if sensor_name is not None:
        body["sensorName"] = sensor_name
    return body


def _forecast_insufficient_points(
    sensor_id: str,
    sensor_name: str,
    exc: InsufficientContextError,
) -> Dict[str, Any]:
    warnings.warn(exc.message, UserWarning, stacklevel=2)
    return {
        "sensorId": sensor_id,
        "sensorName": normalize_sensor_desc(sensor_name),
        "success": False,
        "predicted": None,
        "warning": exc.message,
        "n_points": exc.n_points,
        "required_points": exc.required_points,
    }


def forecast_vibration(
    vib_df: pd.DataFrame,
    *,
    sensor_model_map: SensorModelMap,
    smooth_window: int = 48,
    max_interval_warning_hours: float = 2.0,
    forecast_step_minutes: float = 30.0,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Run transformer forecasts for every sensor in *sensor_model_map*.

    Colleagues should supply at most ``input_len`` rows per sensor (48 for current models).

    - **< input_len points** → ``success: false``, ``predicted: null``, warning
      ``"less than 48 points"``; model is not run.
    - **= input_len points**, all gaps ≤ ``max_interval_warning_hours`` → forecast only.
    - **= input_len points**, some gaps > ``max_interval_warning_hours`` → forecast plus
      ``interval_check`` warning with violation count.

    *vib_df* must include ``TIMESTAMP``, ``SENSOR_DESC``, and ``Acceleration RMS``
    (same schema as the vibration export CSV).

    *sensor_model_map* example::

        {
            "91E2": {
                "sensorId": "91E2",
                "sensorName": "AHU 2-9 Blower NDE A",
                "models": {
                    "transformer": {
                        "modelId": "uuid-xgb",
                        "modelName": "vib_transformer",
                        "version": "v2",
                        "filePath": "C:/.../deployment/models/AHU_2_9_Blower_NDE_A_v2.pth",
                    },
                },
            }
        }

    Returns a dict keyed by sensor id (``sensorId``). Successful entries include
    ``context_timestamps``, ``context_values``, ``timestamps``, ``predicted``, and
    optional quantile fields; failures set ``success`` to ``False`` with ``error``.
    """
    results: Dict[str, Any] = {}
    for sensor_id, entry in sensor_model_map.items():
        sensor_name = entry.get("sensorName")
        if not sensor_name:
            results[sensor_id] = _forecast_failure(
                sensor_id,
                None,
                "Missing sensorName in sensor_model_map entry.",
            )
            continue

        try:
            ckpt_path, model_meta = _resolve_transformer_checkpoint(entry)
            ckpt_file = Path(ckpt_path)
            if not ckpt_file.is_file():
                raise FileNotFoundError(f"Model file not found: {ckpt_path}")

            model, ckpt, args = load_checkpoint(str(ckpt_file), device)
            body = _inference_body_for_sensor(
                vib_df,
                sensor_name,
                model,
                ckpt,
                args,
                smooth_window=smooth_window,
                max_interval_warning_hours=max_interval_warning_hours,
                forecast_step_minutes=forecast_step_minutes,
                device=device,
            )
            results[sensor_id] = {
                "sensorId": sensor_id,
                "sensorName": normalize_sensor_desc(sensor_name),
                "success": True,
                "modelId": model_meta.get("modelId"),
                "modelName": model_meta.get("modelName"),
                "version": model_meta.get("version"),
                "filePath": str(ckpt_file.resolve()),
                **body,
            }
        except FileNotFoundError as exc:
            results[sensor_id] = _forecast_failure(sensor_id, sensor_name, str(exc))
        except InsufficientContextError as exc:
            results[sensor_id] = _forecast_insufficient_points(sensor_id, sensor_name, exc)
        except InferenceValidationError as exc:
            results[sensor_id] = _forecast_failure(sensor_id, exc.sensor_desc, exc.message)
        except ValueError as exc:
            results[sensor_id] = _forecast_failure(sensor_id, sensor_name, str(exc))

    return results


def _build_smoothed_timeline(part: pd.DataFrame, smooth_window: int) -> pd.DataFrame:
    """Smoothed RMS timeline aligned row-for-row with *part*."""
    rms = pd.to_numeric(part[VALUE_COLUMN], errors="coerce").to_numpy(dtype=np.float64)
    smooth = smooth_target_series_1d(rms, smooth_window)
    ts = parse_timestamp_series(part["TIMESTAMP"], name="TIMESTAMP")
    return pd.DataFrame({"ts": ts.values, "value": smooth.astype(float)})


def _window_result_from_body(
    body: Dict[str, Any],
    *,
    window_index: int,
    anchor_index: int,
    timeline: pd.DataFrame,
    pred_len: int,
    sensor_id: str,
    sensor_name: str,
    model_meta: Dict[str, Any],
    ckpt_file: Path,
) -> Dict[str, Any]:
    """Package one successful forecast window with holdout actuals for plotting/JSON."""
    act_end = min(anchor_index + pred_len, len(timeline))
    actual_ts = timeline["ts"].iloc[anchor_index:act_end]
    actual_val = timeline["value"].iloc[anchor_index:act_end]
    return {
        "window_index": window_index,
        "anchor_index": int(anchor_index),
        "anchor_timestamp": pd.Timestamp(timeline["ts"].iloc[anchor_index - 1]).isoformat(),
        "sensorId": sensor_id,
        "sensorName": normalize_sensor_desc(sensor_name),
        "success": True,
        "modelId": model_meta.get("modelId"),
        "modelName": model_meta.get("modelName"),
        "version": model_meta.get("version"),
        "filePath": str(ckpt_file.resolve()),
        "actual_timestamps": [pd.Timestamp(t).isoformat() for t in actual_ts.tolist()],
        "actual_values": [float(v) for v in actual_val.tolist()],
        **body,
    }


def _window_failure(
    *,
    window_index: int,
    anchor_index: Optional[int],
    sensor_id: str,
    sensor_name: str,
    message: str,
    n_points: Optional[int] = None,
    required_points: Optional[int] = None,
) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "window_index": window_index,
        "sensorId": sensor_id,
        "sensorName": normalize_sensor_desc(sensor_name),
        "success": False,
        "predicted": None,
        "warning": message,
    }
    if anchor_index is not None:
        body["anchor_index"] = int(anchor_index)
    if n_points is not None:
        body["n_points"] = n_points
    if required_points is not None:
        body["required_points"] = required_points
    return body


def forecast_rolling_windows(
    vib_df: pd.DataFrame,
    *,
    sensor_model_map: SensorModelMap,
    rolling_windows: int = 5,
    smooth_window: int = 48,
    max_interval_warning_hours: float = 2.0,
    forecast_step_minutes: float = 30.0,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Slide 48-point input windows across each sensor's history and forecast the next 48.

  For each window the same validation rules as :func:`forecast_vibration` apply:
  insufficient points skip the model; interval and training-range issues add warnings
  but still produce a forecast when there are exactly 48 input points.

  When a sensor has fewer than ``input_len + pred_len`` rows (96 for current models),
  a single latest-window forecast is attempted instead (colleague 48-point feed).

  Returns a dict keyed by sensor id. Each value has ``windows``: a list of per-window
  bodies including ``context_*``, ``predicted``, quantiles, ``interval_check``,
  ``range_check``, and ``actual_*`` holdout series when enough history exists.
    """
    results: Dict[str, Any] = {}

    for sensor_id, entry in sensor_model_map.items():
        sensor_name = entry.get("sensorName")
        if not sensor_name:
            results[sensor_id] = _forecast_failure(
                sensor_id,
                None,
                "Missing sensorName in sensor_model_map entry.",
            )
            continue

        try:
            ckpt_path, model_meta = _resolve_transformer_checkpoint(entry)
            ckpt_file = Path(ckpt_path)
            if not ckpt_file.is_file():
                raise FileNotFoundError(f"Model file not found: {ckpt_path}")

            model, ckpt, args = load_checkpoint(str(ckpt_file), device)
            input_len = int(ckpt["input_len"])
            pred_len = int(ckpt["pred_len"])
            min_rows = input_len + pred_len

            part, _ts = load_sensor_prepared_rows(vib_df, sensor_name)
            timeline = _build_smoothed_timeline(part, smooth_window)
            n = len(part)

            if n < input_len:
                results[sensor_id] = _forecast_insufficient_points(
                    sensor_id,
                    sensor_name,
                    InsufficientContextError(
                        normalize_sensor_desc(sensor_name),
                        n_points=n,
                        required_points=input_len,
                    ),
                )
                continue

            windows: List[Dict[str, Any]] = []

            def _run_one_window(
                window_df: pd.DataFrame,
                wi: int,
                anchor_i: Optional[int],
            ) -> Dict[str, Any]:
                try:
                    body = _inference_body_for_sensor(
                        window_df,
                        sensor_name,
                        model,
                        ckpt,
                        args,
                        smooth_window=smooth_window,
                        max_interval_warning_hours=max_interval_warning_hours,
                        forecast_step_minutes=forecast_step_minutes,
                        device=device,
                    )
                    if anchor_i is not None:
                        return _window_result_from_body(
                            body,
                            window_index=wi,
                            anchor_index=anchor_i,
                            timeline=timeline,
                            pred_len=pred_len,
                            sensor_id=sensor_id,
                            sensor_name=sensor_name,
                            model_meta=model_meta,
                            ckpt_file=ckpt_file,
                        )
                    return {
                        "window_index": wi,
                        "sensorId": sensor_id,
                        "sensorName": normalize_sensor_desc(sensor_name),
                        "success": True,
                        "modelId": model_meta.get("modelId"),
                        "modelName": model_meta.get("modelName"),
                        "version": model_meta.get("version"),
                        "filePath": str(ckpt_file.resolve()),
                        **body,
                    }
                except InsufficientContextError as exc:
                    warnings.warn(exc.message, UserWarning, stacklevel=2)
                    return _window_failure(
                        window_index=wi,
                        anchor_index=anchor_i,
                        sensor_id=sensor_id,
                        sensor_name=sensor_name,
                        message=exc.message,
                        n_points=exc.n_points,
                        required_points=exc.required_points,
                    )
                except InferenceValidationError as exc:
                    return _window_failure(
                        window_index=wi,
                        anchor_index=anchor_i,
                        sensor_id=sensor_id,
                        sensor_name=sensor_name,
                        message=exc.message,
                    )

            if n < min_rows or rolling_windows <= 0:
                window_df = part.copy()
                windows.append(_run_one_window(window_df, 0, None))
            else:
                i_min = input_len
                i_max = n - pred_len
                anchors = np.linspace(i_min, i_max, int(rolling_windows), dtype=int)
                anchors = np.unique(anchors)
                for wi, anchor_i in enumerate(anchors):
                    window_df = part.iloc[anchor_i - input_len : anchor_i].copy()
                    windows.append(_run_one_window(window_df, wi, int(anchor_i)))

            ok = sum(1 for w in windows if w.get("success") is True)
            results[sensor_id] = {
                "sensorId": sensor_id,
                "sensorName": normalize_sensor_desc(sensor_name),
                "n_rows": n,
                "n_windows": len(windows),
                "n_succeeded": ok,
                "windows": windows,
            }
        except FileNotFoundError as exc:
            results[sensor_id] = _forecast_failure(sensor_id, sensor_name, str(exc))
        except InferenceValidationError as exc:
            results[sensor_id] = _forecast_failure(sensor_id, exc.sensor_desc, exc.message)
        except ValueError as exc:
            results[sensor_id] = _forecast_failure(sensor_id, sensor_name, str(exc))

    return results
