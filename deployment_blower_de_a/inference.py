"""Single-sensor inference: AHU 2-9 Blower DE A."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

import _bootstrap  # noqa: F401

from config import SENSOR_DESC, VALUE_COLUMN
from io_utils import parse_timestamp_series, prepare_vibration_dataframe, read_vibration_export_csv
from loader import load_checkpoint
from model_utils import median_quantile_index, smooth_target_series_1d

VibrationInput = Union[str, Path, pd.DataFrame]


class InferenceValidationError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


def _load_frame(input: VibrationInput) -> pd.DataFrame:
    if isinstance(input, pd.DataFrame):
        return prepare_vibration_dataframe(input)
    return read_vibration_export_csv(str(input))


def load_context_rows(
    input: VibrationInput,
    context_len: int,
    *,
    value_column: str = VALUE_COLUMN,
) -> Tuple[pd.DataFrame, pd.Series]:
    df = _load_frame(input)
    df["SENSOR_DESC"] = df["SENSOR_DESC"].astype(str).str.strip()
    part = df[df["SENSOR_DESC"] == SENSOR_DESC].copy()
    if part.empty:
        raise InferenceValidationError(
            f"No rows for SENSOR_DESC={SENSOR_DESC!r}. "
            f"Available (sample): {df['SENSOR_DESC'].drop_duplicates().head(5).tolist()}"
        )

    ts = parse_timestamp_series(part["TIMESTAMP"], name="TIMESTAMP", strict=False)
    valid = ts.notna()
    part = part.loc[valid].copy()
    part["_ts"] = ts.loc[valid].values
    part = part.sort_values("_ts", kind="mergesort").reset_index(drop=True)

    if value_column not in part.columns:
        raise InferenceValidationError(f"Missing column {value_column!r}.")

    part = part.drop_duplicates(subset=["TIMESTAMP"], keep="last").reset_index(drop=True)
    if len(part) < context_len:
        raise InferenceValidationError(
            f"Need at least {context_len} rows; found {len(part)} for {SENSOR_DESC}."
        )

    part = part.tail(context_len).reset_index(drop=True)
    ts_out = parse_timestamp_series(part["TIMESTAMP"], name="TIMESTAMP")
    return part.drop(columns=["_ts"]), ts_out


def validate_timestamp_continuity(ts: pd.Series, max_gap_seconds: float) -> Optional[str]:
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
    return (
        f"Gap at rows {first_i}->{first_i + 1} ({t0.isoformat()} -> {t1.isoformat()}): "
        f"{gap_s:.0f}s > max {max_gap_seconds:.0f}s."
    )


def build_smoothed_context(
    df: pd.DataFrame,
    ts: pd.Series,
    *,
    smooth_window: int,
    value_column: str = VALUE_COLUMN,
) -> np.ndarray:
    rms_raw = pd.to_numeric(df[value_column], errors="coerce").to_numpy(dtype=np.float64)
    if np.isnan(rms_raw).any():
        raise InferenceValidationError(f"Non-numeric values in {value_column!r}.")
    return smooth_target_series_1d(rms_raw, smooth_window).astype(np.float32)


def forecast_timestamps_after_context(
    ts: pd.Series,
    pred_len: int,
    *,
    step_minutes: float = 30.0,
) -> List[str]:
    last = pd.Timestamp(ts.iloc[-1])
    step = pd.Timedelta(minutes=step_minutes)
    stamps: List[str] = []
    for _ in range(pred_len):
        last = last + step
        stamps.append(last.isoformat())
    return stamps


def predict(
    model: torch.nn.Module,
    context_smooth: np.ndarray,
    *,
    train_mean: float,
    train_std: float,
    forecast_quantiles: Optional[List[float]],
    device: str,
) -> Dict[str, Any]:
    context_smooth = np.asarray(context_smooth, dtype=np.float32).reshape(-1)
    norm = (context_smooth - train_mean) / train_std
    x = torch.tensor(norm, dtype=torch.float32, device=device).view(1, 1, -1)
    last_val = float(norm[-1])

    with torch.no_grad():
        pred_delta = model(x)

    if forecast_quantiles is not None and len(forecast_quantiles) >= 2:
        pd_np = pred_delta.cpu().numpy()[0]
        pred_abs_norm = pd_np + last_val
        pred_raw = (pred_abs_norm * train_std + train_mean).astype(np.float32)
        fq = [float(q) for q in forecast_quantiles]
        mi = median_quantile_index(fq)
        return {
            "forecast_quantiles": fq,
            "predicted": [float(v) for v in pred_raw[mi]],
            "quantiles": {
                f"{float(q):g}": [float(v) for v in pred_raw[qi]]
                for qi, q in enumerate(fq)
            },
        }

    pd_np = pred_delta.cpu().numpy()[0]
    pred_abs_norm = pd_np + last_val
    pred_raw = (pred_abs_norm * train_std + train_mean).astype(np.float32)
    return {"predicted": [float(v) for v in pred_raw]}


def run_inference(
    input: VibrationInput,
    checkpoint: str | Path | None = None,
    *,
    smooth_window: int = 48,
    max_gap_seconds: float = 36000.0,
    forecast_step_minutes: float = 30.0,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Run inference for AHU 2-9 Blower DE A; returns one JSON-ready dict."""
    try:
        model, ckpt, args, ckpt_path = load_checkpoint(checkpoint, device)
        input_len = int(ckpt["input_len"])
        pred_len = int(ckpt["pred_len"])

        df, ts = load_context_rows(input, input_len)
        gap_err = validate_timestamp_continuity(ts, max_gap_seconds)
        if gap_err:
            raise InferenceValidationError(gap_err)

        context_smooth = build_smoothed_context(df, ts, smooth_window=smooth_window)
        forecast = predict(
            model,
            context_smooth,
            train_mean=float(ckpt["train_mean"]),
            train_std=float(ckpt["train_std"]),
            forecast_quantiles=args.forecast_quantiles,
            device=device,
        )

        ctx_ts = parse_timestamp_series(df["TIMESTAMP"], name="TIMESTAMP")
        body: Dict[str, Any] = {
            "sensor": SENSOR_DESC,
            "model_type": args.model_type,
            "checkpoint": str(ckpt_path),
            "input_len": input_len,
            "pred_len": pred_len,
            "context_timestamps": [pd.Timestamp(t).isoformat() for t in ctx_ts.tolist()],
            "context_values": [float(v) for v in context_smooth],
            "timestamps": forecast_timestamps_after_context(
                ts, pred_len, step_minutes=forecast_step_minutes
            ),
            "predicted": forecast["predicted"],
        }
        if "quantiles" in forecast:
            body["forecast_quantiles"] = forecast["forecast_quantiles"]
            body["quantiles"] = forecast["quantiles"]
        return body
    except InferenceValidationError as exc:
        return {"sensor": SENSOR_DESC, "success": False, "error": exc.message}
    except FileNotFoundError as exc:
        return {"sensor": SENSOR_DESC, "success": False, "error": str(exc)}
    except Exception as exc:
        return {"sensor": SENSOR_DESC, "success": False, "error": f"{type(exc).__name__}: {exc}"}


def write_json(payload: Dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)
