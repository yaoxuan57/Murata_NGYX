"""Plot inference JSON: input context + forecast (median and quantile bands)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sensors import checkpoint_stem_for_sensor, normalize_sensor_desc


def _slug(sensor_name: str) -> str:
    return checkpoint_stem_for_sensor(sensor_name)


def _ordered_quantile_keys(
    quantiles: Dict[str, List[float]],
    forecast_quantiles: Optional[List[float]],
) -> List[str]:
    if forecast_quantiles:
        keys = [f"{float(q):g}" for q in forecast_quantiles]
        return [k for k in keys if k in quantiles]
    return sorted(quantiles.keys(), key=lambda k: float(k))


def _context_series(body: Dict[str, Any]) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
    """Return (context timestamps, context values) if stored in JSON."""
    ctx_ts = body.get("context_timestamps")
    ctx_val = body.get("context_values")
    if not ctx_ts or not ctx_val or len(ctx_ts) != len(ctx_val):
        return None, None
    ts = pd.to_datetime(ctx_ts)
    if isinstance(ts, pd.DatetimeIndex):
        ts = pd.Series(ts, name="TIMESTAMP")
    return ts, pd.Series(ctx_val, dtype=float)


def load_history_before_json_context(
    sensor_name: str,
    csv_path: Path,
    body: Dict[str, Any],
    *,
    history_before: int,
    smooth_window: int,
) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
    """
    Rows strictly before the model input window stored in JSON (same CSV as inference).

    Does not recompute the 48-point input — that must come from ``context_*`` in JSON.
    """
    from inference import build_smoothed_single_context
    from io_utils import parse_timestamp_series, read_vibration_export_csv

    ctx_ts, _ = _context_series(body)
    if ctx_ts is None or history_before <= 0:
        return None, None

    canon = normalize_sensor_desc(sensor_name)
    df = read_vibration_export_csv(str(csv_path))
    df["SENSOR_DESC"] = df["SENSOR_DESC"].map(normalize_sensor_desc)
    part = df[df["SENSOR_DESC"] == canon].copy()
    if part.empty:
        return None, None

    ts = parse_timestamp_series(part["TIMESTAMP"], name="TIMESTAMP", strict=False)
    valid = ts.notna()
    part = part.loc[valid].copy()
    part["_ts"] = ts.loc[valid].values
    part = part.sort_values("_ts", kind="mergesort").reset_index(drop=True)
    part = part.drop_duplicates(subset=["TIMESTAMP"], keep="last").reset_index(drop=True)

    t0 = pd.Timestamp(ctx_ts.iloc[0])
    prior = part[part["_ts"] < t0].tail(int(history_before))
    if prior.empty:
        return None, None

    prior_ts = parse_timestamp_series(prior["TIMESTAMP"], name="TIMESTAMP")
    prior_df = prior.drop(columns=["_ts"], errors="ignore")
    smooth = build_smoothed_single_context(
        prior_df, prior_ts, smooth_window=smooth_window
    )
    return prior_ts.reset_index(drop=True), pd.Series(smooth, dtype=float)


def enrich_context_from_csv(
    body: Dict[str, Any],
    sensor_name: str,
    csv_path: Path,
    *,
    smooth_window: int,
    max_gap_seconds: float,
    checkpoint: Path,
) -> Dict[str, Any]:
    """Rebuild context_* fields when plotting an older predictions.json."""
    from inference import (
        build_smoothed_single_context,
        load_checkpoint,
        load_sensor_context_rows,
        validate_timestamp_continuity,
    )
    from inference import InferenceValidationError

    canon = normalize_sensor_desc(sensor_name)
    if body.get("context_timestamps") and body.get("context_values"):
        return body

    ckpt_path = checkpoint
    if checkpoint.is_dir():
        from sensors import resolve_sensor_checkpoint

        ckpt_path = resolve_sensor_checkpoint(checkpoint, canon)

    model, ckpt, args = load_checkpoint(str(ckpt_path), "cpu")
    del model, args

    input_len = int(ckpt["input_len"])
    df, ts = load_sensor_context_rows(str(csv_path), canon, context_len=input_len)
    gap_err = validate_timestamp_continuity(ts, max_gap_seconds)
    if gap_err is not None:
        raise InferenceValidationError(canon, gap_err)

    context_smooth = build_smoothed_single_context(df, ts, smooth_window=smooth_window)
    from io_utils import parse_timestamp_series

    ctx_ts = parse_timestamp_series(df["TIMESTAMP"], name="TIMESTAMP")
    out = dict(body)
    out["context_timestamps"] = [pd.Timestamp(t).isoformat() for t in ctx_ts.tolist()]
    out["context_values"] = [float(v) for v in context_smooth]
    return out


def plot_sensor_forecast(
    sensor_name: str,
    body: Dict[str, Any],
    out_path: Path,
    *,
    title_suffix: str = "",
    history_before: int = 0,
    input_csv: Optional[Path] = None,
    models_dir: Optional[Path] = None,
    smooth_window: int = 48,
) -> None:
    """Save one PNG: optional history + input context + median forecast + quantile band."""
    ts_fc = pd.to_datetime(body["timestamps"])
    pred = body["predicted"]
    quantiles = body.get("quantiles") or {}
    fq = body.get("forecast_quantiles")

    hist_ts: Optional[pd.Series] = None
    hist_val: Optional[pd.Series] = None
    ctx_ts, ctx_val = _context_series(body)

    # Model input + forecast must match inference JSON (do not rebuild 48-pt context from CSV tail).
    if history_before > 0 and input_csv and input_csv.is_file() and ctx_ts is not None:
        hist_ts, hist_val = load_history_before_json_context(
            sensor_name,
            input_csv,
            body,
            history_before=history_before,
            smooth_window=smooth_window,
        )

    n_hist = len(hist_val) if hist_val is not None else 0
    n_ctx = len(ctx_val) if ctx_val is not None else 0
    n_fc = len(pred)
    i_ctx = n_hist
    i_fc = i_ctx + n_ctx

    fig, ax = plt.subplots(figsize=(14, 4.5))

    # History + input: sample index (no calendar time — avoids huge gaps from outages).
    if hist_val is not None and n_hist > 0:
        x_hist = np.arange(0, n_hist, dtype=float)
        ax.plot(
            x_hist,
            hist_val.to_numpy() if hasattr(hist_val, "to_numpy") else hist_val,
            color="#6baed6",
            lw=1.5,
            marker="o",
            markersize=2,
            alpha=0.9,
            label=f"history before input (n={n_hist})",
        )

    if ctx_val is not None and n_ctx > 0:
        x_ctx = np.arange(i_ctx, i_ctx + n_ctx, dtype=float)
        y_ctx = np.asarray(ctx_val, dtype=float).reshape(-1)
        ax.plot(
            x_ctx,
            y_ctx,
            color="#2ca02c",
            lw=2.0,
            marker="o",
            markersize=3,
            label=f"model input window (n={n_ctx})",
        )
        if n_fc > 0:
            jump = float(pred[0]) - float(y_ctx[-1])
            if abs(jump) > 0.05 * max(abs(float(y_ctx[-1])), 1e-6):
                ax.annotate(
                    f"Δ={jump:+.2f} at forecast start",
                    xy=(i_fc - 0.5, (y_ctx[-1] + pred[0]) / 2),
                    fontsize=7,
                    color="0.45",
                )

    if n_ctx > 0 or n_fc > 0:
        ax.axvline(
            i_fc - 0.5,
            color="0.35",
            ls=":",
            lw=1.2,
            label="forecast start",
        )

    x_fc = np.arange(i_fc, i_fc + n_fc, dtype=float)
    if quantiles:
        keys = _ordered_quantile_keys(quantiles, fq)
        if len(keys) >= 2:
            low_k, high_k = keys[0], keys[-1]
            ax.fill_between(
                x_fc,
                quantiles[low_k],
                quantiles[high_k],
                alpha=0.25,
                color="steelblue",
                label=f"forecast band q{low_k}–q{high_k}",
            )
            for k in keys:
                if k in (low_k, high_k):
                    continue
                ax.plot(x_fc, quantiles[k], lw=1.0, ls="--", alpha=0.7, label=f"forecast q{k}")

    ax.plot(
        x_fc,
        pred,
        color="darkorange",
        lw=2.0,
        marker="o",
        markersize=3,
        label=f"forecast median (n={n_fc})",
    )

    ax.set_xlabel("Sample index (history + model input)")
    if n_fc > 0:
        ax_top = ax.twiny()
        ax_top.set_xlim(ax.get_xlim())
        n_ticks = min(8, n_fc)
        tick_pos = np.linspace(i_fc, i_fc + n_fc - 1, n_ticks, dtype=int)
        ax_top.set_xticks(tick_pos)
        ax_top.set_xticklabels(
            [
                pd.Timestamp(ts_fc[int(pos - i_fc)]).strftime("%m-%d %H:%M")
                for pos in tick_pos
            ],
            fontsize=8,
            rotation=25,
            ha="left",
        )
        ax_top.set_xlabel("Forecast timestamp")
    ax.set_ylabel("Acceleration RMS (smoothed in model)")
    if n_hist and n_ctx:
        title = f"{sensor_name} — history {n_hist} + input {n_ctx} + forecast {n_fc}"
    elif n_ctx:
        title = f"{sensor_name} — input {n_ctx} + forecast {n_fc}"
    else:
        title = f"{sensor_name} — {n_fc}-step forecast"
    if title_suffix:
        title += f" ({title_suffix})"
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_all_sensors_from_json(
    predictions_path: Path,
    plots_dir: Path,
    *,
    input_csv: Optional[Path] = None,
    models_dir: Optional[Path] = None,
    smooth_window: int = 48,
    max_gap_seconds: float = 36000.0,
    history_before: int = 0,
) -> Tuple[int, int, List[str]]:
    """
    Read *predictions_path* and write one PNG per successful sensor.

    If context is missing from JSON, pass *input_csv* and *models_dir* to rebuild it.
    """
    with open(predictions_path, encoding="utf-8") as fp:
        payload = json.load(fp)

    plotted = 0
    skipped = 0
    reasons: List[str] = []

    for sensor_name, body in payload.items():
        canon = normalize_sensor_desc(sensor_name)
        if body.get("success") is False:
            skipped += 1
            reasons.append(f"{canon}: {body.get('error', 'failed')}")
            continue
        if "timestamps" not in body or "predicted" not in body:
            skipped += 1
            reasons.append(f"{canon}: missing timestamps/predicted")
            continue

        plot_body = body
        if not body.get("context_timestamps") and input_csv and models_dir:
            try:
                plot_body = enrich_context_from_csv(
                    body,
                    canon,
                    input_csv,
                    smooth_window=smooth_window,
                    max_gap_seconds=max_gap_seconds,
                    checkpoint=models_dir,
                )
            except Exception as exc:
                reasons.append(f"{canon}: context enrich failed ({exc})")

        out_path = plots_dir / f"{_slug(canon)}_forecast.png"
        try:
            plot_sensor_forecast(
                canon,
                plot_body,
                out_path,
                history_before=history_before,
                input_csv=input_csv,
                models_dir=models_dir,
                smooth_window=smooth_window,
            )
            plotted += 1
            print(f"  plot -> {out_path.name}")
        except Exception as exc:
            skipped += 1
            reasons.append(f"{canon}: plot failed ({exc})")

    return plotted, skipped, reasons


def plot_combined_overview(
    predictions_path: Path,
    out_path: Path,
    *,
    max_sensors: int = 8,
) -> None:
    """Grid of small multiples: context + forecast per sensor."""
    with open(predictions_path, encoding="utf-8") as fp:
        payload = json.load(fp)

    ok_items = [
        (normalize_sensor_desc(k), v)
        for k, v in payload.items()
        if v.get("success") is not False and "predicted" in v
    ][:max_sensors]
    if not ok_items:
        return

    n = len(ok_items)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.2 * nrows), squeeze=False)

    for idx, (name, body) in enumerate(ok_items):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        ctx_ts, ctx_val = _context_series(body)
        n_ctx = len(ctx_val) if ctx_val is not None else 0
        i_fc = n_ctx
        if ctx_val is not None and n_ctx > 0:
            ax.plot(np.arange(n_ctx), ctx_val, color="#2ca02c", lw=1.2)
            ax.axvline(i_fc - 0.5, color="0.4", ls=":", lw=0.8)
        ts_fc = pd.to_datetime(body["timestamps"])
        pred = body["predicted"]
        n_fc = len(pred)
        x_fc = np.arange(i_fc, i_fc + n_fc)
        quantiles = body.get("quantiles") or {}
        fq = body.get("forecast_quantiles")
        if quantiles:
            keys = _ordered_quantile_keys(quantiles, fq)
            if len(keys) >= 2:
                ax.fill_between(
                    x_fc,
                    quantiles[keys[0]],
                    quantiles[keys[-1]],
                    alpha=0.25,
                    color="steelblue",
                )
        ax.plot(x_fc, pred, color="darkorange", lw=1.5)
        short = re.sub(r"^AHU 2-9 ", "", name)
        ax.set_title(short, fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.tick_params(axis="x", rotation=25, labelsize=7)

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    fig.suptitle("All sensors — input window + forecast", fontsize=11)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  overview -> {out_path.name}")
