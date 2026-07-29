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


def _forecast_plot_title(
    sensor_name: str,
    body: Dict[str, Any],
    *,
    n_hist: int,
    n_ctx: int,
    n_pred: int,
    title_suffix: str = "",
) -> str:
    if n_hist > 0:
        title = f"{sensor_name} — history {n_hist} + input {n_ctx} + forecast {n_pred}"
    else:
        title = f"{sensor_name} — input {n_ctx} + forecast {n_pred}"
    if title_suffix:
        title += f" ({title_suffix})"
    return title


def _resolve_plot_format(out_path: Path, plot_format: Optional[str]) -> str:
    if plot_format:
        fmt = plot_format.lower()
        if fmt not in ("html", "png"):
            raise ValueError(f"plot_format must be 'html' or 'png', got {plot_format!r}")
        return fmt
    return "html" if out_path.suffix.lower() in (".html", ".htm") else "png"


def _plot_output_path(base: Path, plot_format: str) -> Path:
    ext = ".html" if plot_format == "html" else ".png"
    return base.with_suffix(ext)


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


def load_actual_values_at_times(
    csv_path: Path,
    sensor_name: str,
    target_ts: pd.Series | pd.DatetimeIndex,
    *,
    smooth_window: int = 48,
    tolerance_minutes: float = 20.0,
) -> pd.Series:
    """Causal-smoothed RMS from *csv_path* aligned to each timestamp in *target_ts*."""
    from io_utils import parse_timestamp_series, read_vibration_export_csv
    from model_utils import smooth_target_series_1d
    from windowing import VALUE_COLUMN

    canon = normalize_sensor_desc(sensor_name)
    df = read_vibration_export_csv(str(csv_path))
    df["SENSOR_DESC"] = df["SENSOR_DESC"].map(normalize_sensor_desc)
    part = df[df["SENSOR_DESC"] == canon].copy()
    targets = pd.to_datetime(target_ts)
    if part.empty:
        return pd.Series(np.nan, index=np.arange(len(targets)), dtype=float)

    ts = parse_timestamp_series(part["TIMESTAMP"], name="TIMESTAMP", strict=False)
    valid = ts.notna()
    part = part.loc[valid].copy()
    part["_ts"] = ts.loc[valid].values
    part = part.sort_values("_ts", kind="mergesort").reset_index(drop=True)
    part = part.drop_duplicates(subset=["_ts"], keep="last").reset_index(drop=True)

    rms = pd.to_numeric(part[VALUE_COLUMN], errors="coerce").to_numpy(dtype=np.float64)
    smooth = smooth_target_series_1d(rms, smooth_window)
    series = pd.DataFrame({"ts": part["_ts"].values, "value": smooth.astype(float)})

    query = pd.DataFrame({"ts": targets})
    merged = pd.merge_asof(
        query.sort_values("ts"),
        series.sort_values("ts"),
        on="ts",
        direction="nearest",
        tolerance=pd.Timedelta(minutes=float(tolerance_minutes)),
    )
    return merged["value"].reset_index(drop=True)


def load_actual_series_in_forecast_range(
    csv_path: Path,
    sensor_name: str,
    forecast_start: pd.Timestamp,
    forecast_end: pd.Timestamp,
    *,
    smooth_window: int = 48,
) -> Tuple[pd.Series, pd.Series]:
    """All causal-smoothed readings from CSV within [forecast_start, forecast_end]."""
    from io_utils import parse_timestamp_series, read_vibration_export_csv
    from model_utils import smooth_target_series_1d
    from windowing import VALUE_COLUMN

    canon = normalize_sensor_desc(sensor_name)
    df = read_vibration_export_csv(str(csv_path))
    df["SENSOR_DESC"] = df["SENSOR_DESC"].map(normalize_sensor_desc)
    part = df[df["SENSOR_DESC"] == canon].copy()
    if part.empty:
        return pd.Series(dtype="datetime64[ns]"), pd.Series(dtype=float)

    ts = parse_timestamp_series(part["TIMESTAMP"], name="TIMESTAMP", strict=False)
    valid = ts.notna()
    part = part.loc[valid].copy()
    part["_ts"] = ts.loc[valid].values
    part = part.sort_values("_ts", kind="mergesort").reset_index(drop=True)
    part = part.drop_duplicates(subset=["_ts"], keep="last").reset_index(drop=True)

    rms = pd.to_numeric(part[VALUE_COLUMN], errors="coerce").to_numpy(dtype=np.float64)
    smooth = smooth_target_series_1d(rms, smooth_window)
    series_ts = pd.Series(part["_ts"].values)
    series_val = pd.Series(smooth.astype(float))

    t0 = pd.Timestamp(forecast_start)
    t1 = pd.Timestamp(forecast_end)
    mask = (series_ts >= t0) & (series_ts <= t1)
    return series_ts.loc[mask].reset_index(drop=True), series_val.loc[mask].reset_index(drop=True)


def build_sensor_smoothed_timeline(
    csv_path: Path,
    sensor_name: str,
    *,
    smooth_window: int = 48,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (*part*, *timeline*) where *part* is the sorted sensor export rows and
    *timeline* has ``ts`` and ``value`` (causal-smoothed RMS) aligned row-for-row.
    """
    from io_utils import parse_timestamp_series, read_vibration_export_csv
    from model_utils import smooth_target_series_1d
    from windowing import VALUE_COLUMN

    canon = normalize_sensor_desc(sensor_name)
    df = read_vibration_export_csv(str(csv_path))
    df["SENSOR_DESC"] = df["SENSOR_DESC"].map(normalize_sensor_desc)
    part = df[df["SENSOR_DESC"] == canon].copy()
    if part.empty:
        return part, pd.DataFrame(columns=["ts", "value"])

    ts = parse_timestamp_series(part["TIMESTAMP"], name="TIMESTAMP", strict=False)
    valid = ts.notna()
    part = part.loc[valid].copy()
    part["_ts"] = ts.loc[valid].values
    part = part.sort_values("_ts", kind="mergesort").reset_index(drop=True)
    part = part.drop_duplicates(subset=["_ts"], keep="last").reset_index(drop=True)

    rms = pd.to_numeric(part[VALUE_COLUMN], errors="coerce").to_numpy(dtype=np.float64)
    smooth = smooth_target_series_1d(rms, smooth_window)
    timeline = pd.DataFrame({"ts": part["_ts"].values, "value": smooth.astype(float)})
    return part, timeline


def _render_input_forecast_actual_png(
    sensor_name: str,
    body: Dict[str, Any],
    out_path: Path,
    *,
    hist_ts: Optional[pd.Series] = None,
    hist_val: Optional[pd.Series] = None,
    act_ts: Optional[pd.Series] = None,
    act_val: Optional[pd.Series] = None,
    title_suffix: str = "",
) -> int:
    """Draw and save one PNG; returns count of actual points plotted in forecast window."""
    import matplotlib.dates as mdates

    ctx_ts, ctx_val = _context_series(body)
    if ctx_ts is None or ctx_val is None:
        raise ValueError("JSON missing context_timestamps / context_values")

    ts_fc = pd.to_datetime(body["timestamps"])
    pred = np.asarray(body["predicted"], dtype=float)
    quantiles = body.get("quantiles") or {}
    fq = body.get("forecast_quantiles")

    n_matched = int(len(act_ts)) if act_ts is not None and len(act_ts) else 0

    fig, ax = plt.subplots(figsize=(14, 5))

    if hist_val is not None and len(hist_val) > 0:
        ax.plot(
            hist_ts,
            hist_val,
            color="#6baed6",
            lw=1.5,
            marker="o",
            markersize=2,
            alpha=0.9,
            label=f"history before input (n={len(hist_val)})",
        )

    ax.plot(
        ctx_ts,
        ctx_val,
        color="#2ca02c",
        lw=2.0,
        marker="o",
        markersize=3,
        label=f"model input window (n={len(ctx_val)})",
    )

    ax.axvline(
        pd.Timestamp(ctx_ts.iloc[-1]),
        color="0.35",
        ls=":",
        lw=1.4,
        label="forecast start",
    )

    if quantiles:
        keys = _ordered_quantile_keys(quantiles, fq)
        if len(keys) >= 2:
            low_k, high_k = keys[0], keys[-1]
            ax.fill_between(
                ts_fc,
                quantiles[low_k],
                quantiles[high_k],
                alpha=0.22,
                color="steelblue",
                label=f"predicted band q{low_k}–q{high_k}",
            )

    ax.plot(
        ts_fc,
        pred,
        color="darkorange",
        lw=2.0,
        marker="o",
        markersize=3,
        label=f"predicted median (n={len(pred)})",
    )

    if act_ts is not None and len(act_ts) > 0:
        ax.plot(
            act_ts,
            act_val,
            color="#1f77b4",
            lw=2.0,
            marker="s",
            markersize=4,
            linestyle="--",
            label=f"actual in forecast window (n={n_matched})",
        )
    else:
        ax.text(
            0.02,
            0.02,
            "No actual points in forecast window (CSV ends before forecast period)",
            transform=ax.transAxes,
            fontsize=8,
            color="0.45",
        )

    n_hist = len(hist_val) if hist_val is not None and len(hist_val) > 0 else 0
    ax.set_title(
        _forecast_plot_title(
            sensor_name,
            body,
            n_hist=n_hist,
            n_ctx=len(ctx_val),
            n_pred=len(pred),
            title_suffix=title_suffix,
        )
    )
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Acceleration RMS (causal smoothed)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    fig.autofmt_xdate(rotation=25, ha="right")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return n_matched


def _render_input_forecast_actual_html(
    sensor_name: str,
    body: Dict[str, Any],
    out_path: Path,
    *,
    hist_ts: Optional[pd.Series] = None,
    hist_val: Optional[pd.Series] = None,
    act_ts: Optional[pd.Series] = None,
    act_val: Optional[pd.Series] = None,
    title_suffix: str = "",
) -> int:
    """Draw and save one interactive HTML chart (zoom/pan); returns actual point count."""
    import plotly.graph_objects as go

    ctx_ts, ctx_val = _context_series(body)
    if ctx_ts is None or ctx_val is None:
        raise ValueError("JSON missing context_timestamps / context_values")

    ts_fc = pd.to_datetime(body["timestamps"])
    pred = np.asarray(body["predicted"], dtype=float)
    quantiles = body.get("quantiles") or {}
    fq = body.get("forecast_quantiles")
    n_matched = int(len(act_ts)) if act_ts is not None and len(act_ts) else 0

    fig = go.Figure()

    if hist_val is not None and len(hist_val) > 0:
        fig.add_trace(
            go.Scatter(
                x=hist_ts,
                y=hist_val,
                mode="lines+markers",
                name=f"history before input (n={len(hist_val)})",
                line=dict(color="#6baed6", width=1.5),
                marker=dict(size=4),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=ctx_ts,
            y=ctx_val,
            mode="lines+markers",
            name=f"model input window (n={len(ctx_val)})",
            line=dict(color="#2ca02c", width=2),
            marker=dict(size=5),
        )
    )

    forecast_start = pd.Timestamp(ctx_ts.iloc[-1]).to_pydatetime()
    fig.add_vline(
        x=forecast_start,
        line=dict(color="rgba(80,80,80,0.8)", width=1.4, dash="dot"),
    )
    fig.add_annotation(
        x=forecast_start,
        y=1.0,
        yref="paper",
        text="forecast start",
        showarrow=False,
        yanchor="bottom",
        font=dict(size=11, color="rgba(80,80,80,0.9)"),
    )

    if quantiles:
        keys = _ordered_quantile_keys(quantiles, fq)
        if len(keys) >= 2:
            low_k, high_k = keys[0], keys[-1]
            fig.add_trace(
                go.Scatter(
                    x=ts_fc,
                    y=quantiles[high_k],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=ts_fc,
                    y=quantiles[low_k],
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor="rgba(70,130,180,0.22)",
                    name=f"predicted band q{low_k}–q{high_k}",
                    hoverinfo="skip",
                )
            )

    fig.add_trace(
        go.Scatter(
            x=ts_fc,
            y=pred,
            mode="lines+markers",
            name=f"predicted median (n={len(pred)})",
            line=dict(color="darkorange", width=2),
            marker=dict(size=5),
        )
    )

    if act_ts is not None and len(act_ts) > 0:
        fig.add_trace(
            go.Scatter(
                x=act_ts,
                y=act_val,
                mode="lines+markers",
                name=f"actual in forecast window (n={n_matched})",
                line=dict(color="#1f77b4", width=2, dash="dash"),
                marker=dict(size=6, symbol="square"),
            )
        )

    n_hist = len(hist_val) if hist_val is not None and len(hist_val) > 0 else 0
    fig.update_layout(
        title=_forecast_plot_title(
            sensor_name,
            body,
            n_hist=n_hist,
            n_ctx=len(ctx_val),
            n_pred=len(pred),
            title_suffix=title_suffix,
        ),
        xaxis_title="Timestamp",
        yaxis_title="Acceleration RMS (causal smoothed)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=30, t=80, b=60),
    )
    if act_ts is None or len(act_ts) == 0:
        fig.add_annotation(
            text="No actual points in forecast window (CSV ends before forecast period)",
            xref="paper",
            yref="paper",
            x=0.02,
            y=0.02,
            showarrow=False,
            font=dict(size=10, color="gray"),
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn", full_html=True)
    return n_matched


def _render_input_forecast_actual_figure(
    sensor_name: str,
    body: Dict[str, Any],
    out_path: Path,
    *,
    hist_ts: Optional[pd.Series] = None,
    hist_val: Optional[pd.Series] = None,
    act_ts: Optional[pd.Series] = None,
    act_val: Optional[pd.Series] = None,
    title_suffix: str = "",
    plot_format: Optional[str] = None,
) -> int:
    """Save input + forecast chart as interactive HTML (default) or PNG."""
    fmt = _resolve_plot_format(out_path, plot_format)
    out_path = _plot_output_path(out_path, fmt)
    renderer = _render_input_forecast_actual_html if fmt == "html" else _render_input_forecast_actual_png
    return renderer(
        sensor_name,
        body,
        out_path,
        hist_ts=hist_ts,
        hist_val=hist_val,
        act_ts=act_ts,
        act_val=act_val,
        title_suffix=title_suffix,
    )


def plot_sensor_input_forecast_actual(
    sensor_name: str,
    body: Dict[str, Any],
    out_path: Path,
    *,
    input_csv: Path,
    smooth_window: int = 48,
    history_before: int = 0,
    tolerance_minutes: float = 20.0,
    plot_format: str = "html",
) -> int:
    """One chart: optional history + 48-pt input + forecast + actuals from CSV."""
    hist_ts, hist_val = None, None
    if history_before > 0:
        hist_ts, hist_val = load_history_before_json_context(
            sensor_name,
            input_csv,
            body,
            history_before=history_before,
            smooth_window=smooth_window,
        )

    ts_fc = pd.to_datetime(body["timestamps"])
    act_ts, act_val = load_actual_series_in_forecast_range(
        input_csv,
        sensor_name,
        pd.Timestamp(ts_fc[0]),
        pd.Timestamp(ts_fc[-1]),
        smooth_window=smooth_window,
    )
    return _render_input_forecast_actual_figure(
        sensor_name,
        body,
        out_path,
        hist_ts=hist_ts,
        hist_val=hist_val,
        act_ts=act_ts,
        act_val=act_val,
        plot_format=plot_format,
    )


def plot_rolling_windows_from_csv(
    csv_path: Path,
    models_dir: Path,
    plots_dir: Path,
    *,
    rolling_windows: int = 5,
    history_before: int = 0,
    smooth_window: int = 48,
    device: str = "cpu",
    plot_format: str = "html",
) -> Tuple[int, int, List[str]]:
    """
    Run inference at *rolling_windows* evenly spaced anchors per sensor and plot each.

    Each plot: history + 48-pt input + 48-step forecast + actual holdout (when available).
    Writes under ``plots_dir/rolling/<sensor_slug>/roll_XXX.<html|png>``.
    """
    from inference import forecast_vibration
    from sensors import build_vibration_sensor_model_map

    sensor_map = build_vibration_sensor_model_map(models_dir)
    input_len = 48
    pred_len = 48
    min_rows = input_len + pred_len

    plotted = 0
    skipped = 0
    reasons: List[str] = []

    for sensor_id, entry in sensor_map.items():
        sensor_name = entry["sensorName"]
        canon = normalize_sensor_desc(sensor_name)
        part, timeline = build_sensor_smoothed_timeline(
            csv_path, canon, smooth_window=smooth_window
        )
        n = len(part)
        if n < min_rows:
            skipped += 1
            reasons.append(f"{canon}: need >={min_rows} rows for rolling plots, found {n}")
            continue

        # Context ends at index i-1; holdout is i .. i+pred_len-1.
        i_min = input_len
        i_max = n - pred_len
        anchors = np.linspace(i_min, i_max, int(rolling_windows), dtype=int)
        anchors = np.unique(anchors)

        sensor_dir = plots_dir / "rolling" / _slug(canon)
        sensor_plotted = 0

        for wi, i in enumerate(anchors):
            infer_df = part.iloc[i - input_len : i].drop(columns=["_ts"], errors="ignore").copy()
            try:
                results = forecast_vibration(
                    infer_df,
                    sensor_model_map={sensor_id: entry},
                    smooth_window=smooth_window,
                    device=device,
                )
            except Exception as exc:
                reasons.append(f"{canon} roll@{i}: inference failed ({exc})")
                continue

            body = results.get(sensor_id, {})
            if body.get("success") is not True:
                msg = body.get("warning") or body.get("error", "failed")
                reasons.append(f"{canon} roll@{i}: {msg}")
                continue

            h0 = max(0, i - input_len - history_before)
            h1 = i - input_len
            hist_ts = timeline["ts"].iloc[h0:h1].reset_index(drop=True)
            hist_val = timeline["value"].iloc[h0:h1].reset_index(drop=True)
            act_ts = timeline["ts"].iloc[i : i + pred_len].reset_index(drop=True)
            act_val = timeline["value"].iloc[i : i + pred_len].reset_index(drop=True)

            anchor_ts = pd.Timestamp(timeline["ts"].iloc[i - 1]).strftime("%Y%m%d_%H%M")
            out_path = _plot_output_path(
                sensor_dir / f"roll_{wi:03d}_{anchor_ts}",
                plot_format,
            )
            try:
                n_act = _render_input_forecast_actual_figure(
                    canon,
                    body,
                    out_path,
                    hist_ts=hist_ts,
                    hist_val=hist_val,
                    act_ts=act_ts,
                    act_val=act_val,
                    title_suffix=f"rolling window {wi + 1}/{len(anchors)}",
                    plot_format=plot_format,
                )
                sensor_plotted += 1
                plotted += 1
                print(f"  roll -> {out_path}  (actual n={n_act})")
            except Exception as exc:
                skipped += 1
                reasons.append(f"{canon} roll@{i}: plot failed ({exc})")

        if sensor_plotted == 0:
            skipped += 1

    return plotted, skipped, reasons


def plot_rolling_inference_results(
    rolling_results: Dict[str, Any],
    plots_dir: Path,
    *,
    input_csv: Path,
    smooth_window: int = 48,
    history_before: int = 0,
    plot_format: str = "html",
) -> Tuple[int, int, List[str]]:
    """
    Plot each successful window from :func:`inference.forecast_rolling_windows` output.

    Writes ``plots_dir/rolling/<sensor_slug>/roll_XXX_<anchor>.<html|png>``.
    """
    plotted = 0
    skipped = 0
    reasons: List[str] = []

    for sensor_id, sensor_block in rolling_results.items():
        windows = sensor_block.get("windows")
        if not windows:
            if sensor_block.get("success") is False:
                skipped += 1
                msg = sensor_block.get("warning") or sensor_block.get("error", "failed")
                reasons.append(f"{sensor_id}: {msg}")
            continue

        sensor_name = sensor_block.get("sensorName") or sensor_id
        canon = normalize_sensor_desc(sensor_name)
        part, timeline = build_sensor_smoothed_timeline(
            input_csv, canon, smooth_window=smooth_window
        )
        sensor_dir = plots_dir / "rolling" / _slug(canon)
        sensor_plotted = 0
        n_windows = len(windows)

        for body in windows:
            wi = int(body.get("window_index", 0))
            if body.get("success") is not True:
                msg = body.get("warning") or body.get("error", "failed")
                reasons.append(f"{canon} window {wi}: {msg}")
                continue

            anchor_i = body.get("anchor_index")
            hist_ts, hist_val = None, None
            act_ts, act_val = None, None

            if anchor_i is not None and not timeline.empty:
                anchor_i = int(anchor_i)
                pred_len = len(body.get("predicted") or [])
                h0 = max(0, anchor_i - 48 - history_before)
                h1 = anchor_i - 48
                if h1 > h0:
                    hist_ts = timeline["ts"].iloc[h0:h1].reset_index(drop=True)
                    hist_val = timeline["value"].iloc[h0:h1].reset_index(drop=True)
                act_ts = timeline["ts"].iloc[anchor_i : anchor_i + pred_len].reset_index(drop=True)
                act_val = timeline["value"].iloc[anchor_i : anchor_i + pred_len].reset_index(drop=True)
            elif body.get("actual_timestamps") and body.get("actual_values"):
                act_ts = pd.to_datetime(body["actual_timestamps"])
                act_val = pd.Series(body["actual_values"], dtype=float)

            anchor_label = "latest"
            if body.get("anchor_timestamp"):
                anchor_label = pd.Timestamp(body["anchor_timestamp"]).strftime("%Y%m%d_%H%M")
            elif anchor_i is not None and not timeline.empty:
                anchor_label = pd.Timestamp(timeline["ts"].iloc[int(anchor_i) - 1]).strftime(
                    "%Y%m%d_%H%M"
                )

            out_path = _plot_output_path(
                sensor_dir / f"roll_{wi:03d}_{anchor_label}",
                plot_format,
            )
            try:
                n_act = _render_input_forecast_actual_figure(
                    canon,
                    body,
                    out_path,
                    hist_ts=hist_ts,
                    hist_val=hist_val,
                    act_ts=act_ts,
                    act_val=act_val,
                    title_suffix=f"window {wi + 1}/{n_windows}",
                    plot_format=plot_format,
                )
                sensor_plotted += 1
                plotted += 1
                print(f"  plot -> {out_path}  (actual n={n_act})")
            except Exception as exc:
                skipped += 1
                reasons.append(f"{canon} window {wi}: plot failed ({exc})")

        if sensor_plotted == 0 and windows:
            skipped += 1

    return plotted, skipped, reasons


def plot_all_sensors_forecast_vs_actual(
    predictions_path: Path,
    plots_dir: Path,
    *,
    input_csv: Path,
    smooth_window: int = 48,
    history_before: int = 0,
    tolerance_minutes: float = 20.0,
    plot_format: str = "html",
) -> Tuple[int, int, List[str]]:
    """Write one latest-forecast chart per successful sensor (under ``plots_dir/latest/``)."""
    with open(predictions_path, encoding="utf-8") as fp:
        payload = json.load(fp)

    if not input_csv.is_file():
        raise FileNotFoundError(f"Data CSV not found: {input_csv}")

    plotted = 0
    skipped = 0
    reasons: List[str] = []

    for _key, body in payload.items():
        sensor_name = body.get("sensorName") or _key
        canon = normalize_sensor_desc(sensor_name)
        if body.get("success") is False:
            skipped += 1
            reasons.append(f"{canon}: {body.get('warning') or body.get('error', 'failed')}")
            continue
        required = ("timestamps", "predicted", "context_timestamps", "context_values")
        if any(f not in body for f in required):
            skipped += 1
            reasons.append(f"{canon}: missing fields for plotting")
            continue

        out_path = _plot_output_path(
            plots_dir / "latest" / f"{_slug(canon)}_input_forecast_actual",
            plot_format,
        )
        try:
            n_act = plot_sensor_input_forecast_actual(
                canon,
                body,
                out_path,
                input_csv=input_csv,
                smooth_window=smooth_window,
                history_before=history_before,
                tolerance_minutes=tolerance_minutes,
                plot_format=plot_format,
            )
            plotted += 1
            print(f"  plot -> {out_path.name}  (actual points in forecast window: {n_act})")
        except Exception as exc:
            skipped += 1
            reasons.append(f"{canon}: plot failed ({exc})")

    return plotted, skipped, reasons
