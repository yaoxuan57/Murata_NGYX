"""Plot forecast JSON for AHU 2-9 Blower DE A."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import CHECKPOINT_STEM, SENSOR_DESC, VALUE_COLUMN
from inference import build_smoothed_context, load_context_rows
from io_utils import parse_timestamp_series, read_vibration_export_csv


def _ordered_quantile_keys(
    quantiles: Dict[str, List[float]],
    forecast_quantiles: Optional[List[float]],
) -> List[str]:
    if forecast_quantiles:
        keys = [f"{float(q):g}" for q in forecast_quantiles]
        return [k for k in keys if k in quantiles]
    return sorted(quantiles.keys(), key=lambda k: float(k))


def load_history_before_context(
    csv_path: Path,
    body: Dict[str, Any],
    *,
    history_before: int,
    smooth_window: int,
) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
    ctx_ts_raw = body.get("context_timestamps")
    if not ctx_ts_raw or history_before <= 0:
        return None, None

    ctx_ts = pd.to_datetime(ctx_ts_raw)
    df = read_vibration_export_csv(str(csv_path))
    df["SENSOR_DESC"] = df["SENSOR_DESC"].astype(str).str.strip()
    part = df[df["SENSOR_DESC"] == SENSOR_DESC].copy()
    if part.empty:
        return None, None

    ts = parse_timestamp_series(part["TIMESTAMP"], name="TIMESTAMP", strict=False)
    valid = ts.notna()
    part = part.loc[valid].copy()
    part["_ts"] = ts.loc[valid].values
    part = part.sort_values("_ts", kind="mergesort").reset_index(drop=True)
    part = part.drop_duplicates(subset=["TIMESTAMP"], keep="last").reset_index(drop=True)

    t0 = pd.Timestamp(ctx_ts[0])
    prior = part[part["_ts"] < t0].tail(int(history_before))
    if prior.empty:
        return None, None

    prior_ts = parse_timestamp_series(prior["TIMESTAMP"], name="TIMESTAMP")
    prior_df = prior.drop(columns=["_ts"], errors="ignore")
    smooth = build_smoothed_context(prior_df, prior_ts, smooth_window=smooth_window)
    return prior_ts.reset_index(drop=True), pd.Series(smooth, dtype=float)


def plot_forecast(
    body: Dict[str, Any],
    out_path: Path,
    *,
    input_csv: Optional[Path] = None,
    history_before: int = 100,
    smooth_window: int = 48,
) -> None:
    if body.get("success") is False:
        raise ValueError(body.get("error", "inference failed"))

    ts_fc = pd.to_datetime(body["timestamps"])
    pred = body["predicted"]
    quantiles = body.get("quantiles") or {}
    fq = body.get("forecast_quantiles")

    ctx_ts = pd.to_datetime(body.get("context_timestamps", []))
    ctx_val = body.get("context_values", [])

    hist_ts, hist_val = None, None
    if history_before > 0 and input_csv and input_csv.is_file() and len(ctx_ts):
        hist_ts, hist_val = load_history_before_context(
            input_csv,
            body,
            history_before=history_before,
            smooth_window=smooth_window,
        )

    n_hist = len(hist_val) if hist_val is not None else 0
    n_ctx = len(ctx_val)
    n_fc = len(pred)
    i_ctx = n_hist
    i_fc = i_ctx + n_ctx

    fig, ax = plt.subplots(figsize=(14, 4.5))

    if hist_val is not None and n_hist > 0:
        ax.plot(
            np.arange(0, n_hist),
            hist_val.to_numpy(),
            color="#6baed6",
            lw=1.5,
            marker="o",
            markersize=2,
            label=f"history (n={n_hist})",
        )

    if n_ctx > 0:
        x_ctx = np.arange(i_ctx, i_ctx + n_ctx)
        y_ctx = np.asarray(ctx_val, dtype=float)
        ax.plot(
            x_ctx,
            y_ctx,
            color="#2ca02c",
            lw=2.0,
            marker="o",
            markersize=3,
            label=f"model input (n={n_ctx})",
        )

    if n_ctx > 0 or n_fc > 0:
        ax.axvline(i_fc - 0.5, color="0.35", ls=":", lw=1.2, label="forecast start")

    x_fc = np.arange(i_fc, i_fc + n_fc)
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
                label=f"band q{low_k}-q{high_k}",
            )

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
            [pd.Timestamp(ts_fc[int(pos - i_fc)]).strftime("%m-%d %H:%M") for pos in tick_pos],
            fontsize=8,
            rotation=25,
            ha="left",
        )
        ax_top.set_xlabel("Forecast timestamp")

    model_tag = body.get("model_type", "model")
    ax.set_ylabel(f"{VALUE_COLUMN} (smoothed)")
    ax.set_title(f"{SENSOR_DESC} — {model_tag} — input {n_ctx} + forecast {n_fc}")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
