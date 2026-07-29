#!/usr/bin/env python3
"""Compare base vs finetuned transformer forecasts on each sensor's test split.

For every ``finetune/data_<STEM>/splits/`` folder:
  - Base model: highest ``model_to_finetune/<STEM>_v*.pth``
  - Finetuned model: ``finetune/finetuned_models/<STEM>_ft.pth`` (fallback: training output dirs)

Builds rolling windows on train+val+test (input may use history; forecast horizon must lie in test.csv),
runs both checkpoints, writes stitched Plotly HTML like training, plus a side-by-side overlay.

Usage (repo root):
  python finetune/compare_base_vs_finetuned_test.py
  python finetune/compare_base_vs_finetuned_test.py --stems AHU_2_9_Blower_DE_A AHU_2_9_Blower_NDE_A
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from forecast_sweep_common import (  # noqa: E402
    MultiStepDeltaDataset,
    collect_predictions,
    compute_timestep_window_start_indices,
    mape_np,
    parse_timestamp_series,
    prediction_interval_band_label,
    save_stitched_test_forecast_html,
    smooth_target_series_1d,
    subsample_window_starts,
)
from train_transformer_sweep import make_model  # noqa: E402

FINETUNE_DIR = Path(__file__).resolve().parent
BASE_MODEL_DIR = FINETUNE_DIR / "model_to_finetune"
FINETUNED_MODEL_DIR = FINETUNE_DIR / "finetuned_models"
VALUE_COL = "Acceleration RMS"
DEFAULT_FORECAST_QUANTILES = [0.05, 0.5, 0.95]


def sensor_short(stem: str) -> str:
    text = re.sub(r"^AHU_[0-9]+_[0-9]+_", "", stem)
    return re.sub(r"_+", "_", text).lower()


def discover_sensor_stems() -> List[str]:
    stems = []
    for path in sorted(FINETUNE_DIR.glob("data_*")):
        if (path / "splits" / "test.csv").is_file():
            stems.append(path.name.replace("data_", "", 1))
    return stems


def resolve_highest_version_pth(directory: Path, stem: str) -> Optional[Path]:
    best: Optional[Path] = None
    best_n = -1
    if not directory.is_dir():
        return None
    for p in directory.glob(f"{stem}_v*.pth"):
        m = re.search(r"_v(\d+)\.pth$", p.name)
        if not m:
            continue
        n = int(m.group(1))
        if n > best_n:
            best_n = n
            best = p
    return best


def resolve_base_checkpoint(directory: Path, stem: str) -> Optional[Path]:
    """Resolve base checkpoint: highest _vN, else <stem>_merged.pth (with 4-4 alias)."""
    best = resolve_highest_version_pth(directory, stem)
    if best is not None:
        return best
    if not directory.is_dir():
        return None
    candidates = [
        directory / f"{stem}_merged.pth",
        directory / f"{stem.replace('4_4', '4-4')}_merged.pth",
        directory / f"{stem.replace('2_9', '2-9')}_merged.pth",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def resolve_finetuned_pth(stem: str, finetuned_dir: Optional[Path] = None) -> Optional[Path]:
    # Preferred handoff location: finetune/finetuned_models/<STEM>_ft.pth
    handoff = (finetuned_dir or FINETUNED_MODEL_DIR) / f"{stem}_ft.pth"
    if handoff.is_file():
        return handoff

    short = sensor_short(stem)
    pattern = f"outputs_transformer_finetune_finetune_{short}_jan2jun_hist_jun_chrono*/transformer_finetune_{short}_best.pth"
    matches = sorted(FINETUNE_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def load_checkpoint_bundle(checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    required = {"model_state_dict", "train_mean", "train_std", "input_len", "pred_len", "model_config"}
    missing = required - set(ckpt.keys())
    if missing:
        raise ValueError(f"{checkpoint_path} missing keys: {sorted(missing)}")

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
    return model, ckpt, args, input_len, pred_len


def load_split_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if VALUE_COL not in df.columns and "DATA12" in df.columns:
        df[VALUE_COL] = pd.to_numeric(df["DATA12"], errors="coerce")
    if VALUE_COL not in df.columns:
        raise ValueError(f"{path}: missing {VALUE_COL!r}")
    df = df.copy()
    df["TIMESTAMP"] = parse_timestamp_series(df["TIMESTAMP"], str(path))
    return df.sort_values("TIMESTAMP", kind="mergesort").reset_index(drop=True)


def apply_target_smoothing(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Match finetune sbatch: causal MA on value column per CSV before windowing."""
    if window <= 1:
        return df
    out = df.copy()
    out[VALUE_COL] = smooth_target_series_1d(out[VALUE_COL].to_numpy(dtype=np.float32), window)
    return out


def load_splits_smoothed(splits_dir: Path, target_smoothing_window: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_train = apply_target_smoothing(load_split_csv(splits_dir / "train.csv"), target_smoothing_window)
    df_val = apply_target_smoothing(load_split_csv(splits_dir / "val.csv"), target_smoothing_window)
    df_test = apply_target_smoothing(load_split_csv(splits_dir / "test.csv"), target_smoothing_window)
    return df_train, df_val, df_test


def build_full_series_from_parts(
    df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame
) -> Tuple[pd.DataFrame, np.ndarray]:
    test_ts = set(pd.to_datetime(df_test["TIMESTAMP"]))
    df_full = pd.concat([df_train, df_val, df_test], ignore_index=True)
    df_full = df_full.sort_values("TIMESTAMP", kind="mergesort")
    df_full = df_full.drop_duplicates(subset=["TIMESTAMP"], keep="last").reset_index(drop=True)
    forecast_in_test = df_full["TIMESTAMP"].isin(test_ts).to_numpy(dtype=bool)
    return df_full, forecast_in_test


def build_split_windows(
    df_split: pd.DataFrame,
    *,
    input_len: int,
    pred_len: int,
    require_uniform_timestep: bool,
    uniform_step_seconds: float,
    uniform_tol_seconds: float,
    max_gap_seconds: Optional[float],
    window_stride: int,
    split_name: str,
) -> np.ndarray:
    """Sliding windows entirely inside one split CSV (train, val, or test)."""
    span = input_len + pred_len
    n = len(df_split)
    if n < span:
        return np.array([], dtype=np.int64)
    starts = np.arange(0, n - span + 1, dtype=np.int64)
    if require_uniform_timestep or max_gap_seconds is not None:
        allowed = compute_timestep_window_start_indices(
            df_split["TIMESTAMP"],
            span,
            nominal_seconds=uniform_step_seconds if require_uniform_timestep else None,
            tolerance_seconds=uniform_tol_seconds,
            max_consecutive_gap_seconds=max_gap_seconds,
        )
        allowed_set = set(int(x) for x in allowed)
        starts = np.asarray([i for i in starts if int(i) in allowed_set], dtype=np.int64)
    return subsample_window_starts(starts, window_stride, split_name=split_name)


def build_test_only_windows(
    df_test: pd.DataFrame,
    *,
    input_len: int,
    pred_len: int,
    require_uniform_timestep: bool,
    uniform_step_seconds: float,
    uniform_tol_seconds: float,
    max_gap_seconds: Optional[float],
    test_stride: int,
) -> np.ndarray:
    """Match finetune training eval: sliding windows entirely inside test.csv."""
    return build_split_windows(
        df_test,
        input_len=input_len,
        pred_len=pred_len,
        require_uniform_timestep=require_uniform_timestep,
        uniform_step_seconds=uniform_step_seconds,
        uniform_tol_seconds=uniform_tol_seconds,
        max_gap_seconds=max_gap_seconds,
        window_stride=test_stride,
        split_name="test",
    )


def prepare_eval_frames(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    *,
    eval_mode: str,
    input_len: int,
    pred_len: int,
    require_uniform_timestep: bool,
    uniform_step_seconds: float,
    uniform_tol_seconds: float,
    max_gap_seconds: Optional[float],
    test_stride: int,
    target_smoothing_window: int,
) -> Tuple[pd.DataFrame, np.ndarray, str]:
    """Return (series dataframe for model+plot, window starts, mode label)."""
    smooth_note = f", target-smoothing-window={target_smoothing_window}"
    if eval_mode == "test_only":
        starts = build_test_only_windows(
            df_test,
            input_len=input_len,
            pred_len=pred_len,
            require_uniform_timestep=require_uniform_timestep,
            uniform_step_seconds=uniform_step_seconds,
            uniform_tol_seconds=uniform_tol_seconds,
            max_gap_seconds=max_gap_seconds,
            test_stride=test_stride,
        )
        return (
            df_test,
            starts,
            f"test.csv only ({len(df_test)} rows{smooth_note})",
        )

    df_full, forecast_in_test = build_full_series_from_parts(df_train, df_val, df_test)
    starts = test_horizon_window_starts(
        forecast_in_test,
        input_len=input_len,
        pred_len=pred_len,
        require_uniform_timestep=require_uniform_timestep,
        uniform_step_seconds=uniform_step_seconds,
        uniform_tol_seconds=uniform_tol_seconds,
        max_gap_seconds=max_gap_seconds,
        timestamps=df_full["TIMESTAMP"],
        test_stride=test_stride,
    )
    return df_full, starts, f"train+val+test ({len(df_full)} rows{smooth_note})"


def test_horizon_window_starts(
    forecast_in_test: np.ndarray,
    *,
    input_len: int,
    pred_len: int,
    require_uniform_timestep: bool,
    uniform_step_seconds: float,
    uniform_tol_seconds: float,
    max_gap_seconds: Optional[float],
    timestamps: pd.Series,
    test_stride: int,
) -> np.ndarray:
    n = len(forecast_in_test)
    span = input_len + pred_len
    candidates: List[int] = []
    for i in range(0, n - span + 1):
        if forecast_in_test[i + input_len : i + span].all():
            candidates.append(i)
    if not candidates:
        return np.array([], dtype=np.int64)

    starts = np.asarray(candidates, dtype=np.int64)
    if require_uniform_timestep or max_gap_seconds is not None:
        allowed = compute_timestep_window_start_indices(
            timestamps,
            span,
            nominal_seconds=uniform_step_seconds if require_uniform_timestep else None,
            tolerance_seconds=uniform_tol_seconds,
            max_consecutive_gap_seconds=max_gap_seconds,
        )
        allowed_set = set(int(x) for x in allowed)
        starts = np.asarray([i for i in starts if int(i) in allowed_set], dtype=np.int64)

    return subsample_window_starts(starts, test_stride, split_name="test")


def run_model_on_test_windows(
    checkpoint_path: Path,
    df_eval: pd.DataFrame,
    window_starts: np.ndarray,
    *,
    norm_df: pd.DataFrame,
    device: torch.device,
    batch_size: int,
) -> Dict[str, Any]:
    model, ckpt, args, input_len, pred_len = load_checkpoint_bundle(checkpoint_path, device)
    feature_cols = [VALUE_COL]
    series = df_eval[VALUE_COL].to_numpy(dtype=np.float32)
    features = df_eval[feature_cols].to_numpy(dtype=np.float32)
    norm_features = norm_df[feature_cols].to_numpy(dtype=np.float32)

    train_mean = float(ckpt["train_mean"])
    train_std = float(ckpt["train_std"])
    feat_mean = norm_features.mean(axis=0).astype(np.float32)
    feat_std = norm_features.std(axis=0).astype(np.float32)
    feat_std[feat_std < 1e-8] = 1.0

    feat_norm = (features - feat_mean) / feat_std
    target_norm = (series - train_mean) / train_std

    if window_starts.size == 0:
        raise ValueError(f"No test windows for {checkpoint_path.name}")

    dataset = MultiStepDeltaDataset(
        feat_norm,
        target_norm,
        input_len=input_len,
        pred_len=pred_len,
        sample_starts=window_starts,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    fq = args.forecast_quantiles
    preds_raw, targets_raw, preds_q_raw = collect_predictions(
        model, loader, train_std, train_mean, device, forecast_quantiles=fq
    )
    return {
        "checkpoint": str(checkpoint_path),
        "input_len": input_len,
        "pred_len": pred_len,
        "forecast_quantiles": fq,
        "preds_raw": preds_raw,
        "targets_raw": targets_raw,
        "preds_quantiles_raw": preds_q_raw,
        "rmse": float(np.sqrt(np.mean((targets_raw - preds_raw) ** 2))),
        "mae": float(np.mean(np.abs(targets_raw - preds_raw))),
        "mape": mape_np(targets_raw, preds_raw),
    }


def stitch_forecast_frame(
    timestamps: pd.Series,
    window_starts: np.ndarray,
    preds_raw: np.ndarray,
    targets_raw: np.ndarray,
    input_len: int,
    pred_len: int,
    preds_quantiles_raw: Optional[np.ndarray],
    forecast_quantiles: Optional[List[float]],
) -> pd.DataFrame:
    ts_all = pd.to_datetime(timestamps, errors="coerce").reset_index(drop=True)
    rows = []
    for w, si in enumerate(window_starts):
        si = int(si)
        for h in range(pred_len):
            ri = si + input_len + h
            entry = {
                "ts": ts_all.iloc[ri],
                "actual": float(targets_raw[w, h]),
                "pred": float(preds_raw[w, h]),
                "lo": float("nan"),
                "hi": float("nan"),
            }
            if preds_quantiles_raw is not None and forecast_quantiles:
                pq = preds_quantiles_raw[w]
                if pq.ndim == 2 and pq.shape[0] >= 2:
                    entry["lo"] = float(pq[0, h])
                    entry["hi"] = float(pq[-1, h])
            rows.append(entry)
    fd = pd.DataFrame(rows)
    return fd.groupby("ts", sort=False, as_index=False).last().sort_values("ts").reset_index(drop=True)


def save_comparison_html(
    out_path: Path,
    *,
    stem: str,
    timestamps: pd.Series,
    window_starts: np.ndarray,
    input_len: int,
    pred_len: int,
    history_series: np.ndarray,
    base: Dict[str, Any],
    finetuned: Dict[str, Any],
) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        print(f"Skipping comparison HTML for {stem} ({exc})")
        return

    row0 = int(window_starts[0])
    ts_all = pd.to_datetime(timestamps, errors="coerce").reset_index(drop=True)
    x_in = ts_all.iloc[row0 : row0 + input_len]
    y_in = history_series[row0 : row0 + input_len]

    base_fd = stitch_forecast_frame(
        timestamps,
        window_starts,
        base["preds_raw"],
        base["targets_raw"],
        input_len,
        pred_len,
        base.get("preds_quantiles_raw"),
        base.get("forecast_quantiles"),
    )
    ft_fd = stitch_forecast_frame(
        timestamps,
        window_starts,
        finetuned["preds_raw"],
        finetuned["targets_raw"],
        input_len,
        pred_len,
        finetuned.get("preds_quantiles_raw"),
        finetuned.get("forecast_quantiles"),
    )

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            f"{stem} — base (before finetune) | RMSE={base['rmse']:.4f}",
            f"{stem} — finetuned | RMSE={finetuned['rmse']:.4f}",
        ),
    )

    for row, fd, label, color in (
        (1, base_fd, "Base median", "#ff7f0e"),
        (2, ft_fd, "Finetuned median", "#2ca02c"),
    ):
        fig.add_trace(
            go.Scatter(x=x_in, y=y_in, mode="lines", name="Input context", line=dict(color="gray"), showlegend=row == 1),
            row=row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=fd["ts"], y=fd["actual"], mode="lines", name="Actual", line=dict(color="#1f77b4")),
            row=row,
            col=1,
        )
        if np.any(np.isfinite(fd["lo"])) and np.any(np.isfinite(fd["hi"])):
            fig.add_trace(
                go.Scatter(
                    x=fd["ts"],
                    y=fd["hi"],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=fd["ts"],
                    y=fd["lo"],
                    mode="lines",
                    fill="tonexty",
                    fillcolor="rgba(255, 165, 0, 0.18)",
                    line=dict(width=0),
                    name="~90% band",
                    showlegend=row == 1,
                ),
                row=row,
                col=1,
            )
        fig.add_trace(
            go.Scatter(x=fd["ts"], y=fd["pred"], mode="lines", name=label, line=dict(color=color)),
            row=row,
            col=1,
        )

    fig.update_layout(
        title=f"Test-set comparison — {stem} (same windows, {pred_len}-step horizon)",
        template="plotly_white",
        height=820,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_xaxes(title_text="Timestamp", row=2, col=1)
    fig.update_yaxes(title_text=VALUE_COL, row=1, col=1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs=True)
    print(f"Saved comparison HTML: {out_path}")


def resolve_test_stride(arg_stride: int, pred_len: int) -> int:
    if arg_stride <= 0:
        return max(1, int(pred_len))
    return max(1, int(arg_stride))


def process_sensor(
    stem: str,
    *,
    out_dir: Path,
    device: torch.device,
    batch_size: int,
    require_uniform_timestep: bool,
    uniform_step_seconds: float,
    uniform_tol_seconds: float,
    max_gap_seconds: Optional[float],
    test_stride: int,
    eval_mode: str,
    eval_splits: List[str],
    target_smoothing_window: int,
    skip_missing_finetuned: bool,
    base_only: bool,
    finetuned_dir: Optional[Path] = None,
    base_checkpoint: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    splits_dir = FINETUNE_DIR / f"data_{stem}" / "splits"
    test_csv = splits_dir / "test.csv"
    if not test_csv.is_file():
        print(f"[skip] {stem}: missing {test_csv}")
        return None

    base_pth = base_checkpoint or resolve_base_checkpoint(BASE_MODEL_DIR, stem)
    if base_pth is None:
        print(f"[skip] {stem}: no base checkpoint in {BASE_MODEL_DIR}")
        return None
    base_pth = base_pth.resolve()

    finetuned_pth = resolve_finetuned_pth(stem, finetuned_dir=finetuned_dir)
    if finetuned_pth is None and not base_only:
        msg = f"[skip] {stem}: no finetuned checkpoint found"
        if skip_missing_finetuned:
            print(msg)
            return None
        raise FileNotFoundError(msg)

    print(f"\n=== {stem} ===")
    print(f"  base:      {base_pth}")
    if finetuned_pth is not None:
        print(f"  finetuned: {finetuned_pth}")

    _, _, _, input_len, pred_len = load_checkpoint_bundle(base_pth, device)
    stride = resolve_test_stride(test_stride, pred_len)
    df_train, df_val, df_test = load_splits_smoothed(splits_dir, target_smoothing_window)
    split_frames = {"train": df_train, "val": df_val, "test": df_test}

    sensor_out = out_dir / stem
    sensor_out.mkdir(parents=True, exist_ok=True)
    split_summaries: Dict[str, Any] = {}

    for split_name in eval_splits:
        if split_name not in split_frames:
            raise ValueError(f"Unknown split {split_name!r}; expected train, val, or test.")
        split_csv = splits_dir / f"{split_name}.csv"
        if not split_csv.is_file():
            print(f"[skip] {stem}/{split_name}: missing {split_csv}")
            continue

        if eval_mode == "test_only" or split_name == "test":
            df_eval = split_frames[split_name]
            window_starts = build_split_windows(
                df_eval,
                input_len=input_len,
                pred_len=pred_len,
                require_uniform_timestep=require_uniform_timestep,
                uniform_step_seconds=uniform_step_seconds,
                uniform_tol_seconds=uniform_tol_seconds,
                max_gap_seconds=max_gap_seconds,
                window_stride=stride,
                split_name=split_name,
            )
            mode_label = f"{split_name}.csv only ({len(df_eval)} rows)"
        else:
            df_eval, window_starts, mode_label = prepare_eval_frames(
                df_train,
                df_val,
                df_test,
                eval_mode=eval_mode,
                input_len=input_len,
                pred_len=pred_len,
                require_uniform_timestep=require_uniform_timestep,
                uniform_step_seconds=uniform_step_seconds,
                uniform_tol_seconds=uniform_tol_seconds,
                max_gap_seconds=max_gap_seconds,
                test_stride=stride,
                target_smoothing_window=target_smoothing_window,
            )
            split_name = "test"

        print(f"  [{split_name}] {mode_label}")
        print(f"  [{split_name}] windows: {len(window_starts)}")
        if window_starts.size == 0:
            print(f"[skip] {stem}/{split_name}: no valid windows")
            continue

        base = run_model_on_test_windows(
            base_pth, df_eval, window_starts, norm_df=df_train, device=device, batch_size=batch_size
        )
        finetuned: Optional[Dict[str, Any]] = None
        if finetuned_pth is not None:
            finetuned = run_model_on_test_windows(
                finetuned_pth,
                df_eval,
                window_starts,
                norm_df=df_train,
                device=device,
                batch_size=batch_size,
            )

        history = df_eval[VALUE_COL].to_numpy(dtype=np.float64)
        fq = base.get("forecast_quantiles")
        base_html = sensor_out / f"stitched_{split_name}_base.html"
        save_stitched_test_forecast_html(
            html_path=str(base_html),
            timestamps=df_eval["TIMESTAMP"],
            window_starts=window_starts,
            input_len=input_len,
            pred_len=pred_len,
            history_series_raw=history,
            preds_raw=base["preds_raw"],
            targets_raw=base["targets_raw"],
            y_axis_label=VALUE_COL,
            input_context_label=VALUE_COL,
            pred_smoothing_window=1,
            preds_quantiles_raw=base.get("preds_quantiles_raw"),
            forecast_quantiles=fq,
            prediction_interval_label=prediction_interval_band_label(fq),
            max_consecutive_gap_seconds=max_gap_seconds,
        )

        finetuned_html = None
        comparison_html = None
        if finetuned is not None:
            fq_ft = finetuned.get("forecast_quantiles")
            finetuned_html = sensor_out / f"stitched_{split_name}_finetuned.html"
            save_stitched_test_forecast_html(
                html_path=str(finetuned_html),
                timestamps=df_eval["TIMESTAMP"],
                window_starts=window_starts,
                input_len=input_len,
                pred_len=pred_len,
                history_series_raw=history,
                preds_raw=finetuned["preds_raw"],
                targets_raw=finetuned["targets_raw"],
                y_axis_label=VALUE_COL,
                input_context_label=VALUE_COL,
                pred_smoothing_window=1,
                preds_quantiles_raw=finetuned.get("preds_quantiles_raw"),
                forecast_quantiles=fq_ft,
                prediction_interval_label=prediction_interval_band_label(fq_ft),
                max_consecutive_gap_seconds=max_gap_seconds,
            )
            comparison_html = sensor_out / f"comparison_{split_name}_base_vs_finetuned.html"
            save_comparison_html(
                comparison_html,
                stem=stem,
                timestamps=df_eval["TIMESTAMP"],
                window_starts=window_starts,
                input_len=input_len,
                pred_len=pred_len,
                history_series=history,
                base=base,
                finetuned=finetuned,
            )

        split_summaries[split_name] = {
            "n_windows": int(len(window_starts)),
            "base_rmse": base["rmse"],
            "finetuned_rmse": finetuned["rmse"] if finetuned else None,
            "base_mae": base["mae"],
            "finetuned_mae": finetuned["mae"] if finetuned else None,
            "base_mape": base["mape"],
            "finetuned_mape": finetuned["mape"] if finetuned else None,
            "outputs": {
                "base_html": str(base_html),
                "finetuned_html": str(finetuned_html) if finetuned_html else None,
                "comparison_html": str(comparison_html) if comparison_html else None,
            },
        }
        print(
            f"  [{split_name}] RMSE={base['rmse']:.4f} MAE={base['mae']:.4f} MAPE={base['mape']:.2f}%"
        )

    if not split_summaries:
        print(f"[skip] {stem}: no splits produced outputs")
        return None

    summary = {
        "stem": stem,
        "base_checkpoint": str(base_pth),
        "finetuned_checkpoint": str(finetuned_pth) if finetuned_pth else None,
        "splits": split_summaries,
    }
    with open(sensor_out / "metrics.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)
        fp.write("\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stems",
        nargs="*",
        default=None,
        help="Sensor stems (default: all finetune/data_*/splits with test.csv).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=FINETUNE_DIR / "test_inference_results",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--require-uniform-timestep", action="store_true")
    parser.add_argument("--uniform-step-seconds", type=float, default=1800.0)
    parser.add_argument("--uniform-tol-seconds", type=float, default=60.0)
    parser.add_argument("--max-gap-seconds", type=float, default=None)
    parser.add_argument(
        "--target-smoothing-window",
        type=int,
        default=48,
        help="Causal MA on Acceleration RMS per CSV before eval (finetune sbatch default: 48).",
    )
    parser.add_argument(
        "--eval-mode",
        choices=("test_only", "full_context"),
        default="test_only",
        help="test_only = windows inside each split CSV only. "
        "full_context = train+val history for input (deployment-like; test split only).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val", "test"),
        default=["test"],
        help="Which split CSVs to evaluate (default: test). Use: --splits train val test",
    )
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=None,
        help="Override base .pth path (default: model_to_finetune/<STEM>_merged.pth or highest _vN).",
    )
    parser.add_argument(
        "--test-window-stride",
        type=int,
        default=1,
        help="Stride between test window starts (default 1, dense overlap like finetune plots). "
        "Use 0 to match finetune sbatch flag (--test-window-stride 0 → stride=pred_len).",
    )
    parser.add_argument(
        "--base-only",
        action="store_true",
        help="Only plot base-model predictions (skip finetuned / comparison).",
    )
    parser.add_argument(
        "--finetuned-dir",
        type=Path,
        default=FINETUNED_MODEL_DIR,
        help="Directory with finetuned checkpoints (<STEM>_ft.pth).",
    )
    parser.add_argument(
        "--allow-missing-finetuned",
        action="store_true",
        help="Skip sensors without a finetuned .pth instead of erroring.",
    )
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    stems = args.stems or discover_sensor_stems()
    if not stems:
        raise SystemExit("No sensor split folders found under finetune/data_*/splits/")

    summaries: List[Dict[str, Any]] = []
    for stem in stems:
        try:
            row = process_sensor(
                stem,
                out_dir=args.out_dir,
                device=device,
                batch_size=args.batch_size,
                require_uniform_timestep=args.require_uniform_timestep,
                uniform_step_seconds=args.uniform_step_seconds,
                uniform_tol_seconds=args.uniform_tol_seconds,
                max_gap_seconds=args.max_gap_seconds,
                test_stride=args.test_window_stride,
                eval_mode=args.eval_mode,
                eval_splits=list(args.splits),
                target_smoothing_window=args.target_smoothing_window,
                skip_missing_finetuned=args.allow_missing_finetuned or args.base_only,
                base_only=args.base_only,
                finetuned_dir=args.finetuned_dir,
                base_checkpoint=args.base_checkpoint,
            )
            if row:
                summaries.append(row)
        except Exception as exc:
            print(f"[error] {stem}: {exc}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump(summaries, fp, indent=2)
        fp.write("\n")
    print(f"\nWrote {len(summaries)} sensor summaries to {summary_path}")


if __name__ == "__main__":
    main()
