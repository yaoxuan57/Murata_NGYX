import copy
import json
import os
import random
from typing import Callable, Dict, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset


def add_common_args(parser, default_output_dir: str, default_checkpoint_name: str):
    parser.add_argument(
        "--train-csv",
        type=str,
        default=None,
        metavar="PATH",
        help="If set together with --val-csv, load three disjoint CSVs (train / val / test). "
        "Normalization uses train only. Prefer files cut at timestamp gaps so each file is one contiguous run.",
    )
    parser.add_argument(
        "--val-csv",
        type=str,
        default=None,
        metavar="PATH",
        help="Companion to --train-csv for an explicit validation set (disables row-based val split).",
    )
    parser.add_argument("--train-val-csv", type=str, default="data_train_val.csv")
    parser.add_argument("--test-csv", type=str, default="data_test_anomalous.csv")
    parser.add_argument(
        "--single-csv",
        type=str,
        default=None,
        metavar="PATH",
        help="Load one chronological CSV and split train/val/test by window counts (see --train-ratio, --val-ratio, --test-ratio). "
        "Each window still requires uniform TIMESTAMP steps (--require-uniform-timestep). "
        "Cannot be combined with --train-csv/--val-csv or the two-file train_val+test setup.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="With --single-csv: target fraction of valid windows for training (chronological). Ignored otherwise.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=None,
        help="With --single-csv: test window fraction; default 1 - train_ratio - val_ratio.",
    )
    parser.add_argument(
        "--min-windows-per-split",
        type=int,
        default=1,
        metavar="N",
        help="With --single-csv: minimum windows in each split after ratio allocation (raised if impossible).",
    )
    parser.add_argument(
        "--value-column",
        type=str,
        default="Acceleration RMS (smoothed)",
        help="CSV column used as the univariate forecast target. Use 'Acceleration RMS' for raw (unsmoothed) CSVs.",
    )
    parser.add_argument(
        "--feature-columns",
        type=str,
        nargs="+",
        default=None,
        help="Input feature columns. If omitted, uses only --value-column.",
    )
    parser.add_argument(
        "--use-all-numeric-features",
        dest="use_all_numeric_features",
        action="store_true",
        help="Use all numeric columns except TIMESTAMP as input features.",
    )
    parser.add_argument(
        "--no-use-all-numeric-features",
        dest="use_all_numeric_features",
        action="store_false",
    )
    parser.set_defaults(use_all_numeric_features=False)
    parser.add_argument(
        "--raw-compare-column",
        type=str,
        default=None,
        metavar="COLUMN",
        help="Reserved/unused by current plotting code. Rolling-window PNGs plot the history of "
        "--value-column exactly as used after optional --target-smoothing-window (aligned with "
        "train/val/test). Forecast series are raw model outputs unless --pred-smoothing-window>1.",
    )
    parser.add_argument("--output-dir", type=str, default=default_output_dir)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--input-lens", type=int, nargs="+", default=[432,576,864])
    parser.add_argument("--pred-lens", type=int, nargs="+", default=[288])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="1) Legacy: holdout fraction of windows from --train-val-csv. "
        "2) With --single-csv: val window fraction (with --train-ratio, --test-ratio). "
        "Ignored when --train-csv and --val-csv are set.",
    )
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--scheduler-patience", type=int, default=3)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--plot-sample-idx", type=int, default=200)
    parser.add_argument("--checkpoint-name", type=str, default=default_checkpoint_name)
    parser.add_argument("--loss-huber-delta", type=float, default=1.0)
    parser.add_argument("--loss-point-weight", type=float, default=0.2)
    parser.add_argument("--loss-diff-weight", type=float, default=5)
    parser.add_argument("--loss-curvature-weight", type=float, default=2.0)
    parser.add_argument("--loss-variance-weight", type=float, default=0.8)
    parser.add_argument(
        "--loss-laplacian-weight",
        type=float,
        default=0.3,
        help="Extra penalty on mean squared second difference of the predicted trajectory "
        "(suppresses high-frequency jaggedness). Set 0 to disable.",
    )
    parser.add_argument(
        "--loss-tail-weight",
        type=float,
        default=1.0,
        help="Per-step loss weight at horizon=pred_len-1 (horizon 0 has weight 1.0). "
        "Loss weights interpolate linearly from 1.0 (head) to this value (tail). "
        "Default 1.0 = flat (no decay). Use <1.0 to down-weight late horizons "
        "(historical default was 0.8, which let the tail drift up).",
    )
    parser.add_argument(
        "--pred-smoothing-window",
        type=int,
        default=1,
        metavar="K",
        help="If K>1, apply a centered length-K moving average to each test forecast row "
        "(raw model output) before metrics and saved plots/CSVs. K is bumped up by 1 if even. K=1 disables.",
    )
    parser.add_argument(
        "--target-smoothing-window",
        type=int,
        default=1,
        metavar="K",
        help="If K>1, apply a centered length-K moving average to --value-column in each loaded CSV "
        "before building train/val/test windows (per file only; other feature columns unchanged). "
        "K is bumped to odd if even. K=1 disables.",
    )
    parser.add_argument("--save-window-plots", dest="save_window_plots", action="store_true")
    parser.add_argument("--no-window-plots", dest="save_window_plots", action="store_false")
    parser.set_defaults(save_window_plots=True)
    parser.add_argument(
        "--rolling-window-artifact-limit",
        type=int,
        default=None,
        metavar="N",
        help="If set, write at most N per-window CSV and PNG files under rolling_window_forecasts/, "
        "evenly spaced from the first to last window. All windows remain in all_windows_forecasts.csv.",
    )
    parser.add_argument(
        "--require-uniform-timestep",
        dest="require_uniform_timestep",
        action="store_true",
        help="Only build windows whose TIMESTAMP steps are uniformly spaced (see --uniform-step-seconds). "
        "Skips spanning gaps where the machine was off/rested. Rows must be chronological.",
    )
    parser.add_argument(
        "--no-require-uniform-timestep",
        dest="require_uniform_timestep",
        action="store_false",
    )
    parser.set_defaults(require_uniform_timestep=True)
    parser.add_argument(
        "--uniform-step-seconds",
        type=float,
        default=5.0,
        help="Nominal interval between consecutive rows when uniform timestep filtering is enabled.",
    )
    parser.add_argument(
        "--uniform-step-tolerance-seconds",
        type=float,
        default=1.01,
        help="Half-width tolerance on each step vs nominal (seconds). Rows outside nominal±tol break a contiguous run.",
    )
    return parser


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mse_np(y_true, y_pred):
    err = y_true - y_pred
    return float(np.mean(err ** 2))


def rmse_np(y_true, y_pred):
    return float(np.sqrt(mse_np(y_true, y_pred)))


def mae_np(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def mape_np(y_true, y_pred):
    denom = np.maximum(np.abs(y_true), 1e-8)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def smooth_forecast_horizons(preds: np.ndarray, window: int) -> np.ndarray:
    """Row-wise centered moving average for arrays shaped (n_windows, pred_len)."""
    if window <= 1:
        return preds
    w = window if window % 2 == 1 else window + 1
    pad = w // 2
    kernel = np.ones(w, dtype=np.float64) / w
    out = np.empty_like(preds, dtype=np.float64)
    for i in range(preds.shape[0]):
        y = np.asarray(preds[i], dtype=np.float64)
        y_pad = np.pad(y, (pad, pad), mode="edge")
        out[i] = np.convolve(y_pad, kernel, mode="valid")
    return out.astype(preds.dtype, copy=False)


def smooth_forecast_vector(vec: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average for a single 1D forecast (pred_len,)."""
    if window <= 1:
        return vec
    w = window if window % 2 == 1 else window + 1
    pad = w // 2
    y = np.asarray(vec, dtype=np.float64)
    kernel = np.ones(w, dtype=np.float64) / w
    y_pad = np.pad(y, (pad, pad), mode="edge")
    return np.convolve(y_pad, kernel, mode="valid").astype(vec.dtype, copy=False)


def smooth_target_series_1d(vec: np.ndarray, window: int) -> np.ndarray:
    """Centered MA on a 1D target series (full CSV column) before sliding windows; edge padding."""
    if window <= 1:
        return np.asarray(vec, dtype=np.float32)
    w = window if window % 2 == 1 else window + 1
    pad = w // 2
    y = np.asarray(vec, dtype=np.float64)
    kernel = np.ones(w, dtype=np.float64) / w
    y_pad = np.pad(y, (pad, pad), mode="edge")
    return np.convolve(y_pad, kernel, mode="valid").astype(np.float32)


def r2_np(y_true, y_pred):
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)
    ss_res = float(np.sum((y_true_flat - y_pred_flat) ** 2))
    ss_tot = float(np.sum((y_true_flat - y_true_flat.mean()) ** 2))
    return 1.0 - ss_res / (ss_tot + 1e-12)


def evaluate_metrics(y_true, y_pred):
    return {
        "mse": mse_np(y_true, y_pred),
        "rmse": rmse_np(y_true, y_pred),
        "mae": mae_np(y_true, y_pred),
        "mape": mape_np(y_true, y_pred),
        "r2": r2_np(y_true, y_pred),
    }


# Contiguous horizon bands for reporting (near-equal width). Forecast steps are 1-based (step-ahead index).
HORIZON_PHASE_COUNT = 6


def horizon_phase_step_ranges(pred_len: int, n_phases: int = HORIZON_PHASE_COUNT):
    """Return list of (h_start, h_end) inclusive 1-based step-ahead indices per phase; empty phases -> (None, None)."""
    splits = np.array_split(np.arange(pred_len), n_phases)
    ranges = []
    for part in splits:
        if part.size == 0:
            ranges.append((None, None))
        else:
            ranges.append((int(part[0]) + 1, int(part[-1]) + 1))
    return ranges


def compute_horizon_phase_mapes(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_phases: int = HORIZON_PHASE_COUNT,
):
    """
    Test MAPE (%) per horizon phase: same formula as global test MAPE, pooling all test windows
    and all timesteps whose step-ahead index falls in that phase.
    """
    pred_len = int(y_true.shape[1])
    splits = np.array_split(np.arange(pred_len), n_phases)
    mapes = []
    for part in splits:
        if part.size == 0:
            mapes.append(float("nan"))
            continue
        a, b = int(part[0]), int(part[-1]) + 1
        mapes.append(mape_np(y_true[:, a:b].ravel(), y_pred[:, a:b].ravel()))
    return mapes


def horizon_phase_ranges_csv_string(ranges: list) -> str:
    parts = [f"{lo}-{hi}" for lo, hi in ranges if lo is not None and hi is not None]
    return ";".join(parts)


def parse_timestamp_series(series: pd.Series, name: str) -> pd.Series:
    raw = series.astype(str).str.strip()
    parsed = pd.to_datetime(raw, format="%Y-%m-%d %H:%M:%S", errors="coerce")

    mask = parsed.isna()
    if mask.any():
        parsed.loc[mask] = pd.to_datetime(
            raw.loc[mask],
            format="%Y-%m-%d %H:%M",
            errors="coerce",
        )

    mask = parsed.isna()
    if mask.any():
        parsed.loc[mask] = pd.to_datetime(
            raw.loc[mask],
            format="%d/%m/%Y %H:%M",
            errors="coerce",
        )

    mask = parsed.isna()
    if mask.any():
        parsed.loc[mask] = pd.to_datetime(
            raw.loc[mask],
            dayfirst=True,
            errors="coerce",
        )

    remaining_bad = parsed.isna()
    if remaining_bad.any():
        bad_examples = raw.loc[remaining_bad].head(5).tolist()
        raise ValueError(
            f"Failed to parse some TIMESTAMP values in {name}. "
            f"Examples: {bad_examples}"
        )

    return parsed


def summarize_timestamp_steps(
    timestamps: pd.Series,
    label: str,
    nominal_seconds: float,
    tolerance_seconds: float,
) -> None:
    """Print a short summary of inter-row TIMESTAMP gaps.

    Helps diagnose why a uniform-timestep window filter accepts/rejects rows.
    Reports median/mean/min/max step (seconds) and the share of consecutive
    pairs whose step lies within ``nominal_seconds ± tolerance_seconds``.
    """
    ts = pd.Series(timestamps).reset_index(drop=True)
    if len(ts) < 2:
        print(f"  [step-stats:{label}] not enough rows for diff stats (n={len(ts)}).")
        return
    diffs_s = np.diff(ts.to_numpy(dtype="datetime64[ns]").astype("int64")) / 1e9
    nominal = float(nominal_seconds)
    tol = float(tolerance_seconds)
    pct_within = 100.0 * float(np.mean(np.abs(diffs_s - nominal) <= tol))
    try:
        mode_val = float(pd.Series(np.round(diffs_s, 3)).mode().iloc[0])
    except Exception:
        mode_val = float("nan")
    print(
        f"  [step-stats:{label}] median={np.median(diffs_s):.3f}s mode={mode_val:.3f}s "
        f"mean={diffs_s.mean():.3f}s min={diffs_s.min():.3f}s max={diffs_s.max():.3f}s | "
        f"pairs within {nominal:g}±{tol:g}s = {pct_within:.2f}% (n_diffs={diffs_s.size})"
    )


def compute_uniform_timestep_start_indices(
    timestamps: pd.Series,
    span_len: int,
    nominal_seconds: float = 5.0,
    tolerance_seconds: float = 1.01,
) -> np.ndarray:
    """
    Indices i such that timestamps.iloc[i : i + span_len] advances by nominal_seconds (+/- tol)
    at every consecutive pair. Gaps or irregular steps break contiguous runs — windows never
    glue two runs together.
    """
    ts = pd.Series(timestamps).reset_index(drop=True)
    if span_len <= 0:
        raise ValueError("span_len must be positive.")
    if len(ts) < span_len:
        return np.array([], dtype=np.int64)
    values = ts.to_numpy(dtype="datetime64[ns]")
    if span_len == 1:
        return np.arange(len(ts), dtype=np.int64)

    diffs_ns = np.diff(values.astype("int64"))
    nominal_ns = int(round(float(nominal_seconds) * 1e9))
    tol_ns = int(round(float(tolerance_seconds) * 1e9))
    step_ok = np.abs(diffs_ns - nominal_ns) <= tol_ns
    m = span_len - 1
    if step_ok.size < m:
        return np.array([], dtype=np.int64)
    conv = np.convolve(step_ok.astype(np.int32), np.ones(m, dtype=np.int32), mode="valid")
    return np.where(conv == m)[0].astype(np.int64)


def row_indices_covered_by_windows(starts: np.ndarray, span: int, n_rows: int) -> np.ndarray:
    """Sorted unique row indices touched by any window [s, s + span) for s in starts."""
    starts = np.asarray(starts, dtype=np.int64)
    if starts.size == 0:
        return np.zeros(0, dtype=np.int64)
    if starts.size >= 2 and np.all(np.diff(starts) == 1):
        lo = int(starts[0])
        hi = int(starts[-1]) + int(span)
        return np.arange(lo, min(hi, n_rows), dtype=np.int64)
    m = np.zeros(int(n_rows), dtype=bool)
    for s in starts:
        ss = int(s)
        if ss < 0:
            continue
        m[ss : min(ss + int(span), n_rows)] = True
    return np.flatnonzero(m)


def split_window_counts(
    total: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    min_each: int = 1,
) -> Tuple[int, int, int]:
    """
    Integer train/val/test window counts approximating the given ratios, summing to ``total``.
    Each split receives at least ``min_each`` windows; remaining windows split by ratio.
    """
    if total < 3 * min_each:
        raise ValueError(
            f"Need at least {3 * min_each} valid windows for train/val/test (got {total})."
        )
    w = np.asarray([train_ratio, val_ratio, test_ratio], dtype=np.float64)
    if np.any(w < 0):
        raise ValueError("train/val/test ratios must be non-negative.")
    ssum = float(w.sum())
    if ssum <= 0:
        raise ValueError("train/val/test ratios must sum to a positive value.")
    w /= ssum
    left = total - 3 * min_each
    raw = left * w
    extra = np.floor(raw).astype(int)
    rem = left - int(extra.sum())
    frac_order = np.argsort(-(raw - extra))
    for k in range(rem):
        extra[int(frac_order[k % 3])] += 1
    parts = min_each + extra
    sp = int(parts.sum())
    if sp != total:
        raise RuntimeError(f"split_window_counts internal error: sums to {sp} != {total}.")
    return int(parts[0]), int(parts[1]), int(parts[2])


class MultiStepDeltaDataset(Dataset):
    def __init__(self, features_norm, target_norm, input_len, pred_len, sample_starts=None):
        features_norm = np.asarray(features_norm, dtype=np.float32)
        target_norm = np.asarray(target_norm, dtype=np.float32)
        if features_norm.ndim != 2:
            raise ValueError(f"features_norm must be 2D (time, n_features), got shape={features_norm.shape}")
        if target_norm.ndim != 1:
            raise ValueError(f"target_norm must be 1D (time,), got shape={target_norm.shape}")
        if len(features_norm) != len(target_norm):
            raise ValueError("features_norm and target_norm must have identical time length.")
        self.input_len = input_len
        self.pred_len = pred_len
        tlen = len(target_norm)
        span = input_len + pred_len

        if sample_starts is None:
            n_sliding = tlen - span + 1
            if n_sliding <= 0:
                raise ValueError("Series too short for given input_len and pred_len.")
            self.sample_starts = np.arange(n_sliding, dtype=np.int64)
        else:
            self.sample_starts = np.asarray(sample_starts, dtype=np.int64)
            if self.sample_starts.ndim != 1:
                raise ValueError("sample_starts must be a 1D array of row indices.")
            if self.sample_starts.size == 0:
                raise ValueError("No valid training windows (uniform timestep or length).")
            if (self.sample_starts < 0).any() or ((self.sample_starts + span) > tlen).any():
                raise ValueError(
                    "sample_starts entries must satisfy 0 <= i and i + input_len + pred_len <= len(series)."
                )

        idx = self.sample_starts
        x = np.stack([features_norm[i : i + input_len, :] for i in idx], axis=0).astype(np.float32)
        x = np.transpose(x, (0, 2, 1))

        future = np.stack(
            [target_norm[i + input_len : i + input_len + pred_len] for i in idx],
            axis=0,
        ).astype(np.float32)

        last_val = target_norm[idx + input_len - 1].astype(np.float32)[:, None]
        y_delta = future - last_val

        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y_delta, dtype=torch.float32)
        self.last_val = torch.tensor(last_val, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.last_val[idx]


class TrajectoryAwareLoss(nn.Module):
    def __init__(
        self,
        pred_len,
        delta=1.0,
        point_weight=0.4,
        diff_weight=1.2,
        curvature_weight=0.8,
        variance_weight=0.4,
        laplacian_reg_weight=0.0,
        tail_weight=1.0,
    ):
        super().__init__()
        self.delta = delta
        self.point_weight = point_weight
        self.diff_weight = diff_weight
        self.curvature_weight = curvature_weight
        self.variance_weight = variance_weight
        self.laplacian_reg_weight = laplacian_reg_weight
        self.tail_weight = float(tail_weight)
        w = torch.linspace(1.0, self.tail_weight, pred_len, dtype=torch.float32)
        self.register_buffer("w", w / w.mean())

    def _weighted_huber(self, pred, target, weights):
        err = pred - target
        abs_err = err.abs()
        huber = torch.where(
            abs_err < self.delta,
            0.5 * err ** 2,
            self.delta * (abs_err - 0.5 * self.delta),
        )
        return (huber * weights).mean()

    def forward(self, pred, target):
        point_weights = self.w.to(pred.device)
        point_loss = self._weighted_huber(pred, target, point_weights)

        pred_diff = pred[:, 1:] - pred[:, :-1]
        target_diff = target[:, 1:] - target[:, :-1]
        diff_weights = point_weights[1:]
        diff_loss = self._weighted_huber(pred_diff, target_diff, diff_weights)

        pred_curvature = pred_diff[:, 1:] - pred_diff[:, :-1]
        target_curvature = target_diff[:, 1:] - target_diff[:, :-1]
        curvature_weights = point_weights[2:]
        curvature_loss = self._weighted_huber(pred_curvature, target_curvature, curvature_weights)

        pred_std = pred.std(dim=1, unbiased=False)
        target_std = target.std(dim=1, unbiased=False)
        variance_loss = torch.mean(torch.abs(pred_std - target_std))

        if self.laplacian_reg_weight > 0 and pred.size(1) >= 3:
            d2_pred = pred[:, 2:] - 2.0 * pred[:, 1:-1] + pred[:, :-2]
            laplacian_reg = (d2_pred ** 2).mean()
        else:
            laplacian_reg = pred.new_tensor(0.0)

        return (
            self.point_weight * point_loss
            + self.diff_weight * diff_loss
            + self.curvature_weight * curvature_loss
            + self.variance_weight * variance_loss
            + self.laplacian_reg_weight * laplacian_reg
        )


def run_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    total_count = 0

    with torch.set_grad_enabled(training):
        for x, y_delta, last_val in loader:
            x = x.to(device)
            y_delta = y_delta.to(device)
            last_val = last_val.to(device)

            pred_delta = model(x)
            pred_abs = pred_delta + last_val
            true_abs = y_delta + last_val
            loss = criterion(pred_abs, true_abs)

            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            total_count += batch_size

    return total_loss / max(total_count, 1)


def collect_predictions(model, loader, train_std, train_mean, device):
    all_preds_abs_norm = []
    all_targets_abs_norm = []

    model.eval()
    with torch.no_grad():
        for x, y_delta, last_val in loader:
            x = x.to(device)
            y_delta = y_delta.to(device)
            last_val = last_val.to(device)

            pred_delta = model(x)

            pred_abs_norm = pred_delta + last_val
            true_abs_norm = y_delta + last_val

            all_preds_abs_norm.append(pred_abs_norm.cpu().numpy())
            all_targets_abs_norm.append(true_abs_norm.cpu().numpy())

    all_preds_abs_norm = np.concatenate(all_preds_abs_norm, axis=0)
    all_targets_abs_norm = np.concatenate(all_targets_abs_norm, axis=0)

    all_preds_raw = all_preds_abs_norm * train_std + train_mean
    all_targets_raw = all_targets_abs_norm * train_std + train_mean
    return all_preds_raw, all_targets_raw


def compute_window_rmse(model, loader, train_std, train_mean, device):
    preds_raw, targets_raw = collect_predictions(model, loader, train_std, train_mean, device)
    window_rmse = np.sqrt(np.mean((targets_raw - preds_raw) ** 2, axis=1))
    return float(window_rmse.mean())


def baseline_rmse(test_loader, pred_len, train_std, train_mean):
    baseline_preds_abs_norm = []
    baseline_targets_abs_norm = []

    for x, y_delta, last_val in test_loader:
        y_delta_np = y_delta.numpy()
        last_val_np = last_val.numpy()

        pred_abs_norm = np.repeat(last_val_np, pred_len, axis=1)
        true_abs_norm = y_delta_np + last_val_np

        baseline_preds_abs_norm.append(pred_abs_norm)
        baseline_targets_abs_norm.append(true_abs_norm)

    baseline_preds_raw = np.concatenate(baseline_preds_abs_norm, axis=0) * train_std + train_mean
    baseline_targets_raw = np.concatenate(baseline_targets_abs_norm, axis=0) * train_std + train_mean
    return rmse_np(baseline_targets_raw, baseline_preds_raw)


def save_plot(path, title, x_label, y_label, x, y1, y1_label, y2=None, y2_label=None, rotate_dates=False):
    plt.figure(figsize=(10, 4))
    plt.plot(x, y1, label=y1_label)
    if y2 is not None:
        plt.plot(x, y2, label=y2_label)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)
    if rotate_dates:
        plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def build_horizon_forecast_dataframe(timestamps, actual, predicted, horizon):
    return pd.DataFrame(
        {
            "timestamp": timestamps.astype(str).to_list(),
            "horizon": [horizon] * len(actual),
            "actual": actual,
            "predicted": predicted,
        }
    )


def _forecast_smoothing_caption(pred_smoothing_window: int) -> str:
    if pred_smoothing_window <= 1:
        return "No post-smoothing (pred-smoothing-window = 1)"
    return f"Post-smoothing applied (pred-smoothing-window = {pred_smoothing_window})"


def evenly_spaced_window_indices(n_windows: int, k: int) -> Set[int]:
    """Up to k indices in [0, n_windows-1], equally spaced (endpoints included when k >= 2)."""
    if n_windows <= 0 or k <= 0:
        return set()
    if n_windows <= k:
        return set(range(n_windows))
    if k == 1:
        return {0}
    out = {int(round(i * (n_windows - 1) / (k - 1))) for i in range(k)}
    return out


def _plot_rolling_window_png(
    window_idx: int,
    input_len: int,
    pred_len: int,
    history_series_raw: np.ndarray,
    targets_row: np.ndarray,
    preds_row: np.ndarray,
    y_axis_label: str,
    path: str,
    dpi: int = 140,
    input_row_start: Optional[int] = None,
    input_context_label: str = "Acceleration RMS",
    pred_smoothing_window: int = 1,
):
    """Left: history of target column (z-scored), same series as train/val/test after target pre-smoothing."""

    row0 = window_idx if input_row_start is None else int(input_row_start)
    hist = np.asarray(history_series_raw[row0 : row0 + input_len], dtype=np.float64)
    x_hist = np.arange(0, input_len, dtype=np.float64)
    x_fore = np.arange(input_len, input_len + pred_len, dtype=np.float64)
    w_in = max(10.0, min(22.0, 6.0 + 0.004 * float(input_len + pred_len)))
    fig, ax_left = plt.subplots(1, 1, figsize=(w_in, 3.8))
    ax_right = ax_left.twinx()

    ax_left.axvline(x=input_len - 0.5, color="0.55", linestyle="--", linewidth=1.2, zorder=1)
    vstd = hist.std()
    norm = (hist - hist.mean()) / (vstd + 1e-8)
    ax_left.plot(
        x_hist,
        norm,
        linewidth=0.95,
        alpha=0.85,
        color="0.35",
        label=f"{input_context_label} — input (z-scored)",
        zorder=2,
    )

    ax_right.plot(x_fore, targets_row, color="C0", linewidth=1.3, label="Actual target", zorder=4)
    ax_right.plot(x_fore, preds_row, color="C1", linewidth=1.1, label="Predicted target", zorder=4)
    ax_left.set_title(
        f"Window {window_idx} — {input_context_label} input + forecast ({pred_len}-step)\n"
        f"{_forecast_smoothing_caption(pred_smoothing_window)}",
        fontsize=10,
    )
    ax_left.set_ylabel(f"{input_context_label} (z-scored)")
    ax_right.set_ylabel(y_axis_label)
    ax_left.set_xlabel(
        f"Step index (0-{input_len - 1}: input context | {input_len}-{input_len + pred_len - 1}: horizon)"
    )
    ax_left.grid(True, alpha=0.35)
    h1, l1 = ax_left.get_legend_handles_labels()
    h2, l2 = ax_right.get_legend_handles_labels()
    max_left_legend = min(len(h1), 8)
    ax_left.legend(h1[:max_left_legend] + h2, l1[:max_left_legend] + l2, loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def save_rolling_window_forecasts(
    output_dir,
    preds_raw,
    targets_raw,
    timestamps,
    input_len,
    pred_len,
    save_plots=True,
    max_per_window_artifacts: Optional[int] = None,
    y_axis_label: str = "Value",
    history_series_raw: Optional[np.ndarray] = None,
    window_input_row_starts: Optional[np.ndarray] = None,
    input_context_label: str = "Acceleration RMS",
    pred_smoothing_window: int = 1,
):
    windows_dir = os.path.join(output_dir, "rolling_window_forecasts")
    plots_dir = os.path.join(windows_dir, "plots")
    csv_dir = os.path.join(windows_dir, "csv")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    all_rows = []
    n_windows = preds_raw.shape[0]
    if window_input_row_starts is None:
        input_starts = np.arange(n_windows, dtype=np.int64)
    else:
        input_starts = np.asarray(window_input_row_starts, dtype=np.int64).reshape(-1)
        if input_starts.shape[0] != n_windows:
            raise ValueError(
                f"window_input_row_starts length {input_starts.shape[0]} != n_windows {n_windows}"
            )

    if max_per_window_artifacts is None or max_per_window_artifacts <= 0:
        save_indices = set(range(n_windows))
    else:
        save_indices = evenly_spaced_window_indices(n_windows, max_per_window_artifacts)

    if (
        max_per_window_artifacts is not None
        and max_per_window_artifacts > 0
        and n_windows > len(save_indices)
    ):
        print(
            f"Rolling-window artifacts: writing {len(save_indices)} per-window CSV/PNG files "
            f"(evenly spaced over {n_windows} windows). Full series in all_windows_forecasts.csv."
        )

    for window_idx in range(n_windows):
        row0 = int(input_starts[window_idx])
        start_idx = row0 + input_len
        ts_window = timestamps.iloc[start_idx : start_idx + pred_len]

        window_df = pd.DataFrame(
            {
                "window_index": [window_idx] * pred_len,
                "step_ahead": np.arange(1, pred_len + 1),
                "timestamp": ts_window.astype(str).to_list(),
                "actual": targets_raw[window_idx],
                "predicted": preds_raw[window_idx],
            }
        )

        if window_idx in save_indices:
            window_csv_path = os.path.join(csv_dir, f"window_{window_idx:06d}.csv")
            window_df.to_csv(window_csv_path, index=False)

        all_rows.append(window_df)

        if save_plots and window_idx in save_indices:
            window_plot_path = os.path.join(plots_dir, f"window_{window_idx:06d}.png")
            if history_series_raw is not None:
                need_len = (
                    preds_raw.shape[0] + input_len + pred_len - 1
                    if window_input_row_starts is None
                    else int(np.max(input_starts)) + input_len + pred_len
                )
                if len(history_series_raw) < need_len:
                    raise ValueError(
                        f"history_series_raw length {len(history_series_raw)} < required {need_len} "
                        f"for rolling plots (windows={preds_raw.shape[0]}, input_len={input_len}, pred_len={pred_len})."
                    )
                _plot_rolling_window_png(
                    window_idx=window_idx,
                    input_len=input_len,
                    pred_len=pred_len,
                    history_series_raw=np.asarray(history_series_raw, dtype=np.float32),
                    targets_row=np.asarray(window_df["actual"], dtype=np.float32),
                    preds_row=np.asarray(window_df["predicted"], dtype=np.float32),
                    y_axis_label=y_axis_label,
                    path=window_plot_path,
                    input_row_start=row0,
                    input_context_label=input_context_label,
                    pred_smoothing_window=pred_smoothing_window,
                )
            else:
                plt.figure(figsize=(8, 3))
                plt.plot(window_df["step_ahead"], window_df["actual"], label="Actual")
                plt.plot(window_df["step_ahead"], window_df["predicted"], label="Predicted")
                plt.title(
                    f"Window {window_idx} Forecast ({pred_len}-step)\n"
                    f"{_forecast_smoothing_caption(pred_smoothing_window)}"
                )
                plt.xlabel("Step Ahead")
                plt.ylabel(y_axis_label)
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(window_plot_path, dpi=140)
                plt.close()

    combined_df = pd.concat(all_rows, ignore_index=True)
    combined_csv_path = os.path.join(windows_dir, "all_windows_forecasts.csv")
    combined_df.to_csv(combined_csv_path, index=False)

    if save_plots:
        expected_plots = len(save_indices)
        generated_plot_count = len(
            [name for name in os.listdir(plots_dir) if name.startswith("window_") and name.endswith(".png")]
        )
        if generated_plot_count < expected_plots:
            print(
                f"Detected only {generated_plot_count}/{expected_plots} rolling-window PNGs under {plots_dir}. "
                "Regenerating missing plots before exit."
            )
            for window_df in all_rows:
                window_idx = int(window_df["window_index"].iloc[0])
                if window_idx not in save_indices:
                    continue
                window_plot_path = os.path.join(plots_dir, f"window_{window_idx:06d}.png")
                if os.path.isfile(window_plot_path):
                    continue

                if history_series_raw is not None:
                    wix = int(window_df["window_index"].iloc[0])
                    rs = int(input_starts[wix])
                    _plot_rolling_window_png(
                        window_idx=wix,
                        input_len=input_len,
                        pred_len=pred_len,
                        history_series_raw=np.asarray(history_series_raw, dtype=np.float32),
                        targets_row=np.asarray(window_df["actual"], dtype=np.float32),
                        preds_row=np.asarray(window_df["predicted"], dtype=np.float32),
                        y_axis_label=y_axis_label,
                        path=window_plot_path,
                        input_row_start=rs,
                        input_context_label=input_context_label,
                        pred_smoothing_window=pred_smoothing_window,
                    )
                else:
                    plt.figure(figsize=(8, 3))
                    plt.plot(window_df["step_ahead"], window_df["actual"], label="Actual")
                    plt.plot(window_df["step_ahead"], window_df["predicted"], label="Predicted")
                    plt.title(
                        f"Window {window_idx} Forecast ({pred_len}-step)\n"
                        f"{_forecast_smoothing_caption(pred_smoothing_window)}"
                    )
                    plt.xlabel("Step Ahead")
                    plt.ylabel(y_axis_label)
                    plt.legend()
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(window_plot_path, dpi=140)
                    plt.close()

            final_plot_count = len(
                [name for name in os.listdir(plots_dir) if name.startswith("window_") and name.endswith(".png")]
            )
            print(f"Rolling-window PNGs available: {final_plot_count}/{expected_plots}")

    return windows_dir, combined_csv_path


def run_sweep(
    args,
    model_factory: Callable[[int, int, object, torch.device], nn.Module],
    model_config_factory: Callable[[object, int, int], Dict],
):
    os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    tr_path = args.train_csv
    va_path = args.val_csv
    if (tr_path is None) ^ (va_path is None):
        raise ValueError("Set both --train-csv and --val-csv, or neither (use --train-val-csv for train+val).")
    explicit_tv = tr_path is not None and va_path is not None
    single_file_mode = args.single_csv is not None

    if single_file_mode and explicit_tv:
        raise ValueError("Use either --single-csv or (--train-csv and --val-csv), not both.")
    if single_file_mode:
        df_all = pd.read_csv(args.single_csv)
        df_all["TIMESTAMP"] = parse_timestamp_series(df_all["TIMESTAMP"], args.single_csv)
        df_all = df_all.sort_values("TIMESTAMP").reset_index(drop=True)
        df_test = df_all
        df_train_val = None
        df_train = None
        df_val = None
        rte_resolve = (
            args.test_ratio if args.test_ratio is not None else (1.0 - args.train_ratio - args.val_ratio)
        )
        ssum = float(args.train_ratio + args.val_ratio + rte_resolve)
        if abs(ssum - 1.0) > 2e-3:
            raise ValueError(
                f"--train-ratio, --val-ratio, and --test-ratio (or implied test) must sum to 1; got {ssum:.6f}."
            )
        setattr(args, "test_ratio_resolved", float(rte_resolve))
        print(
            f"Single-CSV split (window counts): train={args.train_ratio:.4f}, val={args.val_ratio:.4f}, "
            f"test={rte_resolve:.4f}"
        )
    elif explicit_tv:
        df_test = pd.read_csv(args.test_csv)
        df_test["TIMESTAMP"] = parse_timestamp_series(df_test["TIMESTAMP"], args.test_csv)
        df_test = df_test.sort_values("TIMESTAMP").reset_index(drop=True)
        df_train = pd.read_csv(tr_path)
        df_val = pd.read_csv(va_path)
        df_train["TIMESTAMP"] = parse_timestamp_series(df_train["TIMESTAMP"], tr_path)
        df_val["TIMESTAMP"] = parse_timestamp_series(df_val["TIMESTAMP"], va_path)
        df_train = df_train.sort_values("TIMESTAMP").reset_index(drop=True)
        df_val = df_val.sort_values("TIMESTAMP").reset_index(drop=True)
        df_train_val = None
        df_all = None
    else:
        df_all = None
        df_train = df_val = None
        df_test = pd.read_csv(args.test_csv)
        df_test["TIMESTAMP"] = parse_timestamp_series(df_test["TIMESTAMP"], args.test_csv)
        df_test = df_test.sort_values("TIMESTAMP").reset_index(drop=True)
        df_train_val = pd.read_csv(args.train_val_csv)
        df_train_val["TIMESTAMP"] = parse_timestamp_series(df_train_val["TIMESTAMP"], args.train_val_csv)
        df_train_val = df_train_val.sort_values("TIMESTAMP").reset_index(drop=True)

    vc = args.value_column

    def _frames_iter():
        if single_file_mode:
            yield ("data", df_all, args.single_csv)
        elif explicit_tv:
            yield ("train", df_train, tr_path)
            yield ("val", df_val, va_path)
            yield ("test", df_test, args.test_csv)
        else:
            yield ("train_val", df_train_val, args.train_val_csv)
            yield ("test", df_test, args.test_csv)

    for label, frame, csv_path in _frames_iter():
        if vc not in frame.columns:
            raise ValueError(
                f"Value column {vc!r} not found in {label} CSV {csv_path!r}. "
                f"Columns: {list(frame.columns)}. Pass --value-column with an existing column name."
            )

    if single_file_mode:
        ref_frame = df_all
    elif explicit_tv:
        ref_frame = df_train
    else:
        ref_frame = df_train_val
    if args.use_all_numeric_features:
        ignore_cols = {"TIMESTAMP", "Acceleration RMS (smoothed)"}
        feature_cols = [
            c
            for c in ref_frame.columns
            if c not in ignore_cols and pd.api.types.is_numeric_dtype(ref_frame[c])
        ]
    elif args.feature_columns:
        feature_cols = list(args.feature_columns)
    else:
        feature_cols = [vc]

    for label, frame, csv_path in _frames_iter():
        missing = [c for c in feature_cols if c not in frame.columns]
        if missing:
            raise ValueError(
                f"Feature columns missing in {label} CSV {csv_path!r}: {missing}. "
                f"Available columns: {list(frame.columns)}"
            )

    tw_pre = int(getattr(args, "target_smoothing_window", 1))
    if tw_pre > 1:
        w_pre = tw_pre if tw_pre % 2 == 1 else tw_pre + 1
        print(
            f"Target pre-smoothing: centered MA window={w_pre} on {vc!r} "
            "(per CSV, applied before train/val/test windows)."
        )

        def _smooth_value_col(df: pd.DataFrame) -> None:
            df[vc] = smooth_target_series_1d(df[vc].to_numpy(dtype=np.float32), w_pre)

        if single_file_mode:
            _smooth_value_col(df_all)
        elif explicit_tv:
            _smooth_value_col(df_train)
            _smooth_value_col(df_val)
            _smooth_value_col(df_test)
        else:
            _smooth_value_col(df_train_val)
            _smooth_value_col(df_test)

    test_series = df_test[vc].to_numpy(dtype=np.float32)
    test_features = df_test[feature_cols].to_numpy(dtype=np.float32)

    if explicit_tv:
        train_series = df_train[vc].to_numpy(dtype=np.float32)
        val_series = df_val[vc].to_numpy(dtype=np.float32)
        train_features = df_train[feature_cols].to_numpy(dtype=np.float32)
        val_features = df_val[feature_cols].to_numpy(dtype=np.float32)
        tv_series = tv_features = tv_feat_norm = tv_target_norm = None
        train_end_idx = None
        full_series = full_features = None
    elif single_file_mode:
        full_series = df_all[vc].to_numpy(dtype=np.float32)
        full_features = df_all[feature_cols].to_numpy(dtype=np.float32)
        train_series = val_series = train_features = val_features = None
        tv_series = tv_features = tv_feat_norm = tv_target_norm = None
        train_end_idx = None
    else:
        full_series = full_features = None
        train_series = val_series = train_features = val_features = None
        tv_series = df_train_val[vc].to_numpy(dtype=np.float32)
        tv_features = df_train_val[feature_cols].to_numpy(dtype=np.float32)

    n_features = int(len(feature_cols))
    setattr(args, "input_dim", n_features)
    setattr(args, "feature_columns_resolved", feature_cols)
    setattr(args, "data_split_explicit_tv", explicit_tv)
    setattr(args, "data_split_single_csv", single_file_mode)
    print(f"Target column: {vc}")
    print(f"Input feature columns ({n_features}): {feature_cols}")
    if single_file_mode:
        print(f"Data split: single CSV ({args.single_csv}), chronological window ratios")
        print(f"Rows in file: {len(full_series)}")
    elif explicit_tv:
        print("Data split: explicit train / val / test CSVs")
        print("(Val holdout fraction --val-ratio is ignored when --train-csv and --val-csv are set.)")
        print(f"Train rows   : {len(train_series)}  ({tr_path})")
        print(f"Val rows     : {len(val_series)}  ({va_path})")
        print(f"Test rows    : {len(test_series)}  ({args.test_csv})")
    else:
        print("Data split: train_val CSV + test CSV; val = tail fraction of train_val windows")
        print(f"Train+Val series length : {len(tv_series)}")
        print(f"Test series length      : {len(test_series)}")
    if args.require_uniform_timestep:
        print(
            f"Uniform timestep windows: nominal step {args.uniform_step_seconds}s "
            f"(±{args.uniform_step_tolerance_seconds}s per adjacent pair)."
        )
        if single_file_mode:
            summarize_timestamp_steps(
                df_all["TIMESTAMP"],
                "single",
                args.uniform_step_seconds,
                args.uniform_step_tolerance_seconds,
            )
        elif explicit_tv:
            summarize_timestamp_steps(
                df_train["TIMESTAMP"], "train",
                args.uniform_step_seconds, args.uniform_step_tolerance_seconds,
            )
            summarize_timestamp_steps(
                df_val["TIMESTAMP"], "val",
                args.uniform_step_seconds, args.uniform_step_tolerance_seconds,
            )
            summarize_timestamp_steps(
                df_test["TIMESTAMP"], "test",
                args.uniform_step_seconds, args.uniform_step_tolerance_seconds,
            )
        else:
            summarize_timestamp_steps(
                df_train_val["TIMESTAMP"], "train_val",
                args.uniform_step_seconds, args.uniform_step_tolerance_seconds,
            )
            summarize_timestamp_steps(
                df_test["TIMESTAMP"], "test",
                args.uniform_step_seconds, args.uniform_step_tolerance_seconds,
            )

    if single_file_mode:
        train_mean = float("nan")
        train_std = float("nan")
    elif explicit_tv:
        train_mean = train_series.mean()
        train_std = train_series.std() + 1e-8
        feat_mean = train_features.mean(axis=0)
        feat_std = train_features.std(axis=0) + 1e-8
        train_feat_norm = (train_features - feat_mean) / feat_std
        val_feat_norm = (val_features - feat_mean) / feat_std
        test_feat_norm = (test_features - feat_mean) / feat_std
        train_target_norm = (train_series - train_mean) / train_std
        val_target_norm = (val_series - train_mean) / train_std
        test_target_norm = (test_series - train_mean) / train_std
        print(f"train_mean: {train_mean:.6f}")
        print(f"train_std : {train_std:.6f}")
    else:
        train_end_idx = int(len(tv_series) * (1 - args.val_ratio))
        train_mean = tv_series[:train_end_idx].mean()
        train_std = tv_series[:train_end_idx].std() + 1e-8
        feat_train = tv_features[:train_end_idx]
        feat_mean = feat_train.mean(axis=0)
        feat_std = feat_train.std(axis=0) + 1e-8
        tv_feat_norm = (tv_features - feat_mean) / feat_std
        test_feat_norm = (test_features - feat_mean) / feat_std
        tv_target_norm = (tv_series - train_mean) / train_std
        test_target_norm = (test_series - train_mean) / train_std
        train_feat_norm = val_feat_norm = train_target_norm = val_target_norm = None
        print(f"train_mean: {train_mean:.6f}")
        print(f"train_std : {train_std:.6f}")

    experiment_results = []

    for input_len in args.input_lens:
        for pred_len in args.pred_lens:
            try:
                span = input_len + pred_len
                uniform_kw = dict(
                    span_len=span,
                    nominal_seconds=args.uniform_step_seconds,
                    tolerance_seconds=args.uniform_step_tolerance_seconds,
                )
                if single_file_mode:
                    T = int(len(full_series))
                    if args.require_uniform_timestep:
                        all_valid = compute_uniform_timestep_start_indices(df_all["TIMESTAMP"], **uniform_kw)
                        n_slide = max(0, T - span + 1)
                        print(
                            f"  Single CSV uniform windows: {len(all_valid)}/{n_slide} valid starts "
                            f"(INPUT_LEN={input_len}, PRED_LEN={pred_len})."
                        )
                        if len(all_valid) == 0 and n_slide > 0:
                            print(
                                "  [hint] 0 valid uniform-step windows. The data sampling rate likely "
                                "does not match --uniform-step-seconds. Inspect the [step-stats:single] "
                                "line above (median/mode = the most common gap in seconds) and either "
                                "pass --uniform-step-seconds <observed_step> "
                                "(e.g. --uniform-step-seconds 1800 for 30-min cadence), widen "
                                "--uniform-step-tolerance-seconds, or pass "
                                "--no-require-uniform-timestep to disable the filter."
                            )
                    else:
                        n_slide = max(0, T - span + 1)
                        all_valid = np.arange(n_slide, dtype=np.int64) if n_slide > 0 else np.zeros(0, dtype=np.int64)
                        print(
                            f"  Single CSV dense sliding: {len(all_valid)} starts "
                            f"(INPUT_LEN={input_len}, PRED_LEN={pred_len})."
                        )
                    M = int(all_valid.shape[0])
                    rte_loop = (
                        args.test_ratio
                        if args.test_ratio is not None
                        else (1.0 - args.train_ratio - args.val_ratio)
                    )
                    n_tr_w, n_va_w, n_te_w = split_window_counts(
                        M,
                        args.train_ratio,
                        args.val_ratio,
                        rte_loop,
                        min_each=int(args.min_windows_per_split),
                    )
                    train_starts_arr = all_valid[:n_tr_w]
                    val_starts_arr = all_valid[n_tr_w : n_tr_w + n_va_w]
                    test_starts_arr = all_valid[n_tr_w + n_va_w :]
                    print(
                        f"    Window splits: train={n_tr_w} ({n_tr_w / max(M, 1):.4f}), "
                        f"val={n_va_w}, test={n_te_w} (target ratios "
                        f"{args.train_ratio:.4f}/{args.val_ratio:.4f}/{rte_loop:.4f})"
                    )
                    row_tr = row_indices_covered_by_windows(train_starts_arr, span, T)
                    train_mean = float(np.mean(full_series[row_tr]))
                    train_std = float(np.std(full_series[row_tr])) + 1e-8
                    feat_mean = np.mean(full_features[row_tr], axis=0)
                    feat_std = np.std(full_features[row_tr], axis=0) + 1e-8
                    feat_norm_all = (full_features - feat_mean) / feat_std
                    target_norm_all = (full_series - train_mean) / train_std
                    train_dataset = MultiStepDeltaDataset(
                        feat_norm_all,
                        target_norm_all,
                        input_len=input_len,
                        pred_len=pred_len,
                        sample_starts=train_starts_arr,
                    )
                    val_dataset = MultiStepDeltaDataset(
                        feat_norm_all,
                        target_norm_all,
                        input_len=input_len,
                        pred_len=pred_len,
                        sample_starts=val_starts_arr,
                    )
                    test_dataset = MultiStepDeltaDataset(
                        feat_norm_all,
                        target_norm_all,
                        input_len=input_len,
                        pred_len=pred_len,
                        sample_starts=test_starts_arr,
                    )
                    tv_starts = None
                elif args.require_uniform_timestep:
                    if explicit_tv:
                        train_starts = compute_uniform_timestep_start_indices(df_train["TIMESTAMP"], **uniform_kw)
                        val_starts = compute_uniform_timestep_start_indices(df_val["TIMESTAMP"], **uniform_kw)
                        test_starts = compute_uniform_timestep_start_indices(df_test["TIMESTAMP"], **uniform_kw)
                        n_tr_s = max(0, len(train_target_norm) - span + 1)
                        n_va_s = max(0, len(val_target_norm) - span + 1)
                        n_te_s = max(0, len(test_target_norm) - span + 1)
                        print(
                            f"  Uniform windows: train {len(train_starts)}/{n_tr_s} | "
                            f"val {len(val_starts)}/{n_va_s} | "
                            f"test {len(test_starts)}/{n_te_s} "
                            f"(INPUT_LEN={input_len}, PRED_LEN={pred_len})."
                        )
                    else:
                        tv_starts = compute_uniform_timestep_start_indices(df_train_val["TIMESTAMP"], **uniform_kw)
                        test_starts = compute_uniform_timestep_start_indices(df_test["TIMESTAMP"], **uniform_kw)
                        train_starts = val_starts = None
                        n_tv_sliding = max(0, len(tv_target_norm) - span + 1)
                        n_test_sliding = max(0, len(test_target_norm) - span + 1)
                        print(
                            f"  Uniform windows kept: train_val {len(tv_starts)}/{n_tv_sliding} | "
                            f"test {len(test_starts)}/{n_test_sliding} "
                            f"(INPUT_LEN={input_len}, PRED_LEN={pred_len})."
                        )

                    if explicit_tv:
                        train_dataset = MultiStepDeltaDataset(
                            train_feat_norm,
                            train_target_norm,
                            input_len=input_len,
                            pred_len=pred_len,
                            sample_starts=train_starts,
                        )
                        val_dataset = MultiStepDeltaDataset(
                            val_feat_norm,
                            val_target_norm,
                            input_len=input_len,
                            pred_len=pred_len,
                            sample_starts=val_starts,
                        )
                        test_dataset = MultiStepDeltaDataset(
                            test_feat_norm,
                            test_target_norm,
                            input_len=input_len,
                            pred_len=pred_len,
                            sample_starts=test_starts,
                        )
                    else:
                        tv_dataset = MultiStepDeltaDataset(
                            tv_feat_norm,
                            tv_target_norm,
                            input_len=input_len,
                            pred_len=pred_len,
                            sample_starts=tv_starts,
                        )
                        test_dataset = MultiStepDeltaDataset(
                            test_feat_norm,
                            test_target_norm,
                            input_len=input_len,
                            pred_len=pred_len,
                            sample_starts=test_starts,
                        )
                        n_tv = len(tv_dataset)
                        n_train = int(n_tv * (1 - args.val_ratio))
                        train_dataset = Subset(tv_dataset, range(0, n_train))
                        val_dataset = Subset(tv_dataset, range(n_train, n_tv))
                else:
                    tv_starts = train_starts = val_starts = test_starts = None
                    if explicit_tv:
                        train_dataset = MultiStepDeltaDataset(
                            train_feat_norm,
                            train_target_norm,
                            input_len=input_len,
                            pred_len=pred_len,
                            sample_starts=train_starts,
                        )
                        val_dataset = MultiStepDeltaDataset(
                            val_feat_norm,
                            val_target_norm,
                            input_len=input_len,
                            pred_len=pred_len,
                            sample_starts=val_starts,
                        )
                        test_dataset = MultiStepDeltaDataset(
                            test_feat_norm,
                            test_target_norm,
                            input_len=input_len,
                            pred_len=pred_len,
                            sample_starts=test_starts,
                        )
                    else:
                        tv_dataset = MultiStepDeltaDataset(
                            tv_feat_norm,
                            tv_target_norm,
                            input_len=input_len,
                            pred_len=pred_len,
                            sample_starts=tv_starts,
                        )
                        test_dataset = MultiStepDeltaDataset(
                            test_feat_norm,
                            test_target_norm,
                            input_len=input_len,
                            pred_len=pred_len,
                            sample_starts=test_starts,
                        )
                        n_tv = len(tv_dataset)
                        n_train = int(n_tv * (1 - args.val_ratio))
                        train_dataset = Subset(tv_dataset, range(0, n_train))
                        val_dataset = Subset(tv_dataset, range(n_train, n_tv))

                train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
                test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

                model = model_factory(input_len, pred_len, args, device)
                criterion = TrajectoryAwareLoss(
                    pred_len=pred_len,
                    delta=args.loss_huber_delta,
                    point_weight=args.loss_point_weight,
                    diff_weight=args.loss_diff_weight,
                    curvature_weight=args.loss_curvature_weight,
                    variance_weight=args.loss_variance_weight,
                    laplacian_reg_weight=args.loss_laplacian_weight,
                    tail_weight=args.loss_tail_weight,
                ).to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=args.scheduler_factor,
                    patience=args.scheduler_patience,
                )

                best_val_loss = float("inf")
                best_val_window_rmse = float("inf")
                best_state = copy.deepcopy(model.state_dict())
                history = []
                patience_counter = 0

                print(f"\n--- Training experiment: INPUT_LEN={input_len}, PRED_LEN={pred_len} ---")
                print(
                    f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)} | Test samples: {len(test_dataset)}"
                )
                if args.pred_smoothing_window > 1:
                    w = (
                        args.pred_smoothing_window
                        if args.pred_smoothing_window % 2 == 1
                        else args.pred_smoothing_window + 1
                    )
                    print(
                        f"Test forecast post-smoothing: centered MA window={w} "
                        "(applied to predictions only; validation RMSE during training is unsmoothed)."
                    )

                for epoch in range(1, args.epochs + 1):
                    train_loss = run_epoch(model, train_loader, criterion, optimizer, device)
                    val_loss = run_epoch(model, val_loader, criterion, optimizer=None, device=device)
                    val_window_rmse = compute_window_rmse(model, val_loader, train_std, train_mean, device)
                    scheduler.step(val_window_rmse)
                    current_lr = optimizer.param_groups[0]["lr"]

                    history.append(
                        {
                            "epoch": epoch,
                            "train_loss": train_loss,
                            "val_loss": val_loss,
                            "val_window_rmse": val_window_rmse,
                            "lr": current_lr,
                        }
                    )

                    print(
                        f"Epoch {epoch:03d} | train={train_loss:.6f} | val={val_loss:.6f} "
                        f"| val_window_rmse={val_window_rmse:.6f} | lr={current_lr:.2e}"
                    )

                    if val_window_rmse < best_val_window_rmse - args.min_delta:
                        best_val_window_rmse = val_window_rmse
                        best_val_loss = val_loss
                        best_state = copy.deepcopy(model.state_dict())
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= args.early_stopping_patience:
                            print(f"Early stopping triggered at epoch {epoch}.")
                            break

                model.load_state_dict(best_state)
                model.eval()

                all_preds_raw, all_targets_raw = collect_predictions(model, test_loader, train_std, train_mean, device)
                if args.pred_smoothing_window > 1:
                    all_preds_raw = smooth_forecast_horizons(all_preds_raw, args.pred_smoothing_window)
                metrics = evaluate_metrics(all_targets_raw, all_preds_raw)
                baseline = baseline_rmse(test_loader, pred_len, train_std, train_mean)
                horizon_rmse = [rmse_np(all_targets_raw[:, h], all_preds_raw[:, h]) for h in range(pred_len)]
                horizon_phase_mape = compute_horizon_phase_mapes(all_targets_raw, all_preds_raw)
                horizon_phase_h_ranges = horizon_phase_step_ranges(pred_len)

                sample_idx = min(args.plot_sample_idx, len(test_dataset) - 1)
                x, y_delta, last_val = test_dataset[sample_idx]
                with torch.no_grad():
                    pred_delta = model(x.unsqueeze(0).to(device)).cpu().numpy()[0]

                last_val = float(last_val.numpy()[0])
                pred_raw = (pred_delta + last_val) * train_std + train_mean
                true_raw = (y_delta.numpy() + last_val) * train_std + train_mean
                if args.pred_smoothing_window > 1:
                    pred_raw = smooth_forecast_vector(pred_raw, args.pred_smoothing_window)

                ts_off = int(test_dataset.sample_starts[sample_idx])
                pred_ts = df_test["TIMESTAMP"].iloc[ts_off + input_len : ts_off + input_len + pred_len]

                experiment_results.append(
                    {
                        "input_len": input_len,
                        "pred_len": pred_len,
                        "model_state_dict": copy.deepcopy(model.state_dict()),
                        "best_val_loss": best_val_loss,
                        "best_val_window_rmse": best_val_window_rmse,
                        "history": pd.DataFrame(history),
                        "metrics": metrics,
                        "baseline_rmse": baseline,
                        "horizon_rmse": horizon_rmse,
                        "horizon_phase_mape": horizon_phase_mape,
                        "horizon_phase_h_ranges": horizon_phase_h_ranges,
                        "all_preds_raw": all_preds_raw,
                        "all_targets_raw": all_targets_raw,
                        "sample_pred_raw": pred_raw,
                        "sample_true_raw": true_raw,
                        "sample_timestamps": pred_ts,
                        "test_sample_starts": np.asarray(test_dataset.sample_starts, dtype=np.int64).copy(),
                        "train_mean": float(train_mean),
                        "train_std": float(train_std),
                    }
                )
            except ValueError as exc:
                print(f"Skipping INPUT_LEN={input_len}, PRED_LEN={pred_len}: {exc}")

    if not experiment_results:
        raise RuntimeError("No valid experiment completed. Reduce INPUT_LEN or PRED_LEN.")

    def _phase_mape_row(result):
        row = {
            "input_len": result["input_len"],
            "pred_len": result["pred_len"],
            "best_val_loss": result["best_val_loss"],
            "best_val_window_rmse": result["best_val_window_rmse"],
            "test_mse": result["metrics"]["mse"],
            "test_rmse": result["metrics"]["rmse"],
            "test_mae": result["metrics"]["mae"],
            "test_mape": result["metrics"]["mape"],
            "test_r2": result["metrics"]["r2"],
            "baseline_rmse": result["baseline_rmse"],
            "test_mape_horizon_phases": horizon_phase_ranges_csv_string(result["horizon_phase_h_ranges"]),
        }
        for p in range(HORIZON_PHASE_COUNT):
            row[f"test_mape_phase_{p + 1}"] = result["horizon_phase_mape"][p]
        return row

    summary_df = pd.DataFrame([_phase_mape_row(r) for r in experiment_results]).sort_values(
        by="best_val_window_rmse"
    ).reset_index(drop=True)

    summary_path = os.path.join(args.output_dir, "experiment_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print("\nExperiment summary:")
    print(summary_df)
    print(f"Saved summary to: {summary_path}")

    best_result = min(experiment_results, key=lambda item: item["best_val_window_rmse"])
    best_input_len = best_result["input_len"]
    best_pred_len = best_result["pred_len"]

    print(
        f"\nBest config -> INPUT_LEN={best_input_len}, PRED_LEN={best_pred_len}, "
        f"best_val_loss={best_result['best_val_loss']:.6f}, "
        f"best_val_window_rmse={best_result['best_val_window_rmse']:.6f}, "
        f"test_rmse={best_result['metrics']['rmse']:.6f}"
    )

    checkpoint_path = os.path.join(args.output_dir, args.checkpoint_name)
    best_checkpoint = {
        "model_state_dict": best_result["model_state_dict"],
        "best_val_loss": float(best_result["best_val_loss"]),
        "best_val_window_rmse": float(best_result["best_val_window_rmse"]),
        "train_mean": float(best_result["train_mean"]),
        "train_std": float(best_result["train_std"]),
        "input_len": int(best_input_len),
        "pred_len": int(best_pred_len),
        "model_config": model_config_factory(args, best_input_len, best_pred_len),
        "summary": summary_df.to_dict(orient="records"),
    }
    torch.save(best_checkpoint, checkpoint_path)
    print(f"Saved best model to: {checkpoint_path}")

    history_path = os.path.join(args.output_dir, "best_history.csv")
    best_result["history"].to_csv(history_path, index=False)

    metrics_path = os.path.join(args.output_dir, "best_metrics.json")
    metrics_payload = {
        "best_input_len": int(best_input_len),
        "best_pred_len": int(best_pred_len),
        "best_val_loss": float(best_result["best_val_loss"]),
        "best_val_window_rmse": float(best_result["best_val_window_rmse"]),
        "metrics": best_result["metrics"],
        "baseline_rmse": float(best_result["baseline_rmse"]),
        "train_mean": float(best_result["train_mean"]),
        "train_std": float(best_result["train_std"]),
        "horizon_phase_mape_pct": [
            float(x) if np.isfinite(x) else None for x in best_result["horizon_phase_mape"]
        ],
        "horizon_phase_step_ranges_1based": [
            {"h_start": a, "h_end": b, "mape_pct": float(m) if np.isfinite(m) else None}
            for (a, b), m in zip(best_result["horizon_phase_h_ranges"], best_result["horizon_phase_mape"])
        ],
    }
    with open(metrics_path, "w", encoding="utf-8") as fp:
        json.dump(metrics_payload, fp, indent=2)

    _ts_cfg = int(getattr(args, "target_smoothing_window", 1))
    target_smooth_stored = int(1 if _ts_cfg <= 1 else (_ts_cfg if _ts_cfg % 2 == 1 else _ts_cfg + 1))

    best_config_path = os.path.join(args.output_dir, "best_config.json")
    best_config_payload = {
        "data_split_mode": (
            "single_csv_window_ratios"
            if single_file_mode
            else ("explicit_train_val_test" if explicit_tv else "train_val_holdout")
        ),
        "single_csv": args.single_csv if single_file_mode else None,
        "train_ratio_window": float(args.train_ratio) if single_file_mode else None,
        "val_ratio_window": float(args.val_ratio) if single_file_mode else None,
        "test_ratio_window": float(getattr(args, "test_ratio_resolved"))
        if single_file_mode
        else None,
        "min_windows_per_split": int(args.min_windows_per_split) if single_file_mode else None,
        "train_csv": tr_path if explicit_tv else None,
        "val_csv": va_path if explicit_tv else None,
        "train_val_csv": None if (explicit_tv or single_file_mode) else args.train_val_csv,
        "test_csv": None if single_file_mode else args.test_csv,
        "value_column": args.value_column,
        "feature_columns": args.feature_columns_resolved,
        "use_all_numeric_features": bool(args.use_all_numeric_features),
        "output_dir": args.output_dir,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "val_ratio": (
            None
            if explicit_tv or single_file_mode
            else float(args.val_ratio)
        ),
        "early_stopping_patience": args.early_stopping_patience,
        "scheduler_patience": args.scheduler_patience,
        "scheduler_factor": args.scheduler_factor,
        "min_delta": args.min_delta,
        "loss_huber_delta": args.loss_huber_delta,
        "loss_point_weight": args.loss_point_weight,
        "loss_diff_weight": args.loss_diff_weight,
        "loss_curvature_weight": args.loss_curvature_weight,
        "loss_variance_weight": args.loss_variance_weight,
        "loss_laplacian_weight": args.loss_laplacian_weight,
        "loss_tail_weight": float(args.loss_tail_weight),
        "pred_smoothing_window": args.pred_smoothing_window,
        "target_smoothing_window": target_smooth_stored,
        "save_window_plots": args.save_window_plots,
        "rolling_window_artifact_limit": args.rolling_window_artifact_limit,
        "require_uniform_timestep": bool(args.require_uniform_timestep),
        "uniform_step_seconds": float(args.uniform_step_seconds),
        "uniform_step_tolerance_seconds": float(args.uniform_step_tolerance_seconds),
        "best_input_len": int(best_input_len),
        "best_pred_len": int(best_pred_len),
        "model_config": model_config_factory(args, best_input_len, best_pred_len),
        "best_val_window_rmse": float(best_result["best_val_window_rmse"]),
        "test_rmse": float(best_result["metrics"]["rmse"]),
        "test_mse": float(best_result["metrics"]["mse"]),
        "test_mae": float(best_result["metrics"]["mae"]),
        "test_mape": float(best_result["metrics"]["mape"]),
        "test_r2": float(best_result["metrics"]["r2"]),
        "horizon_phase_mape_pct": [
            float(x) if np.isfinite(x) else None for x in best_result["horizon_phase_mape"]
        ],
        "horizon_phase_step_ranges_1based": [
            {"h_start": a, "h_end": b, "mape_pct": float(m) if np.isfinite(m) else None}
            for (a, b), m in zip(best_result["horizon_phase_h_ranges"], best_result["horizon_phase_mape"])
        ],
    }
    with open(best_config_path, "w", encoding="utf-8") as fp:
        json.dump(best_config_payload, fp, indent=2)

    horizon_path = os.path.join(args.output_dir, "best_horizon_rmse.csv")
    pd.DataFrame(
        {
            "horizon": np.arange(1, len(best_result["horizon_rmse"]) + 1),
            "rmse": best_result["horizon_rmse"],
        }
    ).to_csv(horizon_path, index=False)

    phase_mape_csv = os.path.join(args.output_dir, "best_horizon_phase_mape.csv")
    pd.DataFrame(
        [
            {
                "phase": i + 1,
                "h_start": best_result["horizon_phase_h_ranges"][i][0],
                "h_end": best_result["horizon_phase_h_ranges"][i][1],
                "mape_pct": best_result["horizon_phase_mape"][i],
            }
            for i in range(HORIZON_PHASE_COUNT)
        ]
    ).to_csv(phase_mape_csv, index=False)

    sample_path = os.path.join(args.output_dir, "best_sample_forecast.csv")
    pd.DataFrame(
        {
            "timestamp": best_result["sample_timestamps"].astype(str).to_list(),
            "actual": best_result["sample_true_raw"],
            "predicted": best_result["sample_pred_raw"],
        }
    ).to_csv(sample_path, index=False)

    save_plot(
        path=os.path.join(args.output_dir, "best_learning_curve.png"),
        title=f"Learning Curve (INPUT_LEN={best_input_len}, PRED_LEN={best_pred_len})",
        x_label="Epoch",
        y_label="Loss",
        x=best_result["history"]["epoch"],
        y1=best_result["history"]["train_loss"],
        y1_label="Train loss",
        y2=best_result["history"]["val_loss"],
        y2_label="Val loss",
    )

    save_plot(
        path=os.path.join(args.output_dir, "best_sample_forecast.png"),
        title=f"Single Forecast Window - Test (INPUT_LEN={best_input_len}, PRED_LEN={best_pred_len})",
        x_label="Date",
        y_label=args.value_column,
        x=best_result["sample_timestamps"],
        y1=best_result["sample_true_raw"],
        y1_label="Actual forecast",
        y2=best_result["sample_pred_raw"],
        y2_label="Predicted forecast",
        rotate_dates=True,
    )

    starts = np.asarray(best_result["test_sample_starts"], dtype=np.int64)
    h = 0
    h_pred = best_result["all_preds_raw"][:, h]
    h_true = best_result["all_targets_raw"][:, h]
    ts_rows = starts + best_input_len + h
    ts_h1 = df_test["TIMESTAMP"].iloc[ts_rows.tolist()].reset_index(drop=True)

    horizon_1_path = os.path.join(args.output_dir, "best_horizon_1_forecast.csv")
    build_horizon_forecast_dataframe(
        timestamps=ts_h1,
        actual=h_true,
        predicted=h_pred,
        horizon=1,
    ).to_csv(horizon_1_path, index=False)

    save_plot(
        path=os.path.join(args.output_dir, "best_horizon_1.png"),
        title=f"Horizon-1 Forecast - Test (INPUT_LEN={best_input_len}, PRED_LEN={best_pred_len})",
        x_label="Date",
        y_label=args.value_column,
        x=ts_h1,
        y1=h_true,
        y1_label="Actual",
        y2=h_pred,
        y2_label="Predicted",
        rotate_dates=True,
    )

    h = min(11, best_pred_len - 1)
    h_pred = best_result["all_preds_raw"][:, h]
    h_true = best_result["all_targets_raw"][:, h]
    ts_rows_n = starts + best_input_len + h
    ts_hn = df_test["TIMESTAMP"].iloc[ts_rows_n.tolist()].reset_index(drop=True)

    horizon_n_path = os.path.join(args.output_dir, f"best_horizon_{h + 1}_forecast.csv")
    build_horizon_forecast_dataframe(
        timestamps=ts_hn,
        actual=h_true,
        predicted=h_pred,
        horizon=h + 1,
    ).to_csv(horizon_n_path, index=False)

    preds_per_h = best_result["all_preds_raw"]
    targets_per_h = best_result["all_targets_raw"]
    horizon_arr = np.arange(1, best_pred_len + 1)
    pred_mean_per_h = preds_per_h.mean(axis=0)
    pred_std_per_h = preds_per_h.std(axis=0)
    target_mean_per_h = targets_per_h.mean(axis=0)
    target_std_per_h = targets_per_h.std(axis=0)
    bias_per_h = pred_mean_per_h - target_mean_per_h
    horizon_bias_df = pd.DataFrame(
        {
            "horizon": horizon_arr,
            "pred_mean": pred_mean_per_h,
            "pred_std": pred_std_per_h,
            "target_mean": target_mean_per_h,
            "target_std": target_std_per_h,
            "bias_pred_minus_target": bias_per_h,
        }
    )
    horizon_bias_csv_path = os.path.join(args.output_dir, "horizon_bias.csv")
    horizon_bias_df.to_csv(horizon_bias_csv_path, index=False)

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax_top.plot(horizon_arr, target_mean_per_h, color="C0", label="Actual mean")
    ax_top.fill_between(
        horizon_arr,
        target_mean_per_h - target_std_per_h,
        target_mean_per_h + target_std_per_h,
        color="C0",
        alpha=0.15,
        label="Actual ±1 std",
    )
    ax_top.plot(horizon_arr, pred_mean_per_h, color="C1", label="Predicted mean")
    ax_top.fill_between(
        horizon_arr,
        pred_mean_per_h - pred_std_per_h,
        pred_mean_per_h + pred_std_per_h,
        color="C1",
        alpha=0.15,
        label="Predicted ±1 std",
    )
    ax_top.set_title(
        f"Per-horizon mean & spread across {preds_per_h.shape[0]} test windows "
        f"(INPUT_LEN={best_input_len}, PRED_LEN={best_pred_len}, "
        f"loss_tail_weight={float(args.loss_tail_weight):g})"
    )
    ax_top.set_ylabel(args.value_column)
    ax_top.grid(True, alpha=0.35)
    ax_top.legend(loc="best", fontsize=8)

    ax_bot.axhline(0.0, color="0.55", linewidth=0.9)
    ax_bot.plot(horizon_arr, bias_per_h, color="C3", label="Predicted - Actual (mean)")
    ax_bot.set_xlabel("Horizon step")
    ax_bot.set_ylabel("Mean bias")
    ax_bot.grid(True, alpha=0.35)
    ax_bot.legend(loc="best", fontsize=8)
    fig.tight_layout()
    horizon_bias_png_path = os.path.join(args.output_dir, "horizon_bias.png")
    fig.savefig(horizon_bias_png_path, dpi=140)
    plt.close(fig)

    # Rolling-window input context matches training targets: --value-column after optional
    # target pre-smoothing (not a separate raw column), so overlays stay coherent with horizons.
    rolling_input_hist = test_series
    rolling_input_label = str(args.value_column)

    rolling_windows_dir, rolling_combined_csv_path = save_rolling_window_forecasts(
        output_dir=args.output_dir,
        preds_raw=best_result["all_preds_raw"],
        targets_raw=best_result["all_targets_raw"],
        timestamps=df_test["TIMESTAMP"],
        input_len=best_input_len,
        pred_len=best_pred_len,
        save_plots=args.save_window_plots,
        max_per_window_artifacts=args.rolling_window_artifact_limit,
        y_axis_label=args.value_column,
        history_series_raw=rolling_input_hist,
        window_input_row_starts=starts,
        input_context_label=rolling_input_label,
        pred_smoothing_window=args.pred_smoothing_window,
    )

    save_plot(
        path=os.path.join(args.output_dir, f"best_horizon_{h + 1}.png"),
        title=f"Horizon-{h + 1} Forecast - Test (INPUT_LEN={best_input_len}, PRED_LEN={best_pred_len})",
        x_label="Date",
        y_label=args.value_column,
        x=ts_hn,
        y1=h_true,
        y1_label="Actual",
        y2=h_pred,
        y2_label="Predicted",
        rotate_dates=True,
    )

    print("\nBest-run metrics:")
    print(f"Test MSE : {best_result['metrics']['mse']:.6f}")
    print(f"Test RMSE: {best_result['metrics']['rmse']:.6f}")
    print(f"Test MAE : {best_result['metrics']['mae']:.6f}")
    print(f"Test MAPE: {best_result['metrics']['mape']:.6f}%")
    print(f"Test R2  : {best_result['metrics']['r2']:.6f}")
    print(f"Baseline RMSE: {best_result['baseline_rmse']:.6f}")

    print("\nRMSE by horizon for best run:")
    for i, horizon_rmse in enumerate(best_result["horizon_rmse"], start=1):
        print(f"Horizon {i:02d} RMSE: {horizon_rmse:.6f}")

    print(
        "\nTest MAPE by horizon phase "
        f"({HORIZON_PHASE_COUNT} contiguous bands over 1..PRED_LEN, pooled over test windows):"
    )
    for i in range(HORIZON_PHASE_COUNT):
        hs, he = best_result["horizon_phase_h_ranges"][i]
        mp = best_result["horizon_phase_mape"][i]
        if hs is None or not np.isfinite(mp):
            print(f"  Phase {i + 1}: (empty) MAPE=n/a")
        else:
            print(f"  Phase {i + 1} (steps {hs}-{he}): MAPE={mp:.6f}%")

    print("\nSaved artifacts:")
    print(f"- {summary_path}")
    print(f"- {history_path}")
    print(f"- {metrics_path}")
    print(f"- {best_config_path}")
    print(f"- {horizon_path}")
    print(f"- {phase_mape_csv}")
    print(f"- {horizon_1_path}")
    print(f"- {horizon_n_path}")
    print(f"- {horizon_bias_csv_path}")
    print(f"- {horizon_bias_png_path}")
    print(f"- {rolling_windows_dir}")
    print(f"- {rolling_combined_csv_path}")
    print(f"- {sample_path}")
    print(f"- {checkpoint_path}")
