"""
Rolling-window ARIMA sweep (univariate target only). Matches data-split conventions in
`forecast_sweep_common.run_sweep`: single CSV ratios, explicit train/val/test CSVs, or
train_val + test. Optional centered moving average on the target column **before** splits
(`--target-smoothing-window`).
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from forecast_sweep_common import (
    add_common_args,
    build_horizon_forecast_dataframe,
    compute_timestep_window_start_indices,
    evaluate_metrics,
    parse_timestamp_series,
    rmse_np,
    row_indices_covered_by_windows,
    save_plot,
    save_rolling_window_forecasts,
    set_seed,
    smooth_forecast_horizons,
    smooth_forecast_vector,
    split_window_counts,
    summarize_timestamp_steps,
    window_start_index_kwargs_from_args,
)


def smooth_series_1d(vec: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average with edge padding (same construction as forecast smoothing)."""
    if window <= 1:
        return np.asarray(vec, dtype=np.float32)
    w = window if window % 2 == 1 else window + 1
    pad = w // 2
    y = np.asarray(vec, dtype=np.float64)
    kernel = np.ones(w, dtype=np.float64) / w
    y_pad = np.pad(y, (pad, pad), mode="edge")
    out = np.convolve(y_pad, kernel, mode="valid")
    return out.astype(np.float32)


def apply_target_smoothing(
    *,
    single_file_mode: bool,
    explicit_tv: bool,
    df_all: Optional[pd.DataFrame],
    df_train: Optional[pd.DataFrame],
    df_val: Optional[pd.DataFrame],
    df_test: pd.DataFrame,
    df_train_val: Optional[pd.DataFrame],
    vc: str,
    window: int,
) -> None:
    if window <= 1:
        return
    if single_file_mode:
        df_all[vc] = smooth_series_1d(df_all[vc].to_numpy(), window)
    elif explicit_tv:
        df_train[vc] = smooth_series_1d(df_train[vc].to_numpy(), window)
        df_val[vc] = smooth_series_1d(df_val[vc].to_numpy(), window)
        df_test[vc] = smooth_series_1d(df_test[vc].to_numpy(), window)
    else:
        df_train_val[vc] = smooth_series_1d(df_train_val[vc].to_numpy(), window)
        df_test[vc] = smooth_series_1d(df_test[vc].to_numpy(), window)


def fit_forecast_arima(
    history: np.ndarray,
    pred_len: int,
    order: Tuple[int, int, int],
    maxiter: int,
) -> np.ndarray:
    """Return length-pred_len forecast; naive flat last value on failure."""
    try:
        from statsmodels.tsa.arima.model import ARIMA

        hist = np.asarray(history, dtype=np.float64)
        if np.any(~np.isfinite(hist)):
            hist = np.nan_to_num(hist, nan=np.nanmedian(hist))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ARIMA(hist, order=order)
            res = model.fit(method_kwargs={"warn_convergence": False}, disp=False, maxiter=maxiter)
            fc = res.forecast(steps=pred_len)
        out = np.asarray(fc, dtype=np.float64).reshape(-1)
        if out.shape[0] != pred_len:
            raise ValueError("unexpected forecast length")
        return out
    except Exception:
        last = float(history[-1]) if len(history) else 0.0
        return np.full(pred_len, last, dtype=np.float64)


def collect_arima_preds(
    series: np.ndarray,
    starts: np.ndarray,
    input_len: int,
    pred_len: int,
    order: Tuple[int, int, int],
    maxiter: int,
) -> Tuple[np.ndarray, np.ndarray]:
    starts = np.asarray(starts, dtype=np.int64)
    n = starts.shape[0]
    preds = np.empty((n, pred_len), dtype=np.float64)
    targets = np.empty((n, pred_len), dtype=np.float64)
    for i, s in enumerate(starts):
        s = int(s)
        hist = series[s : s + input_len]
        fut = series[s + input_len : s + input_len + pred_len]
        preds[i] = fit_forecast_arima(hist, pred_len, order, maxiter)
        targets[i] = fut
    return preds.astype(np.float32), targets.astype(np.float32)


def mean_window_rmse(preds: np.ndarray, targets: np.ndarray) -> float:
    per_win = np.sqrt(np.mean((targets - preds) ** 2, axis=1))
    return float(np.mean(per_win))


def baseline_rmse_flat(series: np.ndarray, starts: np.ndarray, input_len: int, pred_len: int) -> float:
    starts = np.asarray(starts, dtype=np.int64)
    preds = []
    tars = []
    for s in starts:
        s = int(s)
        last = float(series[s + input_len - 1])
        preds.append(np.full(pred_len, last, dtype=np.float64))
        tars.append(series[s + input_len : s + input_len + pred_len].astype(np.float64))
    p = np.stack(preds, axis=0)
    t = np.stack(tars, axis=0)
    return rmse_np(t, p)


def parse_args():
    parser = argparse.ArgumentParser(description="ARIMA rolling-window sweep (univariate)")
    add_common_args(
        parser,
        default_output_dir="outputs_arima_sweep",
        default_checkpoint_name="arima_meta.json",
    )
    parser.add_argument("--arima-p", type=int, default=2)
    parser.add_argument("--arima-d", type=int, default=1)
    parser.add_argument("--arima-q", type=int, default=2)
    parser.add_argument(
        "--arima-maxiter",
        type=int,
        default=100,
        help="Max iterations for statsmodels ARIMA fit.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    tr_path = args.train_csv
    va_path = args.val_csv
    if (tr_path is None) ^ (va_path is None):
        raise ValueError("Set both --train-csv and --val-csv, or neither.")
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
                f"--train-ratio, --val-ratio, and --test-ratio must sum to 1; got {ssum:.6f}."
            )
        setattr(args, "test_ratio_resolved", float(rte_resolve))
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

    def frames_iter():
        if single_file_mode:
            yield ("data", df_all, args.single_csv)
        elif explicit_tv:
            yield ("train", df_train, tr_path)
            yield ("val", df_val, va_path)
            yield ("test", df_test, args.test_csv)
        else:
            yield ("train_val", df_train_val, args.train_val_csv)
            yield ("test", df_test, args.test_csv)

    for label, frame, csv_path in frames_iter():
        if vc not in frame.columns:
            raise ValueError(
                f"Value column {vc!r} not found in {label} CSV {csv_path!r}. Columns: {list(frame.columns)}."
            )

    if args.feature_columns or args.use_all_numeric_features:
        print(
            "Note: ARIMA uses only --value-column (exogenous / multi-feature inputs are ignored)."
        )

    df_plot_ref = df_all if single_file_mode else df_test

    tw = args.target_smoothing_window
    if tw > 1:
        wuse = tw if tw % 2 == 1 else tw + 1
        print(f"Target pre-smoothing: centered MA window={wuse} on {vc!r} (per CSV, before splits).")
        apply_target_smoothing(
            single_file_mode=single_file_mode,
            explicit_tv=explicit_tv,
            df_all=df_all,
            df_train=df_train,
            df_val=df_val,
            df_test=df_test,
            df_train_val=df_train_val,
            vc=vc,
            window=wuse,
        )

    order = (int(args.arima_p), int(args.arima_d), int(args.arima_q))
    print(f"ARIMA order (p,d,q)={order}")

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
    if args.max_consecutive_timestamp_gap_seconds is not None:
        print(
            f"Max TIMESTAMP gap inside any model window: each consecutive step must be "
            f"≤ {args.max_consecutive_timestamp_gap_seconds:g} s (windows crossing larger gaps are dropped)."
        )

    test_series = df_test[vc].to_numpy(dtype=np.float32)
    if explicit_tv:
        train_series = df_train[vc].to_numpy(dtype=np.float32)
        val_series = df_val[vc].to_numpy(dtype=np.float32)
        tv_series = None
        full_series = None
    elif single_file_mode:
        full_series = df_all[vc].to_numpy(dtype=np.float32)
        train_series = val_series = None
        tv_series = None
    else:
        tv_series = df_train_val[vc].to_numpy(dtype=np.float32)
        train_series = val_series = full_series = None

    arima_order = order
    experiment_results: List[Dict] = []

    for input_len in args.input_lens:
        for pred_len in args.pred_lens:
            try:
                span = input_len + pred_len
                wk = window_start_index_kwargs_from_args(args, span)

                if single_file_mode:
                    T = int(len(full_series))
                    n_slide = max(0, T - span + 1)
                    if args.require_uniform_timestep or args.max_consecutive_timestamp_gap_seconds is not None:
                        all_valid = compute_timestep_window_start_indices(df_all["TIMESTAMP"], **wk)
                        print(
                            f"  Single CSV timestamp-filtered windows: {len(all_valid)}/{n_slide} valid starts "
                            f"(INPUT_LEN={input_len}, PRED_LEN={pred_len})."
                        )
                    else:
                        all_valid = np.arange(n_slide, dtype=np.int64) if n_slide > 0 else np.zeros(0, dtype=np.int64)
                        print(f"  Single CSV dense sliding: {len(all_valid)} starts.")

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
                    train_mean = float(np.mean(full_series[row_indices_covered_by_windows(train_starts_arr, span, T)]))
                    train_std = float(np.std(full_series[row_indices_covered_by_windows(train_starts_arr, span, T)])) + 1e-8
                    print(
                        f"    Window splits: train={n_tr_w}, val={n_va_w}, test={n_te_w} "
                        f"(mean/std from train windows only)"
                    )

                    val_preds, val_targets = collect_arima_preds(
                        full_series, val_starts_arr, input_len, pred_len, arima_order, args.arima_maxiter
                    )
                    best_val_window_rmse = mean_window_rmse(val_preds, val_targets)

                    test_preds, test_targets = collect_arima_preds(
                        full_series, test_starts_arr, input_len, pred_len, arima_order, args.arima_maxiter
                    )
                    if args.pred_smoothing_window > 1:
                        test_preds = smooth_forecast_horizons(test_preds, args.pred_smoothing_window)

                    test_starts_for_meta = test_starts_arr

                elif explicit_tv:
                    if args.require_uniform_timestep or args.max_consecutive_timestamp_gap_seconds is not None:
                        train_starts = compute_timestep_window_start_indices(df_train["TIMESTAMP"], **wk)
                        val_starts = compute_timestep_window_start_indices(df_val["TIMESTAMP"], **wk)
                        test_starts = compute_timestep_window_start_indices(df_test["TIMESTAMP"], **wk)
                    else:
                        n_tr = max(0, len(train_series) - span + 1)
                        n_va = max(0, len(val_series) - span + 1)
                        n_te = max(0, len(test_series) - span + 1)
                        train_starts = np.arange(n_tr, dtype=np.int64) if n_tr > 0 else np.zeros(0, dtype=np.int64)
                        val_starts = np.arange(n_va, dtype=np.int64) if n_va > 0 else np.zeros(0, dtype=np.int64)
                        test_starts = np.arange(n_te, dtype=np.int64) if n_te > 0 else np.zeros(0, dtype=np.int64)

                    row_tr = row_indices_covered_by_windows(train_starts, span, len(train_series))
                    train_mean = float(np.mean(train_series[row_tr]))
                    train_std = float(np.std(train_series[row_tr])) + 1e-8

                    val_preds, val_targets = collect_arima_preds(
                        val_series, val_starts, input_len, pred_len, arima_order, args.arima_maxiter
                    )
                    best_val_window_rmse = mean_window_rmse(val_preds, val_targets)

                    test_preds, test_targets = collect_arima_preds(
                        test_series, test_starts, input_len, pred_len, arima_order, args.arima_maxiter
                    )
                    if args.pred_smoothing_window > 1:
                        test_preds = smooth_forecast_horizons(test_preds, args.pred_smoothing_window)

                    test_starts_for_meta = test_starts

                else:
                    if args.require_uniform_timestep or args.max_consecutive_timestamp_gap_seconds is not None:
                        tv_starts = compute_timestep_window_start_indices(df_train_val["TIMESTAMP"], **wk)
                        test_starts = compute_timestep_window_start_indices(df_test["TIMESTAMP"], **wk)
                    else:
                        n_tv_sliding = max(0, len(tv_series) - span + 1)
                        n_test_sliding = max(0, len(test_series) - span + 1)
                        tv_starts = np.arange(n_tv_sliding, dtype=np.int64) if n_tv_sliding > 0 else np.zeros(0, dtype=np.int64)
                        test_starts = (
                            np.arange(n_test_sliding, dtype=np.int64)
                            if n_test_sliding > 0
                            else np.zeros(0, dtype=np.int64)
                        )

                    n_tv = int(tv_starts.shape[0])
                    n_train_w = int(n_tv * (1.0 - args.val_ratio))
                    train_starts = tv_starts[:n_train_w]
                    val_starts = tv_starts[n_train_w:]

                    tv_train_end_idx = int(len(tv_series) * (1.0 - args.val_ratio))
                    train_mean = float(np.mean(tv_series[:tv_train_end_idx]))
                    train_std = float(np.std(tv_series[:tv_train_end_idx])) + 1e-8

                    val_preds, val_targets = collect_arima_preds(
                        tv_series, val_starts, input_len, pred_len, arima_order, args.arima_maxiter
                    )
                    best_val_window_rmse = mean_window_rmse(val_preds, val_targets)

                    test_preds, test_targets = collect_arima_preds(
                        test_series, test_starts, input_len, pred_len, arima_order, args.arima_maxiter
                    )
                    if args.pred_smoothing_window > 1:
                        test_preds = smooth_forecast_horizons(test_preds, args.pred_smoothing_window)

                    test_starts_for_meta = test_starts

                metrics = evaluate_metrics(test_targets, test_preds)
                series_for_baseline = full_series if single_file_mode else test_series
                baseline = baseline_rmse_flat(series_for_baseline, test_starts_for_meta, input_len, pred_len)

                horizon_rmse = [rmse_np(test_targets[:, h], test_preds[:, h]) for h in range(pred_len)]

                sample_idx = min(args.plot_sample_idx, test_preds.shape[0] - 1)
                s0 = int(test_starts_for_meta[sample_idx])
                pred_raw = test_preds[sample_idx]
                true_raw = test_targets[sample_idx]
                if args.pred_smoothing_window > 1:
                    pred_raw = smooth_forecast_vector(pred_raw, args.pred_smoothing_window)
                pred_ts = df_plot_ref["TIMESTAMP"].iloc[s0 + input_len : s0 + input_len + pred_len]

                experiment_results.append(
                    {
                        "input_len": input_len,
                        "pred_len": pred_len,
                        "best_val_window_rmse": best_val_window_rmse,
                        "best_val_loss": float("nan"),
                        "metrics": metrics,
                        "baseline_rmse": baseline,
                        "horizon_rmse": horizon_rmse,
                        "all_preds_raw": test_preds,
                        "all_targets_raw": test_targets,
                        "sample_pred_raw": pred_raw,
                        "sample_true_raw": true_raw,
                        "sample_timestamps": pred_ts,
                        "test_sample_starts": np.asarray(test_starts_for_meta, dtype=np.int64).copy(),
                        "train_mean": float(train_mean),
                        "train_std": float(train_std),
                    }
                )

                print(
                    f"\n--- ARIMA INPUT_LEN={input_len}, PRED_LEN={pred_len} --- "
                    f"val_window_rmse={best_val_window_rmse:.6f} test_rmse={metrics['rmse']:.6f}"
                )
            except ValueError as exc:
                print(f"Skipping INPUT_LEN={input_len}, PRED_LEN={pred_len}: {exc}")

    if not experiment_results:
        raise RuntimeError("No valid ARIMA experiment completed.")

    summary_df = pd.DataFrame(
        [
            {
                "input_len": r["input_len"],
                "pred_len": r["pred_len"],
                "best_val_loss": r["best_val_loss"],
                "best_val_window_rmse": r["best_val_window_rmse"],
                "test_mse": r["metrics"]["mse"],
                "test_rmse": r["metrics"]["rmse"],
                "test_mae": r["metrics"]["mae"],
                "test_mape": r["metrics"]["mape"],
                "test_r2": r["metrics"]["r2"],
                "baseline_rmse": r["baseline_rmse"],
            }
            for r in experiment_results
        ]
    ).sort_values(by="best_val_window_rmse").reset_index(drop=True)

    summary_path = os.path.join(args.output_dir, "experiment_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print("\nExperiment summary:")
    print(summary_df)

    best_result = min(experiment_results, key=lambda item: item["best_val_window_rmse"])
    best_input_len = best_result["input_len"]
    best_pred_len = best_result["pred_len"]

    meta_path = os.path.join(args.output_dir, args.checkpoint_name)
    meta_payload = {
        "model_type": "arima",
        "order": list(arima_order),
        "best_input_len": int(best_input_len),
        "best_pred_len": int(best_pred_len),
        "best_val_window_rmse": float(best_result["best_val_window_rmse"]),
        "train_mean": float(best_result["train_mean"]),
        "train_std": float(best_result["train_std"]),
    }
    with open(meta_path, "w", encoding="utf-8") as fp:
        json.dump(meta_payload, fp, indent=2)
    print(f"Saved run metadata to: {meta_path}")

    history_path = os.path.join(args.output_dir, "best_history.csv")
    pd.DataFrame({"note": ["ARIMA has no epoch-wise training; placeholder row."]}).to_csv(history_path, index=False)

    metrics_path = os.path.join(args.output_dir, "best_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as fp:
        json.dump(
            {
                "best_input_len": int(best_input_len),
                "best_pred_len": int(best_pred_len),
                "best_val_window_rmse": float(best_result["best_val_window_rmse"]),
                "metrics": best_result["metrics"],
                "baseline_rmse": float(best_result["baseline_rmse"]),
                "train_mean": float(best_result["train_mean"]),
                "train_std": float(best_result["train_std"]),
            },
            fp,
            indent=2,
        )

    model_config = {
        "model_type": "arima",
        "arima_p": int(order[0]),
        "arima_d": int(order[1]),
        "arima_q": int(order[2]),
        "arima_maxiter": int(args.arima_maxiter),
    }

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
        "test_ratio_window": float(getattr(args, "test_ratio_resolved", 0.0)) if single_file_mode else None,
        "min_windows_per_split": int(args.min_windows_per_split) if single_file_mode else None,
        "train_csv": tr_path if explicit_tv else None,
        "val_csv": va_path if explicit_tv else None,
        "train_val_csv": None if (explicit_tv or single_file_mode) else args.train_val_csv,
        "test_csv": None if single_file_mode else args.test_csv,
        "value_column": args.value_column,
        "feature_columns": [args.value_column],
        "use_all_numeric_features": False,
        "output_dir": args.output_dir,
        "seed": args.seed,
        "pred_smoothing_window": args.pred_smoothing_window,
        "save_window_plots": args.save_window_plots,
        "rolling_window_artifact_limit": args.rolling_window_artifact_limit,
        "require_uniform_timestep": bool(args.require_uniform_timestep),
        "uniform_step_seconds": float(args.uniform_step_seconds),
        "uniform_step_tolerance_seconds": float(args.uniform_step_tolerance_seconds),
        "max_consecutive_timestamp_gap_seconds": (
            float(args.max_consecutive_timestamp_gap_seconds)
            if args.max_consecutive_timestamp_gap_seconds is not None
            else None
        ),
        "target_smoothing_window": int(
            1 if tw <= 1 else (tw if tw % 2 == 1 else tw + 1)
        ),
        "best_input_len": int(best_input_len),
        "best_pred_len": int(best_pred_len),
        "model_config": model_config,
        "best_val_window_rmse": float(best_result["best_val_window_rmse"]),
        "test_rmse": float(best_result["metrics"]["rmse"]),
        "supports_save_window_plots": True,
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

    sample_path = os.path.join(args.output_dir, "best_sample_forecast.csv")
    pd.DataFrame(
        {
            "timestamp": best_result["sample_timestamps"].astype(str).to_list(),
            "actual": best_result["sample_true_raw"],
            "predicted": best_result["sample_pred_raw"],
        }
    ).to_csv(sample_path, index=False)

    save_plot(
        path=os.path.join(args.output_dir, "best_sample_forecast.png"),
        title=f"ARIMA sample window (INPUT_LEN={best_input_len}, PRED_LEN={best_pred_len}, order={order})",
        x_label="Date",
        y_label=args.value_column,
        x=best_result["sample_timestamps"],
        y1=best_result["sample_true_raw"],
        y1_label="Actual",
        y2=best_result["sample_pred_raw"],
        y2_label="Predicted",
        rotate_dates=True,
    )

    starts = np.asarray(best_result["test_sample_starts"], dtype=np.int64)
    h = 0
    ts_rows = starts + best_input_len + h
    ts_h1 = df_plot_ref["TIMESTAMP"].iloc[ts_rows.tolist()].reset_index(drop=True)
    horizon_1_path = os.path.join(args.output_dir, "best_horizon_1_forecast.csv")
    build_horizon_forecast_dataframe(
        timestamps=ts_h1,
        actual=best_result["all_targets_raw"][:, h],
        predicted=best_result["all_preds_raw"][:, h],
        horizon=1,
    ).to_csv(horizon_1_path, index=False)

    rolling_input_hist = df_plot_ref[vc].to_numpy(dtype=np.float32)
    rolling_input_label = str(args.value_column)

    rolling_windows_dir, rolling_combined_csv_path = save_rolling_window_forecasts(
        output_dir=args.output_dir,
        preds_raw=best_result["all_preds_raw"],
        targets_raw=best_result["all_targets_raw"],
        timestamps=df_plot_ref["TIMESTAMP"],
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

    print("\nBest-run metrics:")
    print(f"Test RMSE: {best_result['metrics']['rmse']:.6f}")
    print(f"Rolling artifacts: {rolling_windows_dir}")


if __name__ == "__main__":
    main()
