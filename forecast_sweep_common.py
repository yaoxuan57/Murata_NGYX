import copy
import json
import os
import random
from typing import Callable, Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset


def add_common_args(parser, default_output_dir: str, default_checkpoint_name: str):
    parser.add_argument("--train-val-csv", type=str, default="data_train_val.csv")
    parser.add_argument("--test-csv", type=str, default="data_test_anomalous.csv")
    parser.add_argument("--output-dir", type=str, default=default_output_dir)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--input-lens", type=int, nargs="+", default=[432])
    parser.add_argument("--pred-lens", type=int, nargs="+", default=[36])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--scheduler-patience", type=int, default=3)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--plot-sample-idx", type=int, default=200)
    parser.add_argument("--checkpoint-name", type=str, default=default_checkpoint_name)
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
        "r2": r2_np(y_true, y_pred),
    }


class MultiStepDeltaDataset(Dataset):
    def __init__(self, series_norm, input_len, pred_len):
        series_norm = np.asarray(series_norm, dtype=np.float32)
        self.input_len = input_len
        self.pred_len = pred_len
        self.n_samples = len(series_norm) - input_len - pred_len + 1

        if self.n_samples <= 0:
            raise ValueError("Series too short for given input_len and pred_len.")

        x = np.array(
            [series_norm[i:i + input_len] for i in range(self.n_samples)],
            dtype=np.float32,
        )[:, np.newaxis, :]

        future = np.array(
            [series_norm[i + input_len:i + input_len + pred_len] for i in range(self.n_samples)],
            dtype=np.float32,
        )

        last_val = x[:, 0, -1][:, None]
        y_delta = future - last_val

        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y_delta, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class TrajectoryAwareLoss(nn.Module):
    def __init__(
        self,
        pred_len,
        delta=1.0,
        point_weight=0.4,
        diff_weight=1.2,
        curvature_weight=0.8,
        variance_weight=0.4,
    ):
        super().__init__()
        self.delta = delta
        self.point_weight = point_weight
        self.diff_weight = diff_weight
        self.curvature_weight = curvature_weight
        self.variance_weight = variance_weight
        w = torch.linspace(1.0, 0.8, pred_len, dtype=torch.float32)
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

        return (
            self.point_weight * point_loss
            + self.diff_weight * diff_loss
            + self.curvature_weight * curvature_loss
            + self.variance_weight * variance_loss
        )


def run_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    total_count = 0

    with torch.set_grad_enabled(training):
        for x, y_delta in loader:
            x = x.to(device)
            y_delta = y_delta.to(device)

            pred_delta = model(x)
            last_val = x[:, 0, -1].unsqueeze(1)
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
        for x, y_delta in loader:
            x = x.to(device)
            y_delta = y_delta.to(device)

            pred_delta = model(x)
            last_val = x[:, 0, -1].unsqueeze(1)

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

    for x, y_delta in test_loader:
        x_np = x.numpy()
        y_delta_np = y_delta.numpy()

        last_val = x_np[:, 0, -1][:, None]
        pred_abs_norm = np.repeat(last_val, pred_len, axis=1)
        true_abs_norm = y_delta_np + last_val

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


def save_rolling_window_forecasts(output_dir, preds_raw, targets_raw, timestamps, input_len, pred_len):
    windows_dir = os.path.join(output_dir, "rolling_window_forecasts")
    plots_dir = os.path.join(windows_dir, "plots")
    csv_dir = os.path.join(windows_dir, "csv")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    all_rows = []
    n_windows = preds_raw.shape[0]

    for window_idx in range(n_windows):
        start_idx = input_len + window_idx
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

        window_csv_path = os.path.join(csv_dir, f"window_{window_idx:06d}.csv")
        window_df.to_csv(window_csv_path, index=False)
        all_rows.append(window_df)

        plt.figure(figsize=(8, 3))
        plt.plot(window_df["step_ahead"], window_df["actual"], label="Actual")
        plt.plot(window_df["step_ahead"], window_df["predicted"], label="Predicted")
        plt.title(f"Window {window_idx} Forecast ({pred_len}-step)")
        plt.xlabel("Step Ahead")
        plt.ylabel("Acceleration RMS")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        window_plot_path = os.path.join(plots_dir, f"window_{window_idx:06d}.png")
        plt.savefig(window_plot_path, dpi=140)
        plt.close()

    combined_df = pd.concat(all_rows, ignore_index=True)
    combined_csv_path = os.path.join(windows_dir, "all_windows_forecasts.csv")
    combined_df.to_csv(combined_csv_path, index=False)

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

    df_train_val = pd.read_csv(args.train_val_csv)
    df_test = pd.read_csv(args.test_csv)

    df_train_val["TIMESTAMP"] = pd.to_datetime(df_train_val["TIMESTAMP"])
    df_test["TIMESTAMP"] = pd.to_datetime(df_test["TIMESTAMP"])

    tv_series = df_train_val["Acceleration RMS"].to_numpy(dtype=np.float32)
    test_series = df_test["Acceleration RMS"].to_numpy(dtype=np.float32)

    print(f"Train+Val series length : {len(tv_series)}")
    print(f"Test series length      : {len(test_series)}")

    train_end_idx = int(len(tv_series) * (1 - args.val_ratio))
    train_mean = tv_series[:train_end_idx].mean()
    train_std = tv_series[:train_end_idx].std() + 1e-8

    tv_norm = (tv_series - train_mean) / train_std
    test_norm = (test_series - train_mean) / train_std

    print(f"train_mean: {train_mean:.6f}")
    print(f"train_std : {train_std:.6f}")

    experiment_results = []

    for input_len in args.input_lens:
        for pred_len in args.pred_lens:
            try:
                tv_dataset = MultiStepDeltaDataset(tv_norm, input_len=input_len, pred_len=pred_len)
                test_dataset = MultiStepDeltaDataset(test_norm, input_len=input_len, pred_len=pred_len)

                n_tv = len(tv_dataset)
                n_train = int(n_tv * (1 - args.val_ratio))

                train_dataset = Subset(tv_dataset, range(0, n_train))
                val_dataset = Subset(tv_dataset, range(n_train, n_tv))

                train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
                test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

                model = model_factory(input_len, pred_len, args, device)
                criterion = TrajectoryAwareLoss(pred_len=pred_len, delta=1.0).to(device)
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
                metrics = evaluate_metrics(all_targets_raw, all_preds_raw)
                baseline = baseline_rmse(test_loader, pred_len, train_std, train_mean)
                horizon_rmse = [rmse_np(all_targets_raw[:, h], all_preds_raw[:, h]) for h in range(pred_len)]

                sample_idx = min(args.plot_sample_idx, len(test_dataset) - 1)
                x, y_delta = test_dataset[sample_idx]
                with torch.no_grad():
                    pred_delta = model(x.unsqueeze(0).to(device)).cpu().numpy()[0]

                last_val = x.numpy()[0, -1]
                pred_raw = (pred_delta + last_val) * train_std + train_mean
                true_raw = (y_delta.numpy() + last_val) * train_std + train_mean

                ts_offset = input_len + sample_idx
                pred_ts = df_test["TIMESTAMP"].iloc[ts_offset: ts_offset + pred_len]

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
                        "all_preds_raw": all_preds_raw,
                        "all_targets_raw": all_targets_raw,
                        "sample_pred_raw": pred_raw,
                        "sample_true_raw": true_raw,
                        "sample_timestamps": pred_ts,
                    }
                )
            except ValueError as exc:
                print(f"Skipping INPUT_LEN={input_len}, PRED_LEN={pred_len}: {exc}")

    if not experiment_results:
        raise RuntimeError("No valid experiment completed. Reduce INPUT_LEN or PRED_LEN.")

    summary_df = pd.DataFrame(
        [
            {
                "input_len": result["input_len"],
                "pred_len": result["pred_len"],
                "best_val_loss": result["best_val_loss"],
                "best_val_window_rmse": result["best_val_window_rmse"],
                "test_rmse": result["metrics"]["rmse"],
                "test_mae": result["metrics"]["mae"],
                "test_r2": result["metrics"]["r2"],
                "baseline_rmse": result["baseline_rmse"],
            }
            for result in experiment_results
        ]
    ).sort_values(by="best_val_window_rmse").reset_index(drop=True)

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
        "train_mean": float(train_mean),
        "train_std": float(train_std),
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
    }
    with open(metrics_path, "w", encoding="utf-8") as fp:
        json.dump(metrics_payload, fp, indent=2)

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
        y_label="Acceleration RMS",
        x=best_result["sample_timestamps"],
        y1=best_result["sample_true_raw"],
        y1_label="Actual forecast",
        y2=best_result["sample_pred_raw"],
        y2_label="Predicted forecast",
        rotate_dates=True,
    )

    h = 0
    h_pred = best_result["all_preds_raw"][:, h]
    h_true = best_result["all_targets_raw"][:, h]
    ts_h1 = df_test["TIMESTAMP"].iloc[best_input_len + h: best_input_len + h + len(h_pred)]

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
        y_label="Acceleration RMS",
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
    ts_hn = df_test["TIMESTAMP"].iloc[best_input_len + h: best_input_len + h + len(h_pred)]

    horizon_n_path = os.path.join(args.output_dir, f"best_horizon_{h + 1}_forecast.csv")
    build_horizon_forecast_dataframe(
        timestamps=ts_hn,
        actual=h_true,
        predicted=h_pred,
        horizon=h + 1,
    ).to_csv(horizon_n_path, index=False)

    rolling_windows_dir, rolling_combined_csv_path = save_rolling_window_forecasts(
        output_dir=args.output_dir,
        preds_raw=best_result["all_preds_raw"],
        targets_raw=best_result["all_targets_raw"],
        timestamps=df_test["TIMESTAMP"],
        input_len=best_input_len,
        pred_len=best_pred_len,
    )

    save_plot(
        path=os.path.join(args.output_dir, f"best_horizon_{h + 1}.png"),
        title=f"Horizon-{h + 1} Forecast - Test (INPUT_LEN={best_input_len}, PRED_LEN={best_pred_len})",
        x_label="Date",
        y_label="Acceleration RMS",
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
    print(f"Test R2  : {best_result['metrics']['r2']:.6f}")
    print(f"Baseline RMSE: {best_result['baseline_rmse']:.6f}")

    print("\nRMSE by horizon for best run:")
    for i, horizon_rmse in enumerate(best_result["horizon_rmse"], start=1):
        print(f"Horizon {i:02d} RMSE: {horizon_rmse:.6f}")

    print("\nSaved artifacts:")
    print(f"- {summary_path}")
    print(f"- {history_path}")
    print(f"- {metrics_path}")
    print(f"- {horizon_path}")
    print(f"- {horizon_1_path}")
    print(f"- {horizon_n_path}")
    print(f"- {rolling_windows_dir}")
    print(f"- {rolling_combined_csv_path}")
    print(f"- {sample_path}")
    print(f"- {checkpoint_path}")
