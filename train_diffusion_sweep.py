import argparse
import copy
import json
import math
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run diffusion forecasting sweep on pre-split train/val and anomalous test data."
    )
    parser.add_argument("--train-val-csv", type=str, default="data_train_val.csv")
    parser.add_argument("--test-csv", type=str, default="data_test_anomalous.csv")
    parser.add_argument("--output-dir", type=str, default="outputs/diffusion_sweep")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--input-lens", type=int, nargs="+", default=[432, 864, 1728])
    parser.add_argument("--pred-lens", type=int, nargs="+", default=[36, 72, 108])
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--context-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=2e-2)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--scheduler-patience", type=int, default=3)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--plot-sample-idx", type=int, default=200)
    parser.add_argument("--num-sample-paths", type=int, default=5)
    parser.add_argument("--checkpoint-name", type=str, default="diffusion_delta_date_split_best.pth")
    return parser.parse_args()


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


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps):
        half_dim = self.dim // 2
        freq_factor = math.log(10000) / max(half_dim - 1, 1)
        freqs = torch.exp(torch.arange(half_dim, device=timesteps.device) * -freq_factor)
        angles = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([angles.sin(), angles.cos()], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class DiffusionForecaster(nn.Module):
    def __init__(self, pred_len, hidden_size=128, context_dim=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.pred_len = pred_len
        self.history_encoder = nn.GRU(
            input_size=1,
            hidden_size=context_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.time_embedding = SinusoidalTimeEmbedding(hidden_size)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.input_proj = nn.Linear(pred_len, hidden_size)
        self.context_proj = nn.Linear(context_dim, hidden_size)
        self.residual_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, hidden_size),
                )
                for _ in range(4)
            ]
        )
        self.output_proj = nn.Linear(hidden_size, pred_len)

    def forward(self, history, noisy_target, timesteps):
        history = history.permute(0, 2, 1)
        _, hidden = self.history_encoder(history)
        context = hidden[-1]

        time_emb = self.time_mlp(self.time_embedding(timesteps))
        hidden_state = self.input_proj(noisy_target) + self.context_proj(context) + time_emb

        for layer in self.residual_layers:
            hidden_state = hidden_state + layer(hidden_state)

        return self.output_proj(hidden_state)


class DiffusionSchedule(nn.Module):
    def __init__(self, num_steps, beta_start, beta_end):
        super().__init__()
        betas = torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]], dim=0)

        self.num_steps = num_steps
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance.clamp(min=1e-12))

    def extract(self, buffer_name, timesteps, shape):
        values = getattr(self, buffer_name).gather(0, timesteps)
        return values.view(-1, *([1] * (len(shape) - 1)))

    def q_sample(self, x_start, timesteps, noise):
        sqrt_alpha_bar = self.extract("sqrt_alphas_cumprod", timesteps, x_start.shape)
        sqrt_one_minus_alpha_bar = self.extract("sqrt_one_minus_alphas_cumprod", timesteps, x_start.shape)
        return sqrt_alpha_bar * x_start + sqrt_one_minus_alpha_bar * noise

    def predict_start_from_noise(self, x_t, timesteps, noise):
        sqrt_alpha_bar = self.extract("sqrt_alphas_cumprod", timesteps, x_t.shape)
        sqrt_one_minus_alpha_bar = self.extract("sqrt_one_minus_alphas_cumprod", timesteps, x_t.shape)
        return (x_t - sqrt_one_minus_alpha_bar * noise) / sqrt_alpha_bar.clamp(min=1e-8)

    def p_sample(self, model, history, x_t, timestep):
        batch_size = x_t.size(0)
        timesteps = torch.full((batch_size,), timestep, device=x_t.device, dtype=torch.long)
        pred_noise = model(history, x_t, timesteps)
        beta_t = self.extract("betas", timesteps, x_t.shape)
        sqrt_one_minus_alpha_bar_t = self.extract("sqrt_one_minus_alphas_cumprod", timesteps, x_t.shape)
        sqrt_recip_alpha_t = self.extract("sqrt_recip_alphas", timesteps, x_t.shape)

        model_mean = sqrt_recip_alpha_t * (x_t - beta_t * pred_noise / sqrt_one_minus_alpha_bar_t.clamp(min=1e-8))

        if timestep == 0:
            return model_mean

        posterior_variance_t = self.extract("posterior_variance", timesteps, x_t.shape)
        noise = torch.randn_like(x_t)
        return model_mean + torch.sqrt(posterior_variance_t) * noise


class DiffusionLoss(nn.Module):
    def __init__(self, pred_len):
        super().__init__()
        w = torch.linspace(1.0, 0.3, pred_len, dtype=torch.float32)
        self.register_buffer("w", w / w.mean())

    def forward(self, pred_noise, true_noise):
        loss = (pred_noise - true_noise) ** 2
        return (loss * self.w.to(pred_noise.device)).mean()


def run_epoch(model, schedule, loader, criterion, optimizer=None, device="cpu"):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    total_count = 0

    with torch.set_grad_enabled(training):
        for history, target_delta in loader:
            history = history.to(device)
            target_delta = target_delta.to(device)

            timesteps = torch.randint(0, schedule.num_steps, (history.size(0),), device=device)
            noise = torch.randn_like(target_delta)
            noisy_target = schedule.q_sample(target_delta, timesteps, noise)
            pred_noise = model(history, noisy_target, timesteps)
            loss = criterion(pred_noise, noise)

            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            batch_size = history.size(0)
            total_loss += loss.item() * batch_size
            total_count += batch_size

    return total_loss / max(total_count, 1)


def sample_future_delta(model, schedule, history, num_paths=1):
    history = history.unsqueeze(0)
    history = history.repeat(num_paths, 1, 1)
    sample = torch.randn((num_paths, model.pred_len), device=history.device)

    model.eval()
    with torch.no_grad():
        for timestep in reversed(range(schedule.num_steps)):
            sample = schedule.p_sample(model, history, sample, timestep)
    return sample


def collect_predictions(model, schedule, loader, train_std, train_mean, device, num_paths=1):
    all_preds_abs_norm = []
    all_targets_abs_norm = []

    for history, target_delta in loader:
        history = history.to(device)
        target_delta = target_delta.to(device)

        batch_preds = []
        for idx in range(history.size(0)):
            sampled_deltas = sample_future_delta(model, schedule, history[idx], num_paths=num_paths)
            batch_preds.append(sampled_deltas.mean(dim=0).cpu())

        pred_delta = torch.stack(batch_preds, dim=0).to(device)
        last_val = history[:, 0, -1].unsqueeze(1)

        pred_abs_norm = pred_delta + last_val
        true_abs_norm = target_delta + last_val

        all_preds_abs_norm.append(pred_abs_norm.cpu().numpy())
        all_targets_abs_norm.append(true_abs_norm.cpu().numpy())

    all_preds_abs_norm = np.concatenate(all_preds_abs_norm, axis=0)
    all_targets_abs_norm = np.concatenate(all_targets_abs_norm, axis=0)

    all_preds_raw = all_preds_abs_norm * train_std + train_mean
    all_targets_raw = all_targets_abs_norm * train_std + train_mean
    return all_preds_raw, all_targets_raw


def baseline_rmse(test_loader, pred_len, train_std, train_mean):
    baseline_preds_abs_norm = []
    baseline_targets_abs_norm = []

    for history, target_delta in test_loader:
        history_np = history.numpy()
        target_delta_np = target_delta.numpy()

        last_val = history_np[:, 0, -1][:, None]
        pred_abs_norm = np.repeat(last_val, pred_len, axis=1)
        true_abs_norm = target_delta_np + last_val

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


def build_forecast_dataframe(preds_raw, targets_raw, timestamps, input_len):
    rows = []
    num_samples, pred_len = preds_raw.shape

    for sample_idx in range(num_samples):
        for horizon_idx in range(pred_len):
            ts_idx = input_len + sample_idx + horizon_idx
            rows.append(
                {
                    "sample_index": sample_idx,
                    "horizon": horizon_idx + 1,
                    "timestamp": timestamps.iloc[ts_idx],
                    "actual": targets_raw[sample_idx, horizon_idx],
                    "predicted": preds_raw[sample_idx, horizon_idx],
                }
            )

    return pd.DataFrame(rows)


def train_one_experiment(
    input_len,
    pred_len,
    tv_norm,
    test_norm,
    val_ratio,
    args,
    train_std,
    train_mean,
    test_timestamps,
    device,
):
    tv_dataset = MultiStepDeltaDataset(tv_norm, input_len=input_len, pred_len=pred_len)
    test_dataset = MultiStepDeltaDataset(test_norm, input_len=input_len, pred_len=pred_len)

    n_tv = len(tv_dataset)
    n_train = int(n_tv * (1 - val_ratio))

    train_dataset = Subset(tv_dataset, range(0, n_train))
    val_dataset = Subset(tv_dataset, range(n_train, n_tv))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = DiffusionForecaster(
        pred_len=pred_len,
        hidden_size=args.hidden_size,
        context_dim=args.context_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    schedule = DiffusionSchedule(
        num_steps=args.diffusion_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
    ).to(device)
    criterion = DiffusionLoss(pred_len=pred_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
    )

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    history = []
    patience_counter = 0

    print(f"\\n--- Training experiment: INPUT_LEN={input_len}, PRED_LEN={pred_len} ---")
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)} | Test samples: {len(test_dataset)}")

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, schedule, train_loader, criterion, optimizer, device)
        val_loss = run_epoch(model, schedule, val_loader, criterion, optimizer=None, device=device)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": current_lr,
            }
        )

        print(f"Epoch {epoch:03d} | train={train_loss:.6f} | val={val_loss:.6f} | lr={current_lr:.2e}")

        if val_loss < best_val_loss - args.min_delta:
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

    all_preds_raw, all_targets_raw = collect_predictions(
        model,
        schedule,
        test_loader,
        train_std,
        train_mean,
        device,
        num_paths=args.num_sample_paths,
    )
    metrics = evaluate_metrics(all_targets_raw, all_preds_raw)
    baseline = baseline_rmse(test_loader, pred_len, train_std, train_mean)
    horizon_rmse = [rmse_np(all_targets_raw[:, h], all_preds_raw[:, h]) for h in range(pred_len)]

    sample_idx = min(args.plot_sample_idx, len(test_dataset) - 1)
    history_x, target_delta = test_dataset[sample_idx]
    sampled_deltas = sample_future_delta(
        model,
        schedule,
        history_x.to(device),
        num_paths=args.num_sample_paths,
    ).mean(dim=0).cpu().numpy()

    last_val = history_x.numpy()[0, -1]
    pred_raw = (sampled_deltas + last_val) * train_std + train_mean
    true_raw = (target_delta.numpy() + last_val) * train_std + train_mean

    ts_offset = input_len + sample_idx
    pred_ts = test_timestamps.iloc[ts_offset: ts_offset + pred_len]

    return {
        "input_len": input_len,
        "pred_len": pred_len,
        "model_state_dict": copy.deepcopy(model.state_dict()),
        "best_val_loss": best_val_loss,
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


def main():
    args = parse_args()
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
                result = train_one_experiment(
                    input_len=input_len,
                    pred_len=pred_len,
                    tv_norm=tv_norm,
                    test_norm=test_norm,
                    val_ratio=args.val_ratio,
                    args=args,
                    train_std=train_std,
                    train_mean=train_mean,
                    test_timestamps=df_test["TIMESTAMP"],
                    device=device,
                )
                experiment_results.append(result)
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
                "test_rmse": result["metrics"]["rmse"],
                "test_mae": result["metrics"]["mae"],
                "test_r2": result["metrics"]["r2"],
                "baseline_rmse": result["baseline_rmse"],
            }
            for result in experiment_results
        ]
    ).sort_values(by="best_val_loss").reset_index(drop=True)

    summary_path = os.path.join(args.output_dir, "experiment_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print("\\nExperiment summary:")
    print(summary_df)
    print(f"Saved summary to: {summary_path}")

    best_result = min(experiment_results, key=lambda item: item["best_val_loss"])
    best_input_len = best_result["input_len"]
    best_pred_len = best_result["pred_len"]

    print(
        f"\\nBest config -> INPUT_LEN={best_input_len}, PRED_LEN={best_pred_len}, "
        f"best_val_loss={best_result['best_val_loss']:.6f}, "
        f"test_rmse={best_result['metrics']['rmse']:.6f}"
    )

    checkpoint_path = os.path.join(args.output_dir, args.checkpoint_name)
    best_checkpoint = {
        "model_state_dict": best_result["model_state_dict"],
        "best_val_loss": float(best_result["best_val_loss"]),
        "train_mean": float(train_mean),
        "train_std": float(train_std),
        "input_len": int(best_input_len),
        "pred_len": int(best_pred_len),
        "model_config": {
            "pred_len": int(best_pred_len),
            "hidden_size": args.hidden_size,
            "context_dim": args.context_dim,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "diffusion_steps": args.diffusion_steps,
            "beta_start": args.beta_start,
            "beta_end": args.beta_end,
            "num_sample_paths": args.num_sample_paths,
        },
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

    test_predictions_path = os.path.join(args.output_dir, "best_test_predictions.npy")
    test_targets_path = os.path.join(args.output_dir, "best_test_targets.npy")
    np.save(test_predictions_path, best_result["all_preds_raw"])
    np.save(test_targets_path, best_result["all_targets_raw"])

    forecast_table_path = os.path.join(args.output_dir, "best_test_forecasts.csv")
    build_forecast_dataframe(
        preds_raw=best_result["all_preds_raw"],
        targets_raw=best_result["all_targets_raw"],
        timestamps=df_test["TIMESTAMP"],
        input_len=best_input_len,
    ).to_csv(forecast_table_path, index=False)

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

    print("\\nBest-run metrics:")
    print(f"Test MSE : {best_result['metrics']['mse']:.6f}")
    print(f"Test RMSE: {best_result['metrics']['rmse']:.6f}")
    print(f"Test MAE : {best_result['metrics']['mae']:.6f}")
    print(f"Test R2  : {best_result['metrics']['r2']:.6f}")
    print(f"Baseline RMSE: {best_result['baseline_rmse']:.6f}")

    print("\\nRMSE by horizon for best run:")
    for i, horizon_rmse in enumerate(best_result["horizon_rmse"], start=1):
        print(f"Horizon {i:02d} RMSE: {horizon_rmse:.6f}")

    print("\\nSaved artifacts:")
    print(f"- {summary_path}")
    print(f"- {history_path}")
    print(f"- {metrics_path}")
    print(f"- {horizon_path}")
    print(f"- {test_predictions_path}")
    print(f"- {test_targets_path}")
    print(f"- {forecast_table_path}")
    print(f"- {sample_path}")
    print(f"- {checkpoint_path}")


if __name__ == "__main__":
    main()
