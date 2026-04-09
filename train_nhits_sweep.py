import argparse

import torch
import torch.nn as nn

from forecast_sweep_common import add_common_args, run_sweep


class NHiTSBlock(nn.Module):
    def __init__(self, input_len, pred_len, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_len, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.backcast = nn.Linear(hidden_dim, input_len)
        self.forecast = nn.Linear(hidden_dim, pred_len)

    def forward(self, residual):
        h = self.net(residual)
        return self.backcast(h), self.forecast(h)


class NHiTSForecaster(nn.Module):
    def __init__(self, input_len, pred_len, hidden_dim=256, n_blocks=3, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList(
            [NHiTSBlock(input_len, pred_len, hidden_dim, dropout) for _ in range(n_blocks)]
        )

    def forward(self, x):
        residual = x[:, 0, :]
        forecast = 0.0
        for block in self.blocks:
            backcast, block_forecast = block(residual)
            residual = residual - backcast
            forecast = forecast + block_forecast
        return forecast


def make_model(input_len, pred_len, args, device):
    return NHiTSForecaster(
        input_len=input_len,
        pred_len=pred_len,
        hidden_dim=args.hidden_dim,
        n_blocks=args.n_blocks,
        dropout=args.dropout,
    ).to(device)


def make_model_config(args, input_len, pred_len):
    return {
        "model_type": "nhits",
        "input_len": input_len,
        "pred_len": pred_len,
        "hidden_dim": args.hidden_dim,
        "n_blocks": args.n_blocks,
        "dropout": args.dropout,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="N-HiTS sweep for delta forecasting")
    add_common_args(
        parser,
        default_output_dir="outputs_nhits_sweep",
        default_checkpoint_name="nhits_delta_huber_best.pth",
    )
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-blocks", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_sweep(args, model_factory=make_model, model_config_factory=make_model_config)
