import argparse
import math

import torch
import torch.nn as nn

from forecast_sweep_common import add_common_args, run_sweep


def trend_basis(length, degree, device):
    t = torch.linspace(0, 1, steps=length, device=device)
    basis = [t ** i for i in range(degree + 1)]
    return torch.stack(basis, dim=0)


def seasonality_basis(length, harmonics, device):
    t = torch.linspace(0, 2 * math.pi, steps=length, device=device)
    basis = [torch.ones_like(t)]
    for k in range(1, harmonics + 1):
        basis.append(torch.sin(k * t))
        basis.append(torch.cos(k * t))
    return torch.stack(basis, dim=0)


class NBeatsBasisBlock(nn.Module):
    def __init__(self, input_len, pred_len, hidden_dim, n_layers, n_theta, block_type):
        super().__init__()
        self.block_type = block_type
        self.input_len = input_len
        self.pred_len = pred_len

        layers = []
        in_dim = input_len
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.GELU()]
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)

        self.theta_backcast = nn.Linear(hidden_dim, n_theta)
        self.theta_forecast = nn.Linear(hidden_dim, n_theta)

    def _build_series(self, theta, length):
        if self.block_type == "trend":
            degree = theta.size(1) - 1
            basis = trend_basis(length, degree, theta.device)
        else:
            harmonics = max(1, (theta.size(1) - 1) // 2)
            basis = seasonality_basis(length, harmonics, theta.device)
            basis = basis[: theta.size(1), :]
        return theta @ basis

    def forward(self, x):
        h = self.backbone(x)
        theta_b = self.theta_backcast(h)
        theta_f = self.theta_forecast(h)

        backcast = self._build_series(theta_b, self.input_len)
        forecast = self._build_series(theta_f, self.pred_len)
        return backcast, forecast


class NBeatsXForecaster(nn.Module):
    def __init__(
        self,
        input_len,
        pred_len,
        hidden_dim=256,
        n_layers=3,
        trend_blocks=2,
        seasonality_blocks=2,
        trend_degree=3,
        seasonality_harmonics=8,
    ):
        super().__init__()
        self.input_len = input_len
        self.pred_len = pred_len

        self.trend_stack = nn.ModuleList(
            [
                NBeatsBasisBlock(
                    input_len=input_len,
                    pred_len=pred_len,
                    hidden_dim=hidden_dim,
                    n_layers=n_layers,
                    n_theta=trend_degree + 1,
                    block_type="trend",
                )
                for _ in range(trend_blocks)
            ]
        )

        self.seasonality_stack = nn.ModuleList(
            [
                NBeatsBasisBlock(
                    input_len=input_len,
                    pred_len=pred_len,
                    hidden_dim=hidden_dim,
                    n_layers=n_layers,
                    n_theta=2 * seasonality_harmonics + 1,
                    block_type="seasonality",
                )
                for _ in range(seasonality_blocks)
            ]
        )

    def forward(self, x):
        residual = x[:, 0, :]
        forecast = torch.zeros(residual.size(0), self.pred_len, device=x.device)

        for block in self.trend_stack:
            backcast, block_forecast = block(residual)
            residual = residual - backcast
            forecast = forecast + block_forecast

        for block in self.seasonality_stack:
            backcast, block_forecast = block(residual)
            residual = residual - backcast
            forecast = forecast + block_forecast

        return forecast


def make_model(input_len, pred_len, args, device):
    return NBeatsXForecaster(
        input_len=input_len,
        pred_len=pred_len,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        trend_blocks=args.trend_blocks,
        seasonality_blocks=args.seasonality_blocks,
        trend_degree=args.trend_degree,
        seasonality_harmonics=args.seasonality_harmonics,
    ).to(device)


def make_model_config(args, input_len, pred_len):
    return {
        "model_type": "nbeatsx_style",
        "input_len": input_len,
        "pred_len": pred_len,
        "hidden_dim": args.hidden_dim,
        "n_layers": args.n_layers,
        "trend_blocks": args.trend_blocks,
        "seasonality_blocks": args.seasonality_blocks,
        "trend_degree": args.trend_degree,
        "seasonality_harmonics": args.seasonality_harmonics,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="N-BEATSx-style sweep for delta forecasting")
    add_common_args(
        parser,
        default_output_dir="outputs_nbeatsx_sweep",
        default_checkpoint_name="nbeatsx_delta_huber_best.pth",
    )
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--trend-blocks", type=int, default=2)
    parser.add_argument("--seasonality-blocks", type=int, default=2)
    parser.add_argument("--trend-degree", type=int, default=3)
    parser.add_argument("--seasonality-harmonics", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_sweep(args, model_factory=make_model, model_config_factory=make_model_config)
