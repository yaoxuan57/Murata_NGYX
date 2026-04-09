import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from forecast_sweep_common import add_common_args, run_sweep


class DLinearForecaster(nn.Module):
    def __init__(
        self,
        input_len,
        pred_len,
        kernel_size=25,
        use_residual_head=True,
        residual_hidden=128,
        residual_dropout=0.1,
        residual_weight=0.25,
    ):
        super().__init__()
        self.input_len = input_len
        self.pred_len = pred_len
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        self.use_residual_head = use_residual_head
        self.residual_weight = residual_weight

        self.linear_trend = nn.Linear(input_len, pred_len)
        self.linear_seasonal = nn.Linear(input_len, pred_len)

        if self.use_residual_head:
            self.residual_head = nn.Sequential(
                nn.Linear(input_len, residual_hidden),
                nn.GELU(),
                nn.Dropout(residual_dropout),
                nn.Linear(residual_hidden, pred_len),
            )
        else:
            self.residual_head = None

        nn.init.constant_(self.linear_trend.weight, 1.0 / input_len)
        nn.init.constant_(self.linear_trend.bias, 0.0)
        nn.init.constant_(self.linear_seasonal.weight, 0.0)
        nn.init.constant_(self.linear_seasonal.bias, 0.0)

    def moving_average(self, seq):
        pad = self.kernel_size // 2
        x = seq.unsqueeze(1)
        trend = F.avg_pool1d(F.pad(x, (pad, pad), mode="replicate"), self.kernel_size, stride=1)
        return trend.squeeze(1)

    def forward(self, x):
        seq = x[:, 0, :]
        trend = self.moving_average(seq)
        seasonal = seq - trend

        trend_out = self.linear_trend(trend)
        seasonal_out = self.linear_seasonal(seasonal)
        out = trend_out + seasonal_out
        if self.residual_head is not None:
            out = out + self.residual_weight * self.residual_head(seq)
        return out


def make_model(input_len, pred_len, args, device):
    return DLinearForecaster(
        input_len=input_len,
        pred_len=pred_len,
        kernel_size=args.kernel_size,
        use_residual_head=args.use_residual_head,
        residual_hidden=args.residual_hidden,
        residual_dropout=args.residual_dropout,
        residual_weight=args.residual_weight,
    ).to(device)


def make_model_config(args, input_len, pred_len):
    return {
        "model_type": "dlinear",
        "input_len": input_len,
        "pred_len": pred_len,
        "kernel_size": args.kernel_size,
        "use_residual_head": args.use_residual_head,
        "residual_hidden": args.residual_hidden,
        "residual_dropout": args.residual_dropout,
        "residual_weight": args.residual_weight,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="DLinear sweep for delta forecasting")
    add_common_args(
        parser,
        default_output_dir="outputs_dlinear_sweep",
        default_checkpoint_name="dlinear_delta_huber_best.pth",
    )
    parser.add_argument("--kernel-size", type=int, default=25)
    parser.add_argument("--use-residual-head", dest="use_residual_head", action="store_true")
    parser.add_argument("--no-residual-head", dest="use_residual_head", action="store_false")
    parser.set_defaults(use_residual_head=True)
    parser.add_argument("--residual-hidden", type=int, default=128)
    parser.add_argument("--residual-dropout", type=float, default=0.1)
    parser.add_argument("--residual-weight", type=float, default=0.25)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_sweep(args, model_factory=make_model, model_config_factory=make_model_config)
