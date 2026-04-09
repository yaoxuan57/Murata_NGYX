import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from forecast_sweep_common import add_common_args, run_sweep


class DLinearForecaster(nn.Module):
    def __init__(self, input_len, pred_len, kernel_size=25):
        super().__init__()
        self.input_len = input_len
        self.pred_len = pred_len
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

        self.linear_trend = nn.Linear(input_len, pred_len)
        self.linear_seasonal = nn.Linear(input_len, pred_len)

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
        return trend_out + seasonal_out


def make_model(input_len, pred_len, args, device):
    return DLinearForecaster(
        input_len=input_len,
        pred_len=pred_len,
        kernel_size=args.kernel_size,
    ).to(device)


def make_model_config(args, input_len, pred_len):
    return {
        "model_type": "dlinear",
        "input_len": input_len,
        "pred_len": pred_len,
        "kernel_size": args.kernel_size,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="DLinear sweep for delta forecasting")
    add_common_args(
        parser,
        default_output_dir="outputs_dlinear_sweep",
        default_checkpoint_name="dlinear_delta_huber_best.pth",
    )
    parser.add_argument("--kernel-size", type=int, default=25)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_sweep(args, model_factory=make_model, model_config_factory=make_model_config)
