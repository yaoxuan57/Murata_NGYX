import argparse

import torch
import torch.nn as nn

from forecast_sweep_common import add_common_args, run_sweep


class CausalConvBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation, dropout):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.pad = pad
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def _causal(self, conv, x):
        x = nn.functional.pad(x, (self.pad, 0))
        return conv(x)

    def forward(self, x):
        h = self._causal(self.conv1, x)
        h = self.norm1(h)
        h = self.act(h)
        h = self.drop(h)

        h = self._causal(self.conv2, h)
        h = self.norm2(h)
        h = self.act(h)
        h = self.drop(h)
        return x + h


class TCNForecaster(nn.Module):
    def __init__(self, pred_len, channels=128, levels=5, kernel_size=3, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Conv1d(1, channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                CausalConvBlock(channels, kernel_size, dilation=2 ** i, dropout=dropout)
                for i in range(levels)
            ]
        )
        self.head = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels, pred_len),
        )

    def forward(self, x):
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        summary = h[:, :, -1]
        return self.head(summary)


def make_model(input_len, pred_len, args, device):
    _ = input_len
    return TCNForecaster(
        pred_len=pred_len,
        channels=args.channels,
        levels=args.levels,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
    ).to(device)


def make_model_config(args, input_len, pred_len):
    return {
        "model_type": "tcn",
        "input_len": input_len,
        "pred_len": pred_len,
        "channels": args.channels,
        "levels": args.levels,
        "kernel_size": args.kernel_size,
        "dropout": args.dropout,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="TCN sweep for delta forecasting")
    add_common_args(
        parser,
        default_output_dir="outputs_tcn_sweep",
        default_checkpoint_name="tcn_delta_huber_best.pth",
    )
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--levels", type=int, default=5)
    parser.add_argument("--kernel-size", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_sweep(args, model_factory=make_model, model_config_factory=make_model_config)
