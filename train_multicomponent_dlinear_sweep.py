import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from forecast_sweep_common import add_common_args, run_sweep


class MultiComponentDLinearForecaster(nn.Module):
    def __init__(
        self,
        input_len,
        pred_len,
        k_slow=97,
        k_mid=33,
        k_fast=9,
        use_residual_head=True,
        residual_hidden=128,
        residual_dropout=0.1,
        residual_weight=0.20,
    ):
        super().__init__()
        self.input_len = input_len
        self.pred_len = pred_len

        self.k_slow = k_slow if k_slow % 2 == 1 else k_slow + 1
        self.k_mid = k_mid if k_mid % 2 == 1 else k_mid + 1
        self.k_fast = k_fast if k_fast % 2 == 1 else k_fast + 1

        self.use_residual_head = use_residual_head
        self.residual_weight = residual_weight

        self.linear_slow = nn.Linear(input_len, pred_len)
        self.linear_mid = nn.Linear(input_len, pred_len)
        self.linear_fast = nn.Linear(input_len, pred_len)
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

        # Start from stable averaging behavior for trend branches and neutral seasonal branch.
        nn.init.constant_(self.linear_slow.weight, 1.0 / input_len)
        nn.init.constant_(self.linear_slow.bias, 0.0)
        nn.init.constant_(self.linear_mid.weight, 1.0 / input_len)
        nn.init.constant_(self.linear_mid.bias, 0.0)
        nn.init.constant_(self.linear_fast.weight, 1.0 / input_len)
        nn.init.constant_(self.linear_fast.bias, 0.0)
        nn.init.constant_(self.linear_seasonal.weight, 0.0)
        nn.init.constant_(self.linear_seasonal.bias, 0.0)

    def moving_average(self, seq, kernel_size):
        pad = kernel_size // 2
        x = seq.unsqueeze(1)
        smoothed = F.avg_pool1d(F.pad(x, (pad, pad), mode="replicate"), kernel_size, stride=1)
        return smoothed.squeeze(1)

    def decompose(self, seq):
        # Hierarchical decomposition: slow trend -> mid trend -> fast trend -> seasonal remainder.
        slow = self.moving_average(seq, self.k_slow)
        rem1 = seq - slow

        mid = self.moving_average(rem1, self.k_mid)
        rem2 = rem1 - mid

        fast = self.moving_average(rem2, self.k_fast)
        seasonal = rem2 - fast

        return slow, mid, fast, seasonal

    def forward(self, x):
        seq = x[:, 0, :]
        slow, mid, fast, seasonal = self.decompose(seq)

        out = (
            self.linear_slow(slow)
            + self.linear_mid(mid)
            + self.linear_fast(fast)
            + self.linear_seasonal(seasonal)
        )

        if self.residual_head is not None:
            out = out + self.residual_weight * self.residual_head(seq)

        return out


def make_model(input_len, pred_len, args, device):
    return MultiComponentDLinearForecaster(
        input_len=input_len,
        pred_len=pred_len,
        k_slow=args.k_slow,
        k_mid=args.k_mid,
        k_fast=args.k_fast,
        use_residual_head=args.use_residual_head,
        residual_hidden=args.residual_hidden,
        residual_dropout=args.residual_dropout,
        residual_weight=args.residual_weight,
    ).to(device)


def make_model_config(args, input_len, pred_len):
    return {
        "model_type": "multicomponent_dlinear",
        "input_len": input_len,
        "pred_len": pred_len,
        "k_slow": args.k_slow,
        "k_mid": args.k_mid,
        "k_fast": args.k_fast,
        "use_residual_head": args.use_residual_head,
        "residual_hidden": args.residual_hidden,
        "residual_dropout": args.residual_dropout,
        "residual_weight": args.residual_weight,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-component DLinear sweep for delta forecasting")
    add_common_args(
        parser,
        default_output_dir="outputs_multicomponent_dlinear_sweep",
        default_checkpoint_name="multicomponent_dlinear_delta_huber_best.pth",
    )

    # Apples-to-apples default with your latest DLinear run.
    parser.set_defaults(input_lens=[576], pred_lens=[36])

    parser.add_argument("--k-slow", type=int, default=97)
    parser.add_argument("--k-mid", type=int, default=33)
    parser.add_argument("--k-fast", type=int, default=9)

    parser.add_argument("--use-residual-head", dest="use_residual_head", action="store_true")
    parser.add_argument("--no-residual-head", dest="use_residual_head", action="store_false")
    parser.set_defaults(use_residual_head=True)
    parser.add_argument("--residual-hidden", type=int, default=128)
    parser.add_argument("--residual-dropout", type=float, default=0.1)
    parser.add_argument("--residual-weight", type=float, default=0.20)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_sweep(args, model_factory=make_model, model_config_factory=make_model_config)
