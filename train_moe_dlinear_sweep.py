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


class MoEDLinearForecaster(nn.Module):
    def __init__(
        self,
        input_len,
        pred_len,
        num_experts=4,
        gate_hidden=64,
        gate_dropout=0.1,
        gate_temperature=1.0,
        kernel_size=25,
        use_residual_head=True,
        residual_hidden=128,
        residual_dropout=0.1,
        residual_weight=0.25,
    ):
        super().__init__()
        self.num_experts = max(1, int(num_experts))
        self.gate_temperature = max(float(gate_temperature), 1e-6)

        self.experts = nn.ModuleList(
            [
                DLinearForecaster(
                    input_len=input_len,
                    pred_len=pred_len,
                    kernel_size=kernel_size,
                    use_residual_head=use_residual_head,
                    residual_hidden=residual_hidden,
                    residual_dropout=residual_dropout,
                    residual_weight=residual_weight,
                )
                for _ in range(self.num_experts)
            ]
        )
        self.gate = nn.Sequential(
            nn.Linear(input_len, gate_hidden),
            nn.GELU(),
            nn.Dropout(gate_dropout),
            nn.Linear(gate_hidden, self.num_experts),
        )

    def forward(self, x):
        seq = x[:, 0, :]
        logits = self.gate(seq) / self.gate_temperature
        weights = torch.softmax(logits, dim=-1)

        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        return torch.sum(expert_outputs * weights.unsqueeze(-1), dim=1)


def make_model(input_len, pred_len, args, device):
    return MoEDLinearForecaster(
        input_len=input_len,
        pred_len=pred_len,
        num_experts=args.num_experts,
        gate_hidden=args.moe_gate_hidden,
        gate_dropout=args.moe_gate_dropout,
        gate_temperature=args.moe_gate_temperature,
        kernel_size=args.kernel_size,
        use_residual_head=args.use_residual_head,
        residual_hidden=args.residual_hidden,
        residual_dropout=args.residual_dropout,
        residual_weight=args.residual_weight,
    ).to(device)


def make_model_config(args, input_len, pred_len):
    return {
        "model_type": "moe_dlinear",
        "input_len": input_len,
        "pred_len": pred_len,
        "num_experts": args.num_experts,
        "moe_gate_hidden": args.moe_gate_hidden,
        "moe_gate_dropout": args.moe_gate_dropout,
        "moe_gate_temperature": args.moe_gate_temperature,
        "kernel_size": args.kernel_size,
        "use_residual_head": args.use_residual_head,
        "residual_hidden": args.residual_hidden,
        "residual_dropout": args.residual_dropout,
        "residual_weight": args.residual_weight,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="MoE-DLinear sweep for delta forecasting")
    add_common_args(
        parser,
        default_output_dir="outputs_moe_dlinear_sweep",
        default_checkpoint_name="moe_dlinear_delta_huber_best.pth",
    )
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--moe-gate-hidden", type=int, default=64)
    parser.add_argument("--moe-gate-dropout", type=float, default=0.1)
    parser.add_argument("--moe-gate-temperature", type=float, default=1.0)
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
