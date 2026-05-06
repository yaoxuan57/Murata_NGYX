import argparse

import torch
import torch.nn as nn

from forecast_sweep_common import add_common_args, run_sweep


class MambaLikeBlock(nn.Module):
    def __init__(self, d_model, d_inner, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * d_inner)
        self.dw_conv = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=3,
            padding=2,
            groups=d_inner,
        )
        self.out_proj = nn.Linear(d_inner, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        z = self.norm(x)
        uv = self.in_proj(z)
        u, v = uv.chunk(2, dim=-1)

        u = u.transpose(1, 2)
        u = self.dw_conv(u)
        u = u[:, :, : x.size(1)]
        u = u.transpose(1, 2)

        y = torch.silu(u) * torch.sigmoid(v)
        y = self.out_proj(y)
        y = self.dropout(y)
        return residual + y


class MambaFormerForecastDelta(nn.Module):
    def __init__(
        self,
        seq_len,
        input_dim=1,
        pred_len=36,
        d_model=128,
        d_inner=256,
        nhead=8,
        num_mamba_layers=3,
        num_former_layers=2,
        dim_feedforward=256,
        dropout=0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        self.mamba_blocks = nn.ModuleList(
            [
                MambaLikeBlock(d_model=d_model, d_inner=d_inner, dropout=dropout)
                for _ in range(num_mamba_layers)
            ]
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.former = nn.TransformerEncoder(encoder_layer, num_layers=num_former_layers)
        self.norm = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.input_proj(x)
        x = x + self.pos_embedding[:, : x.size(1), :]

        for block in self.mamba_blocks:
            x = block(x)

        x = self.former(x)
        x = self.norm(x)
        x = x[:, -1, :]
        return self.head(x)


def make_model(input_len, pred_len, args, device):
    return MambaFormerForecastDelta(
        seq_len=input_len,
        input_dim=1,
        pred_len=pred_len,
        d_model=args.d_model,
        d_inner=args.d_inner,
        nhead=args.nhead,
        num_mamba_layers=args.num_mamba_layers,
        num_former_layers=args.num_former_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    ).to(device)


def make_model_config(args, input_len, pred_len):
    return {
        "model_type": "mambaformer",
        "input_len": input_len,
        "pred_len": pred_len,
        "d_model": args.d_model,
        "d_inner": args.d_inner,
        "nhead": args.nhead,
        "num_mamba_layers": args.num_mamba_layers,
        "num_former_layers": args.num_former_layers,
        "dim_feedforward": args.dim_feedforward,
        "dropout": args.dropout,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="MambaFormer sweep for delta forecasting")
    add_common_args(
        parser,
        default_output_dir="outputs_mambaformer_sweep",
        default_checkpoint_name="mambaformer_delta_huber_date_split_best.pth",
    )
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--d-inner", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-mamba-layers", type=int, default=3)
    parser.add_argument("--num-former-layers", type=int, default=2)
    parser.add_argument("--dim-feedforward", type=int, default=256)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_sweep(args, model_factory=make_model, model_config_factory=make_model_config)
