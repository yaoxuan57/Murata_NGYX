import argparse

import torch
import torch.nn as nn

from forecast_sweep_common import add_common_args, run_sweep


class TransformerForecastDelta(nn.Module):
    def __init__(
        self,
        seq_len,
        input_dim=1,
        pred_len=12,
        d_model=32,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
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
        x = self.encoder(x)
        x = self.norm(x)
        x = x[:, -1, :]
        return self.head(x)


def make_model(input_len, pred_len, args, device):
    return TransformerForecastDelta(
        seq_len=input_len,
        input_dim=1,
        pred_len=pred_len,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    ).to(device)


def make_model_config(args, input_len, pred_len):
    return {
        "model_type": "transformer",
        "input_len": input_len,
        "pred_len": pred_len,
        "d_model": args.d_model,
        "nhead": args.nhead,
        "num_layers": args.num_layers,
        "dim_feedforward": args.dim_feedforward,
        "dropout": args.dropout,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Transformer sweep for delta forecasting")
    add_common_args(
        parser,
        default_output_dir="outputs_transformer_sweep",
        default_checkpoint_name="transformer_delta_huber_date_split_best.pth",
    )
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dim-feedforward", type=int, default=256)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_sweep(args, model_factory=make_model, model_config_factory=make_model_config)
