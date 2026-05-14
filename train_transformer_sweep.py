import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        n_quantiles: int = 1,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.n_quantiles = int(n_quantiles)
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
        out_dim = pred_len * self.n_quantiles if self.n_quantiles > 1 else pred_len
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, out_dim),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.input_proj(x)
        x = x + self.pos_embedding[:, : x.size(1), :]
        x = self.encoder(x)
        x = self.norm(x)
        x = x[:, -1, :]
        raw = self.head(x)
        if self.n_quantiles <= 1:
            return raw
        b = raw.size(0)
        z = raw.view(b, self.n_quantiles, self.pred_len)
        z0 = z[:, 0, :]
        inc = F.softplus(z[:, 1:, :] - z[:, :-1, :])
        return torch.cat([z0.unsqueeze(1), z0.unsqueeze(1) + torch.cumsum(inc, dim=1)], dim=1)


def make_model(input_len, pred_len, args, device):
    n_q = len(args.forecast_quantiles) if getattr(args, "forecast_quantiles", None) else 1
    return TransformerForecastDelta(
        seq_len=input_len,
        input_dim=getattr(args, "input_dim", 1),
        pred_len=pred_len,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        n_quantiles=n_q,
    ).to(device)


def make_model_config(args, input_len, pred_len):
    fq = getattr(args, "forecast_quantiles", None)
    n_q = len(fq) if fq else 1
    return {
        "model_type": "transformer",
        "input_len": input_len,
        "pred_len": pred_len,
        "input_dim": getattr(args, "input_dim", 1),
        "d_model": args.d_model,
        "nhead": args.nhead,
        "num_layers": args.num_layers,
        "dim_feedforward": args.dim_feedforward,
        "dropout": args.dropout,
        "n_quantiles": n_q,
        "forecast_quantiles": list(fq) if fq else None,
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
