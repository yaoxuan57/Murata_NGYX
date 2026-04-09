import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from forecast_sweep_common import add_common_args, run_sweep


class PatchTSTDecompForecaster(nn.Module):
    def __init__(
        self,
        input_len,
        pred_len,
        patch_len=16,
        stride=8,
        d_model=128,
        nhead=8,
        num_layers=3,
        ff_dim=256,
        dropout=0.1,
        trend_kernel=25,
        max_patches=256,
    ):
        super().__init__()
        self.input_len = input_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.trend_kernel = trend_kernel if trend_kernel % 2 == 1 else trend_kernel + 1

        self.patch_proj = nn.Linear(patch_len, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, max_patches, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        self.seasonal_head = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, pred_len),
        )
        self.trend_head = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, pred_len),
        )

    def moving_average(self, seq):
        x = seq.unsqueeze(1)
        pad = self.trend_kernel // 2
        trend = F.avg_pool1d(F.pad(x, (pad, pad), mode="replicate"), self.trend_kernel, stride=1)
        return trend.squeeze(1)

    def encode_component(self, component):
        patches = component.unfold(dimension=1, size=self.patch_len, step=self.stride)
        if patches.size(1) == 0:
            pad = self.patch_len - component.size(1)
            component = F.pad(component, (0, max(pad, 0)))
            patches = component.unfold(dimension=1, size=self.patch_len, step=max(1, self.stride))

        tokens = self.patch_proj(patches)
        n_tokens = tokens.size(1)
        tokens = tokens + self.pos_emb[:, :n_tokens, :]
        encoded = self.encoder(tokens)
        encoded = self.norm(encoded)
        pooled = encoded.mean(dim=1)
        return pooled

    def forward(self, x):
        seq = x[:, 0, :]
        trend = self.moving_average(seq)
        seasonal = seq - trend

        trend_feat = self.encode_component(trend)
        seasonal_feat = self.encode_component(seasonal)

        trend_forecast = self.trend_head(trend_feat)
        seasonal_forecast = self.seasonal_head(seasonal_feat)
        return trend_forecast + seasonal_forecast


def make_model(input_len, pred_len, args, device):
    return PatchTSTDecompForecaster(
        input_len=input_len,
        pred_len=pred_len,
        patch_len=args.patch_len,
        stride=args.patch_stride,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        trend_kernel=args.trend_kernel,
        max_patches=args.max_patches,
    ).to(device)


def make_model_config(args, input_len, pred_len):
    return {
        "model_type": "patchtst_decomposition",
        "input_len": input_len,
        "pred_len": pred_len,
        "patch_len": args.patch_len,
        "patch_stride": args.patch_stride,
        "d_model": args.d_model,
        "nhead": args.nhead,
        "num_layers": args.num_layers,
        "ff_dim": args.ff_dim,
        "dropout": args.dropout,
        "trend_kernel": args.trend_kernel,
        "max_patches": args.max_patches,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="PatchTST+decomposition sweep for delta forecasting")
    add_common_args(
        parser,
        default_output_dir="outputs_patchtst_decomp_sweep",
        default_checkpoint_name="patchtst_decomp_delta_huber_best.pth",
    )
    parser.add_argument("--patch-len", type=int, default=16)
    parser.add_argument("--patch-stride", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--ff-dim", type=int, default=256)
    parser.add_argument("--trend-kernel", type=int, default=25)
    parser.add_argument("--max-patches", type=int, default=256)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_sweep(args, model_factory=make_model, model_config_factory=make_model_config)
