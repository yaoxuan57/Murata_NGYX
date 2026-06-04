"""DLinear delta forecaster (inference)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DLinearForecaster(nn.Module):
    def __init__(
        self,
        input_len,
        pred_len,
        input_dim=1,
        kernel_size=25,
        use_residual_head=True,
        residual_hidden=128,
        residual_dropout=0.1,
        residual_weight=0.25,
        n_quantiles: int = 1,
    ):
        super().__init__()
        self.input_len = input_len
        self.pred_len = pred_len
        self.n_quantiles = int(n_quantiles)
        self.input_dim = input_dim
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        self.use_residual_head = use_residual_head
        self.residual_weight = residual_weight
        self.channel_mixer = nn.Conv1d(input_dim, 1, kernel_size=1, bias=False)

        out_dim = pred_len * self.n_quantiles if self.n_quantiles > 1 else pred_len
        self.linear_trend = nn.Linear(input_len, out_dim)
        self.linear_seasonal = nn.Linear(input_len, out_dim)

        if self.use_residual_head:
            self.residual_head = nn.Sequential(
                nn.Linear(input_len, residual_hidden),
                nn.GELU(),
                nn.Dropout(residual_dropout),
                nn.Linear(residual_hidden, out_dim),
            )
        else:
            self.residual_head = None

        nn.init.constant_(self.linear_trend.weight, 1.0 / input_len)
        nn.init.constant_(self.linear_trend.bias, 0.0)
        nn.init.constant_(self.linear_seasonal.weight, 0.0)
        nn.init.constant_(self.linear_seasonal.bias, 0.0)
        nn.init.constant_(self.channel_mixer.weight, 1.0 / max(input_dim, 1))

    def moving_average(self, seq):
        pad = self.kernel_size // 2
        x = seq.unsqueeze(1)
        trend = F.avg_pool1d(F.pad(x, (pad, pad), mode="replicate"), self.kernel_size, stride=1)
        return trend.squeeze(1)

    def forward(self, x):
        seq = self.channel_mixer(x).squeeze(1)
        trend = self.moving_average(seq)
        seasonal = seq - trend

        trend_out = self.linear_trend(trend)
        seasonal_out = self.linear_seasonal(seasonal)
        out = trend_out + seasonal_out
        if self.residual_head is not None:
            out = out + self.residual_weight * self.residual_head(seq)
        if self.n_quantiles <= 1:
            return out
        b = out.size(0)
        z = out.view(b, self.n_quantiles, self.pred_len)
        z0 = z[:, 0, :]
        inc = F.softplus(z[:, 1:, :] - z[:, :-1, :])
        return torch.cat([z0.unsqueeze(1), z0.unsqueeze(1) + torch.cumsum(inc, dim=1)], dim=1)


def make_dlinear(input_len, pred_len, mc: dict, device):
    fq = mc.get("forecast_quantiles")
    n_q = len(fq) if fq else int(mc.get("n_quantiles", 1))
    return DLinearForecaster(
        input_len=input_len,
        pred_len=pred_len,
        input_dim=int(mc.get("input_dim", 1)),
        kernel_size=int(mc.get("kernel_size", 25)),
        use_residual_head=bool(mc.get("use_residual_head", True)),
        residual_hidden=int(mc.get("residual_hidden", 128)),
        residual_dropout=float(mc.get("residual_dropout", 0.1)),
        residual_weight=float(mc.get("residual_weight", 0.1)),
        n_quantiles=n_q,
    ).to(device)
