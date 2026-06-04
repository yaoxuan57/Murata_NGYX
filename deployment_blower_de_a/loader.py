"""Load training checkpoints (DLinear or transformer)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from config import CHECKPOINT_STEM, DEFAULT_CHECKPOINT
from dlinear_model import make_dlinear
from transformer_model import make_transformer


_CHECKPOINT_CANDIDATES = (
    f"{CHECKPOINT_STEM}_v2.pth",
    f"{CHECKPOINT_STEM}.pth",
    "dlinear_delta_huber_best.pth",
    "model.pth",
)


def resolve_checkpoint(path: str | Path | None = None) -> Path:
    if path is None or (isinstance(path, str) and not str(path).strip()):
        base = DEFAULT_CHECKPOINT
    else:
        base = Path(path)
    if base.is_file():
        return base.resolve()
    if base.is_dir():
        for name in _CHECKPOINT_CANDIDATES:
            candidate = base / name
            if candidate.is_file():
                return candidate.resolve()
        pths = sorted(base.glob("*.pth"))
        if len(pths) == 1:
            return pths[0].resolve()
        raise FileNotFoundError(
            f"No checkpoint in {base}: tried {', '.join(_CHECKPOINT_CANDIDATES)} "
            f"or pass -c path/to/your.pth"
        )
    if base.parent.is_dir():
        for name in _CHECKPOINT_CANDIDATES:
            candidate = base.parent / name
            if candidate.is_file():
                return candidate.resolve()
        pths = sorted(base.parent.glob("*.pth"))
        if len(pths) == 1:
            return pths[0].resolve()
    raise FileNotFoundError(f"Checkpoint not found: {base}")


def load_checkpoint(checkpoint_path: str | Path, device: str):
    ckpt_path = resolve_checkpoint(checkpoint_path)
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    required = {"model_state_dict", "train_mean", "train_std", "input_len", "pred_len", "model_config"}
    missing = required - set(ckpt.keys())
    if missing:
        raise ValueError(f"Checkpoint missing keys: {sorted(missing)}")

    mc = dict(ckpt["model_config"])
    fq = mc.get("forecast_quantiles")
    model_type = str(mc.get("model_type", "transformer")).lower()

    input_len = int(ckpt["input_len"])
    pred_len = int(ckpt["pred_len"])

    if model_type == "dlinear":
        model = make_dlinear(input_len, pred_len, mc, device)
    elif model_type == "transformer":
        model = make_transformer(input_len, pred_len, mc, device)
    else:
        raise ValueError(f"Unsupported model_type={model_type!r} in checkpoint")

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    args = SimpleNamespace(
        model_type=model_type,
        forecast_quantiles=list(fq) if fq else None,
    )
    return model, ckpt, args, ckpt_path
