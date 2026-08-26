"""Checkpoint-compatible construction of the grayscale Uformer-B model."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import torch

from .uformer import Uformer


def build_uformer(image_size: int = 512) -> Uformer:
    return Uformer(
        img_size=image_size,
        embed_dim=32,
        win_size=8,
        token_projection="linear",
        token_mlp="leff",
        depths=[1, 2, 8, 8, 2, 8, 8, 2, 1],
        modulator=True,
        dd_in=1,
        in_chans=1,
    )


def load_checkpoint(model: torch.nn.Module, path: str | Path, device: str | torch.device = "cpu") -> int | None:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch < 2.0, including the archived uformer environments
        checkpoint = torch.load(path, map_location=device)
    state = checkpoint.get("state_dict", checkpoint)
    state = OrderedDict((key.removeprefix("module."), value) for key, value in state.items())
    model.load_state_dict(state, strict=True)
    return checkpoint.get("epoch") if isinstance(checkpoint, dict) else None
