"""Tiled inference shared by command-line tools."""

from __future__ import annotations

import numpy as np
import torch


def denoise(model, image: np.ndarray, tile: int, overlap: int, device: str) -> np.ndarray:
    if tile % 128:
        raise ValueError("tile-size must be divisible by 128")
    if not 0 <= overlap < tile:
        raise ValueError("overlap must satisfy 0 <= overlap < tile-size")
    height, width = image.shape
    padded = np.pad(image, ((0, max(0, tile - height)), (0, max(0, tile - width))), mode="reflect")
    output = np.zeros_like(padded, dtype=np.float32)
    count = np.zeros_like(padded, dtype=np.float32)
    rows = list(range(0, max(1, padded.shape[0] - tile + 1), tile - overlap))
    cols = list(range(0, max(1, padded.shape[1] - tile + 1), tile - overlap))
    rows.append(padded.shape[0] - tile)
    cols.append(padded.shape[1] - tile)
    with torch.inference_mode():
        for row in sorted(set(rows)):
            for col in sorted(set(cols)):
                patch = torch.from_numpy(padded[row : row + tile, col : col + tile][None, None]).to(device)
                restored = model(patch).clamp_(0, 1).cpu().numpy()[0, 0]
                output[row : row + tile, col : col + tile] += restored
                count[row : row + tile, col : col + tile] += 1
    return (output / count)[:height, :width]
