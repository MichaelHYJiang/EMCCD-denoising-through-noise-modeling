"""TIFF I/O and normalization helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile


def read_tiff(path: str | Path, black_level: float = 0.0) -> np.ndarray:
    image = tifffile.imread(path)
    if image.ndim > 2:
        image = image[..., 0]
    return np.clip(image.astype(np.float32) - black_level, 0, 65535)


def write_tiff(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = np.rint(np.clip(image, 0, 1) * 65535).astype(np.uint16)
    tifffile.imwrite(path, encoded)
