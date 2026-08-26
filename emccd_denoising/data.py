"""Datasets for synthetic training and paired validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import random
import torch
from torch.utils.data import Dataset

from .io import read_tiff
from .noise import EMCCDNoiseModel


def _tiffs(directory: str | Path) -> list[Path]:
    paths = sorted([*Path(directory).glob("*.tif"), *Path(directory).glob("*.tiff")])
    if not paths:
        raise FileNotFoundError(f"No TIFF files found in {directory}")
    return paths


def _tensor(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(image[None].astype(np.float32)))


class SyntheticDataset(Dataset):
    def __init__(self, directory: str | Path, noise: EMCCDNoiseModel, patch_size: int, seed: int = 1234, full_frame_noise: bool = True):
        self.paths = _tiffs(directory)
        self.noise = noise
        self.patch_size = patch_size
        self.seed = seed
        self.full_frame_noise = full_frame_noise

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        clean_adu = read_tiff(self.paths[index])
        height, width = clean_adu.shape
        if min(height, width) < self.patch_size:
            raise ValueError(f"{self.paths[index]} is smaller than patch size {self.patch_size}")
        if self.full_frame_noise:
            noisy, _ = self.noise.synthesize_shot_noise(clean_adu)
        row = 0 if height == self.patch_size else int(np.random.randint(0, height - self.patch_size))
        col = 0 if width == self.patch_size else int(np.random.randint(0, width - self.patch_size))
        clean_adu = clean_adu[row : row + self.patch_size, col : col + self.patch_size]
        if not self.full_frame_noise:
            noisy, _ = self.noise.synthesize_shot_noise(clean_adu)
        else:
            noisy = noisy[row : row + self.patch_size, col : col + self.patch_size]
        clean = torch.from_numpy(np.float32(clean_adu / 65535.0)[None])
        noisy = torch.from_numpy(np.float32(noisy)[None])
        transform = random.getrandbits(3)
        rotations = transform % 4
        clean = torch.rot90(clean, rotations, dims=[-1, -2])
        noisy = torch.rot90(noisy, rotations, dims=[-1, -2])
        if transform >= 4:
            clean = clean.flip(-2)
            noisy = noisy.flip(-2)
        read_noise = torch.from_numpy(self.noise.sample_read_noise((self.patch_size, self.patch_size))[None])
        noisy = (torch.clamp(noisy + read_noise, 0, 1) * 65535).int().float()
        if noisy.mean() > 0:
            noisy = noisy / noisy.mean() * clean.mean()
        return clean, noisy


class PairedDataset(Dataset):
    def __init__(
        self, input_dir: str | Path, ground_truth_dir: str | Path,
        patch_size: int | None = None, random_crop: bool = False,
    ):
        self.inputs = _tiffs(input_dir)
        gt_by_name = {path.name: path for path in _tiffs(ground_truth_dir)}
        missing = [path.name for path in self.inputs if path.name not in gt_by_name]
        if missing:
            raise ValueError(f"Missing ground truth for {missing[:3]}")
        self.targets = [gt_by_name[path.name] for path in self.inputs]
        self.patch_size = patch_size
        self.random_crop = random_crop

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int):
        noisy = read_tiff(self.inputs[index]) / 65535.0
        clean = read_tiff(self.targets[index]) / 65535.0
        if self.patch_size is not None:
            height, width = clean.shape
            if min(height, width) < self.patch_size:
                raise ValueError(f"{self.inputs[index]} is smaller than patch size {self.patch_size}")
            if self.random_crop:
                row = 0 if height == self.patch_size else int(np.random.randint(0, height - self.patch_size))
                col = 0 if width == self.patch_size else int(np.random.randint(0, width - self.patch_size))
            else:
                row = (height - self.patch_size) // 2
                col = (width - self.patch_size) // 2
            clean = clean[row : row + self.patch_size, col : col + self.patch_size]
            noisy = noisy[row : row + self.patch_size, col : col + self.patch_size]
        if noisy.mean() > 0:
            noisy = noisy / noisy.mean() * clean.mean()
        return _tensor(clean), _tensor(np.clip(noisy, 0, 1)), self.inputs[index].name
