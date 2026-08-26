"""Calibrated noise synthesis used to train the paper model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Calibration:
    """Runtime calibration parameters exported by the camera experiments."""

    gain: float
    read_noise_samples: np.ndarray

    @classmethod
    def load(cls, directory: str | Path) -> "Calibration":
        directory = Path(directory)
        params = np.load(directory / "params.npy")
        samples = np.load(directory / "S0-sigma-between-0.001s-0.2s.npy")
        if params.shape != (9,):
            raise ValueError(f"Expected params.npy shape (9,), got {params.shape}")
        if samples.ndim not in (2, 4) or samples.shape[1] != 2:
            raise ValueError(
                f"Expected read-noise samples with shape (N, 2) or (N, 2, H, W), got {samples.shape}"
            )
        if not np.isfinite(params).all() or not np.isfinite(samples).all():
            raise ValueError("Calibration arrays contain non-finite values")
        return cls(gain=float(params[0]), read_noise_samples=samples)


class EMCCDNoiseModel:
    """Paper noise model: photon, multiplication, and empirical read noise."""

    def __init__(self, calibration: Calibration, min_ratio: float = 1.0, max_ratio: float = 20.0):
        if calibration.gain <= 0:
            raise ValueError("Calibration gain must be positive")
        if not 0 < min_ratio <= max_ratio:
            raise ValueError("Require 0 < min_ratio <= max_ratio")
        self.calibration = calibration
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def synthesize_shot_noise(
        self,
        clean_adu: np.ndarray,
        *,
        ratio: float | None = None,
        seed: int | None = None,
    ) -> tuple[np.ndarray, float]:
        """Return photon/multiplication noise in normalized [0, 1] ADU."""
        clean = np.asarray(clean_adu, dtype=np.float64)
        if clean.ndim != 2:
            raise ValueError(f"Expected a 2-D grayscale frame, got {clean.shape}")
        rng = np.random if seed is None else np.random.RandomState(seed)
        ratio = float(rng.uniform(self.min_ratio, self.max_ratio) if ratio is None else ratio)
        if not self.min_ratio <= ratio <= self.max_ratio:
            raise ValueError(f"ratio must be in [{self.min_ratio}, {self.max_ratio}]")

        low_exposure = np.clip(clean, 0, None) / ratio
        multiplication = float(rng.uniform(1.3, np.sqrt(2.0)))
        effective_gain = self.calibration.gain / multiplication**2
        photons = low_exposure / effective_gain
        shot = rng.poisson(photons).astype(np.float64) - photons
        noisy = effective_gain * (photons + multiplication * shot)
        return (np.clip(noisy, 0.0, 65535.0) / 65535.0).astype(np.float32), ratio

    def sample_read_noise(self, shape: tuple[int, int], *, seed: int | None = None) -> np.ndarray:
        """Sample the final v5 zero-mean empirical read noise in normalized ADU."""
        rng = np.random if seed is None else np.random.RandomState(seed)
        _, sigma = self.calibration.read_noise_samples[
            rng.randint(len(self.calibration.read_noise_samples))
        ]
        if np.ndim(sigma) == 2 and sigma.shape != shape:
            if sigma.shape[0] < shape[0] or sigma.shape[1] < shape[1]:
                raise ValueError(
                    f"Read-noise calibration shape {sigma.shape} is smaller than frame {shape}"
                )
            row = int(rng.randint(0, sigma.shape[0] - shape[0] + 1))
            col = int(rng.randint(0, sigma.shape[1] - shape[1] + 1))
            sigma = sigma[row : row + shape[0], col : col + shape[1]]
        return (rng.normal(0.0, sigma, size=shape) / 65535.0).astype(np.float32)

    def add_read_noise(self, normalized: np.ndarray, *, seed: int | None = None, quantize: bool = True) -> np.ndarray:
        """Add and optionally quantize empirical read noise."""
        noisy = np.asarray(normalized, dtype=np.float32)
        noisy = noisy + self.sample_read_noise(noisy.shape, seed=seed)
        noisy = np.clip(noisy, 0.0, 1.0)
        if quantize:
            noisy = (noisy * 65535.0).astype(np.uint16).astype(np.float64) / 65535.0
        return noisy.astype(np.float32)

    def synthesize(
        self,
        clean_adu: np.ndarray,
        *,
        ratio: float | None = None,
        seed: int | None = None,
        add_read_noise: bool = True,
        quantize: bool = True,
    ) -> tuple[np.ndarray, float]:
        """Return the complete synthetic low-exposure frame."""
        noisy, ratio = self.synthesize_shot_noise(clean_adu, ratio=ratio, seed=seed)
        if add_read_noise:
            noisy = self.add_read_noise(noisy, seed=None if seed is None else seed + 1, quantize=quantize)
        elif quantize:
            noisy = (noisy * 65535.0).astype(np.uint16).astype(np.float32) / 65535.0
        return noisy, ratio
