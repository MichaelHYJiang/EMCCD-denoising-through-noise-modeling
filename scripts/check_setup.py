#!/usr/bin/env python3
"""Validate that downloaded assets are ready for a reproduction workflow."""

import argparse
import hashlib
from pathlib import Path

import numpy as np


PAPER_SHA256 = "bae2af33916e32c210c142a77fcf9bc6011bf2611bd7dad5ce1e42f66785692b"
CELL_SHA256 = "179ad3bd0e08bd8f219d230f5f4a460ce4def51afc6ffc2d5830f5c693bb01f7"


def tiffs(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing directory: {directory}")
    return sorted([*directory.glob("*.tif"), *directory.glob("*.tiff")])


def require_count(directory: Path, expected: int) -> list[Path]:
    files = tiffs(directory)
    if len(files) != expected:
        raise ValueError(f"Expected {expected} TIFFs in {directory}, found {len(files)}")
    print(f"ok: {directory} ({len(files)} TIFFs)")
    return files


def require_pairs(inputs: Path, targets: Path, expected: int) -> None:
    input_files = require_count(inputs, expected)
    target_names = {path.name for path in require_count(targets, expected)}
    missing = [path.name for path in input_files if path.name not in target_names]
    if missing:
        raise ValueError(f"Missing paired targets in {targets}: {missing[:3]}")


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def require_checkpoint(path: Path, expected: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"Checkpoint checksum mismatch for {path}: {actual}")
    print(f"ok: {path} (SHA-256 verified)")


def require_calibration(directory: Path) -> None:
    params = np.load(directory / "params.npy")
    samples = np.load(directory / "S0-sigma-between-0.001s-0.2s.npy", mmap_mode="r")
    if params.shape != (9,) or samples.ndim not in (2, 4) or samples.shape[1] != 2:
        raise ValueError(f"Invalid runtime calibration shapes: {params.shape}, {samples.shape}")
    print(f"ok: {directory} (gain/read-noise arrays load)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("benchmark", "paper-training", "fine-tuning"))
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.root

    if args.mode == "benchmark":
        require_checkpoint(root / "checkpoints/paper_model_best.pth", PAPER_SHA256)
        require_pairs(
            root / "data/benchmark/preprocessed_input_20240513",
            root / "data/benchmark/gt",
            224,
        )
    elif args.mode == "paper-training":
        require_calibration(root / "data/calibration/runtime")
        require_count(root / "data/training/clean", 231)
        require_pairs(
            root / "data/benchmark/preprocessed_input",
            root / "data/benchmark/new_FPN_removed_GT",
            224,
        )
    else:
        require_calibration(root / "data/calibration/runtime")
        require_checkpoint(root / "checkpoints/paper_model_best.pth", PAPER_SHA256)
        require_count(root / "data/cell_finetune", 24)
        require_pairs(
            root / "data/benchmark/preprocessed_input",
            root / "data/benchmark/new_FPN_removed_GT",
            224,
        )
    print(f"ready: {args.mode}")


if __name__ == "__main__":
    main()
