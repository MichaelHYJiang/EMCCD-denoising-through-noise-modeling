#!/usr/bin/env python3
"""Estimate the EMCCD conversion gain from flat-field mean/variance data."""

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import linregress
import tifffile


def exposure(path: Path) -> str:
    return path.name.split("-")[3].split("_")[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--gain", type=int, default=300)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = sorted(args.raw_root.glob(f"20231010*/*-{args.gain:03d}-*.tif"))
    if not files:
        raise SystemExit("No matching flat-field TIFFs")
    means, variances = [], []
    for path in files:
        image = tifffile.imread(path)
        height, width = image.shape[:2]
        row, col = (height - args.crop_size) // 2, (width - args.crop_size) // 2
        crop = image[row : row + args.crop_size, col : col + args.crop_size]
        means.append(float(np.median(crop)))
        variances.append(float(np.var(crop)))
    fit = linregress(means, variances)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, fit.slope)
    print(f"gain={fit.slope:.8g} intercept={fit.intercept:.8g} r_squared={fit.rvalue**2:.6f}")


if __name__ == "__main__":
    main()
