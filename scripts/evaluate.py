#!/usr/bin/env python3
"""Evaluate restored TIFFs against filename-matched ground truth."""

import argparse
import json
from pathlib import Path

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from emccd_denoising.io import read_tiff


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--mean-match", action="store_true", help="Match output mean to ground truth before scoring")
    args = parser.parse_args()
    scores = []
    for result_path in sorted([*args.result_dir.glob("*.tif"), *args.result_dir.glob("*.tiff")]):
        gt_path = args.gt_dir / result_path.name
        if not gt_path.exists():
            raise FileNotFoundError(f"Missing ground truth: {gt_path}")
        result = read_tiff(result_path) / 65535.0
        target = read_tiff(gt_path) / 65535.0
        if result.shape != target.shape:
            raise ValueError(f"Shape mismatch for {result_path.name}: {result.shape} != {target.shape}")
        if args.mean_match and result.mean() > 0:
            result = np.clip(result / result.mean() * target.mean(), 0, 1)
        scores.append({
            "file": result_path.name,
            "psnr": peak_signal_noise_ratio(target, result, data_range=1.0),
            "ssim": structural_similarity(target, result, data_range=1.0),
        })
    if not scores:
        raise SystemExit(f"No result TIFFs in {args.result_dir}")
    summary = {
        "count": len(scores),
        "psnr": float(np.mean([item["psnr"] for item in scores])),
        "ssim": float(np.mean([item["ssim"] for item in scores])),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
