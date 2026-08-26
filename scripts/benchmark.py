#!/usr/bin/env python3
"""Run the canonical paired benchmark directly from a released checkpoint."""

import argparse
import json
from pathlib import Path

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch

from emccd_denoising.inference import denoise
from emccd_denoising.io import read_tiff, write_tiff
from emccd_denoising.model import build_uformer, load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=51, help="51 reproduces the historical 10%% overlap")
    parser.add_argument("--lpips", action="store_true", help="Also compute LPIPS-VGG (requires the lpips package)")
    args = parser.parse_args()
    model = build_uformer(args.tile_size).to(args.device).eval()
    epoch = load_checkpoint(model, args.weights, args.device)
    perceptual = None
    if args.lpips:
        import lpips
        perceptual = lpips.LPIPS(net="vgg").to(args.device).eval()
    scores = []
    inputs = sorted([*args.input_dir.glob("*.tif"), *args.input_dir.glob("*.tiff")])
    if not inputs:
        raise SystemExit(f"No TIFF files found in {args.input_dir}")
    with torch.inference_mode():
        for index, input_path in enumerate(inputs, 1):
            gt_path = args.gt_dir / input_path.name
            if not gt_path.exists():
                raise FileNotFoundError(f"Missing ground truth: {gt_path}")
            image = read_tiff(input_path) / 65535.0
            target = read_tiff(gt_path) / 65535.0
            if image.shape != target.shape:
                raise ValueError(f"Shape mismatch for {input_path.name}")
            if image.mean() > 0:
                image = image / image.mean() * target.mean()
            restored = denoise(model, image.astype(np.float32), args.tile_size, args.overlap, args.device)
            item = {
                "psnr": peak_signal_noise_ratio(target, restored, data_range=1.0),
                "ssim": structural_similarity(target, restored, data_range=1.0),
            }
            if perceptual is not None:
                output_tensor = torch.from_numpy(restored.reshape(1, 1, *restored.shape)).to(args.device) * 2 - 1
                target_tensor = torch.from_numpy(target.astype(np.float32).reshape(1, 1, *target.shape)).to(args.device) * 2 - 1
                item["lpips"] = float(perceptual(output_tensor, target_tensor))
            scores.append(item)
            if args.output_dir:
                write_tiff(args.output_dir / input_path.name, restored)
            print(f"[{index}/{len(inputs)}] {input_path.name} PSNR={item['psnr']:.4f}", flush=True)
    summary = {"count": len(scores), "checkpoint_epoch": epoch}
    for metric in ("psnr", "ssim", "lpips"):
        if metric in scores[0]:
            summary[metric] = float(np.mean([item[metric] for item in scores]))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
