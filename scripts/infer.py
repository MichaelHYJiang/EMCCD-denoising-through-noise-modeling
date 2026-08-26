#!/usr/bin/env python3
"""Denoise one TIFF or a directory of TIFFs with a released checkpoint."""

import argparse
from pathlib import Path

import torch

from emccd_denoising.inference import denoise
from emccd_denoising.io import read_tiff, write_tiff
from emccd_denoising.model import build_uformer, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--black-level", type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()
    paths = sorted([*args.input.glob("*.tif"), *args.input.glob("*.tiff")]) if args.input.is_dir() else [args.input]
    if not paths:
        raise SystemExit(f"No TIFF files found at {args.input}")
    args.output.mkdir(parents=True, exist_ok=True)
    model = build_uformer(args.tile_size).to(args.device)
    epoch = load_checkpoint(model, args.weights, args.device)
    model.eval()
    print(f"Loaded {args.weights} (epoch {epoch})")
    for path in paths:
        image = read_tiff(path, args.black_level) / 65535.0
        write_tiff(args.output / path.name, denoise(model, image, args.tile_size, args.overlap, args.device))


if __name__ == "__main__":
    main()
