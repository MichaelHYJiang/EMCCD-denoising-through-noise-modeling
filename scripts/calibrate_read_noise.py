#!/usr/bin/env python3
"""Export per-pixel empirical bias mean and read-noise sigma samples."""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import tifffile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--file-list", type=Path, required=True)
    parser.add_argument("--gain", default="300")
    parser.add_argument("--min-exposure", type=float, default=0.001)
    parser.add_argument("--max-exposure", type=float, default=0.2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mapping = json.loads(args.file_list.read_text(encoding="utf-8"))[str(args.gain)]["biased"]
    samples = []
    for exposure, pattern in sorted(mapping.items(), key=lambda item: float(item[0])):
        if not args.min_exposure <= float(exposure) <= args.max_exposure:
            continue
        pattern_path = Path(pattern)
        resolved = pattern if pattern_path.is_absolute() else str(args.raw_root / pattern_path)
        files = sorted(glob.glob(resolved))
        if not files:
            raise FileNotFoundError(f"No frames match {resolved}")
        frames = np.stack([tifffile.imread(path).astype(np.float32) for path in files])
        samples.append(np.stack([frames.mean(axis=0), frames.std(axis=0)]))
    if not samples:
        raise SystemExit("No exposures were selected")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, np.stack(samples))
    print(f"saved {len(samples)} exposure samples with shape {samples[0].shape} to {args.output}")


if __name__ == "__main__":
    main()
