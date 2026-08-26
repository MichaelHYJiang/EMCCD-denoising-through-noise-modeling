#!/usr/bin/env python3
"""Apply the paper's FPN/blooming correction to paired benchmark TIFFs."""

import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
import tifffile


def exposure_from_name(path: Path) -> float:
    token = path.stem.split("-")[-1].split("_")[0]
    return float(f"{token[0]}.{token[1:]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--fpn-dir", type=Path, required=True, help="Directory of exposure-specific FPN .npy files")
    parser.add_argument("--blooming-exposures", type=Path, required=True)
    parser.add_argument("--blooming-factors", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_input = args.output_dir / "preprocessed_input"
    output_gt = args.output_dir / "new_FPN_removed_GT"
    output_input.mkdir(parents=True, exist_ok=True); output_gt.mkdir(parents=True, exist_ok=True)
    exposures = np.load(args.blooming_exposures)
    factors = np.load(args.blooming_factors)
    fpn_by_exposure = {
        float(path.name.split("-")[1][:-4]): path for path in args.fpn_dir.glob("*.npy")
    }
    for input_path in sorted(args.input_dir.glob("*.tif*")):
        gt_path = args.gt_dir / input_path.name
        exposure = exposure_from_name(input_path)
        if exposure not in fpn_by_exposure or not gt_path.exists():
            raise FileNotFoundError(f"Missing FPN or ground truth for {input_path.name}")
        matches = np.flatnonzero(np.isclose(exposures, exposure))
        blooming = float(factors[matches[0]]) if len(matches) else 0.0
        fraction = blooming / (1.0 + blooming)
        fpn = np.load(fpn_by_exposure[exposure])
        input_raw = tifffile.imread(input_path).astype(np.float64)
        gt_raw = tifffile.imread(gt_path).astype(np.float64)
        corrected_fpn = fpn * (1.0 - fraction)
        input_image = (input_raw - input_raw.mean(axis=0, keepdims=True) * fraction - corrected_fpn) / 65535.0
        gt_image = (gt_raw - corrected_fpn) / 65535.0
        objective = lambda values: np.mean((input_image * values[0] + values[1] - gt_image) ** 2)
        ratio, bias = minimize(objective, [1.0, 0.0]).x
        input_image = np.clip(input_image * ratio + bias, -1, 1)
        tifffile.imwrite(output_input / input_path.name, (input_image * 65535).astype(np.float32))
        tifffile.imwrite(output_gt / input_path.name, (np.clip(gt_image, -1, 1) * 65535).astype(np.float32))


if __name__ == "__main__":
    main()
