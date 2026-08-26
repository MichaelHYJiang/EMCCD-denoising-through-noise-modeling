# Camera Calibration

The training noise model uses two calibrated quantities directly: conversion
gain `K` and an empirical collection of per-pixel read-noise standard
deviations. The runtime package also preserves fixed-pattern arrays used during
model development, although the final v5 training path sets their mean
contribution to zero.

## Conversion gain

The photon-transfer estimate fits the slope of spatial variance against median
signal across the central 128×128 region of EM-gain-300 flat-field frames:

```bash
python scripts/calibrate_gain.py --raw-root data/raw_calibration \
  --gain 300 --crop-size 128 --output work/k-128.npy
```

## Empirical read noise

`file_list.json` in the raw archive maps exposure times to bias-frame globs.
For exposures from 1 ms through 200 ms, the calibration computes a per-pixel
mean and standard deviation over repeated frames:

```bash
python scripts/calibrate_read_noise.py --raw-root data/raw_calibration \
  --file-list data/raw_calibration/file_list.json \
  --output work/S0-sigma-between-0.001s-0.2s.npy
```

The released runtime arrays and their expected hashes are listed in
`assets/manifest.json`. Always compare regenerated parameters against those
hashes before training. Calibration acquisition details and intermediate plots
remain in the raw archive so that the fitted parameters can be audited without
placing hundreds of gigabytes in Git.
