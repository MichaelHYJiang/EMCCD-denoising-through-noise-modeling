# Camera Calibration

The training noise model uses two calibrated quantities directly: conversion
gain `K` and an empirical collection of per-pixel read-noise standard
deviations. The runtime package also preserves fixed-pattern arrays used during
model development, although the final v5 training path sets their mean
contribution to zero.

## Flat-field gain audit

The photon-transfer estimate fits the slope of spatial variance against median
signal across the central 128×128 region of EM-gain-300 flat-field frames:

```bash
python scripts/download_assets.py raw-calibration --extract
python scripts/calibrate_gain.py --raw-root data/raw_calibration \
  --gain 300 --crop-size 128 --output work/k-128.npy
```

This command reproduces the archived exploratory
`flat-field-K/k-300-128x128.npy` value, 15.8858947138. It does **not** reproduce
the final `param_ver1/k-128.npy` value, 20.1383119596, stored as element zero of
the released runtime `params.npy`; the final fit's generating script was not
recovered. Therefore this audit output is not a replacement for the complete
nine-element historical parameter vector.

## Empirical read noise

`file_list.json` in the raw archive maps exposure times to bias-frame globs.
For exposures from 1 ms through 200 ms, the calibration computes a per-pixel
mean and standard deviation over repeated frames:

```bash
python scripts/calibrate_read_noise.py --raw-root data/raw_calibration \
  --file-list data/raw_calibration/file_list.json \
  --output work/S0-sigma-between-0.001s-0.2s.npy
```

Exposure samples intentionally retain the insertion order in `file_list.json`,
matching the historical script and released array; sorting exposure keys would
change the array indices used by stochastic training. Float64 mean/std
accumulation is also preserved. For a regenerated 0.001 s sample, the mean is
bitwise identical to released slice 12 and standard deviations agree within
`6.1e-14`; reduction rounding can vary slightly with NumPy versions. Use the
released runtime array, rather than a regenerated one, for metric reproduction.

The released runtime arrays and their expected hashes are listed in
`assets/manifest.json`. Verify and use those released arrays for reproduction;
the audit outputs above are not drop-in replacements. Calibration acquisition
details and intermediate plots remain in the raw archive so the fitted
parameters can be inspected without placing hundreds of gigabytes in Git.

The published training and inference results do not require regenerating these
files. Downloading the `runtime` group installs the exact arrays used by the
verified reproduction:

```bash
python scripts/download_assets.py runtime
```
