# Data and Checkpoints

The complete release is divided into independently downloadable archives:

| Group | Approximate source size | Extracted path | Purpose |
|---|---:|---|---|
| Runtime calibration | 101 MB | `data/calibration/runtime/` | Noise synthesis |
| Raw Andor calibration | 209 GB | `data/raw_calibration/` | Rebuild calibration parameters |
| Clean training set | 5.3 GB | `data/training/clean/` | Paper-model training |
| Canonical benchmark | 6.4 GB | `data/benchmark/` | Training-validation and final-test input/GT variants |
| Cell fine-tuning set | 11 GB | `data/cell_finetune/` | Microscopy adaptation |
| Checkpoints | 1.2 GB | `checkpoints/` | Paper and fine-tuned models |

`assets/manifest.json` is the machine-readable source of filenames, byte sizes,
SHA-256 digests, download URLs, and licenses. At the time of this source
release, Google Drive URLs for the large archives have not yet been assigned;
the downloader fails explicitly rather than fetching an unverifiable file.

The original training-validation configuration uses `preprocessed_input/` with
`new_FPN_removed_GT/`. The final reported benchmark uses
`preprocessed_input_20240513/` with the averaged `gt/` references. Both are
retained because substituting one pair for the other changes the reported
metric.

After downloading, verify any group without network access:

```bash
python scripts/download_assets.py --verify-only runtime checkpoints
```

The data are released under [CC BY 4.0](../DATA_LICENSE). Cite the ICLR 2025
paper when using the calibration frames, training data, or benchmark. The two
small example sets are illustrative subsets under the same license.
