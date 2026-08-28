# Data and Checkpoints

## Download and layout

Run commands from the repository root. The downloader streams each asset from
Google Drive, verifies its byte size and SHA-256 digest, and only then extracts
archives when `--extract` is supplied.

The same files can be browsed manually in the
[public Google Drive release folder](https://drive.google.com/open?id=1-4PHaNjv2FRq4R3oPNHlSjekrsBIwULc),
but the script is recommended because it verifies integrity and creates the
configured directory layout.

```bash
# Published benchmark and inference checkpoint (about 6.7 GB downloaded)
python scripts/download_assets.py runtime checkpoints benchmark --extract

# Paper training inputs (about 9.6 GB downloaded)
python scripts/download_assets.py runtime training benchmark --extract

# Cell fine-tuning inputs and starting checkpoint (about 14.3 GB downloaded)
python scripts/download_assets.py runtime checkpoints benchmark fine-tuning --extract
```

Archives remain under `data/archives/` after extraction so they can be verified
or moved to archival storage. Extracted data and downloaded checkpoints are
ignored by Git.

The release groups produce this layout:

| Group | Approximate source size | Extracted path | Purpose |
|---|---:|---|---|
| Runtime calibration | 101 MB | `data/calibration/runtime/` | Noise synthesis |
| Raw Andor calibration | 209 GB | `data/raw_calibration/` | Audit/rebuild calibration parameters |
| Clean training set | 5.3 GB | `data/training/clean/` | Paper-model training |
| Canonical benchmark | 6.4 GB | `data/benchmark/` | Training-validation and final-test input/GT variants |
| Cell fine-tuning set | 11 GB | `data/cell_finetune/` | Microscopy adaptation |
| Checkpoints | 1.2 GB | `checkpoints/` | Paper and fine-tuned models |

`scripts/check_setup.py` additionally checks the expected 231 clean training
frames, 224 validation pairs, 224 canonical benchmark pairs, and 24 cell
fine-tuning frames before a run begins.

`assets/manifest.json` is the machine-readable source of filenames, byte sizes,
SHA-256 digests, download URLs, and licenses. Raw calibration sessions are
packaged as multiple archives in the `raw-calibration` group. Extract every
part into the same destination to reconstruct the original tree. The downloader
does this automatically, but the full raw group requires space for both the
144.6 GB of downloaded archives and approximately 209 GB of extracted frames
(about 354 GB total while both copies are retained):

```bash
python scripts/download_assets.py raw-calibration --extract
```

Interrupted downloads leave a `.part` file and can be rerun safely; the final
filename is assigned atomically only after transfer completion. Existing final
files are always verified before use.

## Which checkpoint to use

- `paper_model_best.pth` (epoch 537) reproduces the canonical paper benchmark
  and initializes cell fine-tuning.
- `cell_finetuned_model_best.pth` (epoch 1035) is the adapted model for cell
  microscopy inputs. It must not be substituted in the canonical benchmark
  when comparing against the reported paper metrics.

Both checkpoints contain model and optimizer state. `scripts/infer.py` loads
model weights; `scripts/train.py` also restores optimizer state when resuming.

## Benchmark variants

The original training-validation configuration uses `preprocessed_input/` with
`new_FPN_removed_GT/`. The final reported benchmark uses
`preprocessed_input_20240513/` with the averaged `gt/` references. Both are
retained because substituting one pair for the other changes the reported
metric.

After downloading, verify any group without network access:

```bash
python scripts/download_assets.py runtime checkpoints --verify-only
```

For a manual extraction, each archive has one wrapper directory. Strip its
first component and use the manifest's `extract_to` destination. Automatic
extraction is recommended because it enforces the paths expected by the YAML
configs.

The data are released under [CC BY 4.0](../DATA_LICENSE). Cite the ICLR 2025
paper when using the calibration frames, training data, or benchmark. The two
small example sets are illustrative subsets under the same license.
