# Reproducibility Notes

## End-to-end published-metric reproduction

Run from the repository root. For the closest match to the verified result,
use the recovered environment and GPU 0:

```bash
conda env create -f environment-paper.yml
conda activate emccd-paper
pip install -e . --no-deps
python scripts/download_assets.py checkpoints benchmark --extract
python scripts/check_setup.py benchmark
CUDA_VISIBLE_DEVICES=0 python scripts/benchmark.py \
  --input-dir data/benchmark/preprocessed_input_20240513 \
  --gt-dir data/benchmark/gt \
  --weights checkpoints/paper_model_best.pth \
  --output-dir outputs/canonical_reproduction --lpips
```

The final JSON object should report `count: 224`, `checkpoint_epoch: 537`, and
approximately:

```json
{
  "psnr": 48.59215,
  "ssim": 0.985367,
  "lpips": 0.074232
}
```

The archived paper values are 48.59228 / 0.985369 / 0.074235. Differences in
the last few decimal places are expected across CUDA kernels and package builds.
This is the reproduction path for the published accuracy result; do not use the
cell-fine-tuned checkpoint for this benchmark. LPIPS may download its VGG
feature weights on first use; subsequent runs use the local cache.

## End-to-end training reproduction

Install the same environment, then download the exact runtime calibration,
clean training images, and historical training-validation pairs:

```bash
python scripts/download_assets.py runtime training benchmark --extract
python scripts/check_setup.py paper-training
CUDA_VISIBLE_DEVICES=0,2 python scripts/train.py --config configs/paper.yaml
```

GPU 1 is deliberately excluded. A quick integration run, useful before a
27-hour training job, is:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/paper.yaml \
  --epochs 1 --batch-size 1 --max-train-steps 1 --max-validation-items 2 \
  --workers 0 --output-dir outputs/smoke
```

Do not compare the smoke-run loss or PSNR with paper results because it uses
only one training batch and two validation items. A full first epoch uses all
231 training images and 224 validation pairs. In the recovered two-GPU
environment it produced initial validation PSNR 28.53652, epoch loss sum
1.319591, and epoch-1 validation PSNR 38.60086.

Training writes a resolved config, JSON-lines metric log, latest checkpoint,
and best checkpoint under `outputs/paper/`. The historical run selected epoch
537. Random seeds and historical quirks are preserved, but a fresh stochastic
CUDA training run is not guaranteed to yield bitwise-identical weights.

For the separate cell adaptation experiment:

```bash
python scripts/download_assets.py runtime checkpoints benchmark fine-tuning --extract
python scripts/check_setup.py fine-tuning
CUDA_VISIBLE_DEVICES=0,2 python scripts/train.py --config configs/cell_finetune.yaml
```

This resumes model and optimizer state from the epoch-537 paper checkpoint.

## Recovered experiment

The public code was reconstructed from the paper-era working tree and its
January 2024 logs. The authoritative run used Uformer-B with one input/output
channel, 512×512 patches, batch size 4, AdamW (`2e-4`), a three-epoch warmup,
and 1,000 epochs. Random seeds were set to 1234. The original launcher and log
are retained under `historical/` for provenance; their paths are historical and
are not intended to be executed directly.

The public trainer retains the historical random 512×512 validation crop,
full-frame shot-noise synthesis before cropping, eight-way augmentation,
post-augmentation empirical read noise, AMP loss scaling, warmup/cosine
scheduling, and DataParallel behavior. This maximizes fidelity, but CUDA
kernels and multi-worker random-number scheduling do not guarantee bitwise
identical weights across machines.

The January trainer also left the network in `eval()` mode after its initial
validation, so the first epoch ran without stochastic depth/dropout. It switched
to `train()` only after epoch-1 validation. The paper configuration preserves
this non-obvious behavior through `historical_first_epoch_eval_mode: true`;
turn it off for new experiments that do not require checkpoint compatibility.

## Verified training path

The released trainer was run for one complete epoch on all 231 clean training
TIFFs and all 224 validation pairs in the recovered `uformer1` environment on
GPUs 0 and 2. Generated clean/noisy patches match the archived loader
bit-for-bit for fixed RNG state. The released epoch produced initial PSNR
28.53652, loss sum 1.319591, and post-epoch PSNR 38.60086. An untouched archived
trainer control produced 28.5365, 1.3196, and 38.6009; the original January log
records 28.5368, 1.3196, and 38.6000. Optimizer-state resume into epoch 2,
DataParallel checkpoint loading, and strict single-device reload were also
verified.

The full historical run contains 1,000 epochs with a mean of 97.23 seconds per
epoch (about 27 wall-clock hours on two GPUs); its complete log is retained in
`historical/paper_training_log.txt`.

For a full train-from-clean-data replication while excluding GPU 1:

```bash
CUDA_VISIBLE_DEVICES=0,2 python scripts/train.py --config configs/paper.yaml
```

The archived run establishes that this executed training protocol selected the
epoch-537 checkpoint at 47.4564 dB on its training-time validation preprocessing.
The newly released trainer has not been rerun for all 1,000 epochs, so the exact
epoch-1 replication should not be mistaken for a second full convergence study.

The canonical checkpoint has SHA-256
`bae2af33916e32c210c142a77fcf9bc6011bf2611bd7dad5ce1e42f66785692b`.
It reached 47.4564 dB validation PSNR at epoch 537. Re-evaluation after the May
2024 preprocessing update produced 48.5923 dB PSNR, 0.98537 SSIM, and 0.07423
LPIPS. The cell-fine-tuned checkpoint hash is
`179ad3bd0e08bd8f219d230f5f4a460ce4def51afc6ffc2d5830f5c693bb01f7`
and records epoch 1035. Both checkpoints load strictly into the released
50,879,216-parameter architecture.

## Verified release benchmark

On 2026-08-27, the released `scripts/benchmark.py` was run from the canonical
checkpoint over all 224 `preprocessed_input_20240513` images against the
averaged `gt` references. Using GPU 0 in the recovered `uformer1` environment,
it produced 48.5921525 dB PSNR, 0.98536731 SSIM, and 0.07423152 LPIPS-VGG.
These match the recorded in-memory values 48.5922837, 0.98536933, and
0.07423476. Rescoring the archived uint16 outputs independently produced
48.5910100 dB, 0.98537358, and 0.07423351; the small difference is output
quantization and numerical variation.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/benchmark.py \
  --input-dir data/benchmark/preprocessed_input_20240513 \
  --gt-dir data/benchmark/gt \
  --weights checkpoints/paper_model_best.pth --lpips
```

## Noise model

For an exposure reduction ratio sampled uniformly from 1 to 20, the clean ADU
image is divided by that ratio. Shot noise is sampled in photon space using the
calibrated gain. EM multiplication noise uses a factor sampled uniformly from
1.3 to √2. Read-noise standard deviation is sampled from the empirical biased
frame measurements, and the final frame is clipped and quantized to 16 bits.

The historical code loaded per-pixel fixed-pattern logistic coefficients, but
the final “v5” training branch set their mean contribution to zero and used the
empirical read-noise distribution. The public implementation preserves that
executed behavior rather than uncommented exploratory alternatives.

## Known limitations

- The recovered `records` file mixes old preprocessing variants and duplicate
  commands; only values tied to a matching log/config are reported here.
- The raw flat-field script reproduces an exploratory gain estimate, not the
  final production gain fit; use the released checksum-verified `params.npy`
  for result reproduction. See `docs/CALIBRATION.md`.
- Full training and exact metric evaluation require the externally hosted data
  and a CUDA GPU. A paper training run used two GPUs.
- Preliminary cooled-camera experiments informed exploration but were not part
  of the final paper pipeline and are not distributed.

## Recovered software environments

The local `uformer1` environment contains Python 3.9.18, PyTorch 1.12.1 with
CUDA 11.3, NumPy 1.26.1, einops 0.7.0, and timm 0.9.7. The released checkpoint
loads strictly in that environment as epoch 537 with 50,879,216 parameters.
The older `uformer` environment contains Python 3.7.13, PyTorch 1.12.1, NumPy
1.21.6, einops 0.4.1, and timm 0.6.7.

For provenance only, the separate SRDTrans comparison environment contains
Python 3.6.13, PyTorch 1.8.0 with CUDA 11.1, NumPy 1.19.5, einops 0.4.1, and
timm 0.6.12. SRDTrans code is not included because it is GPL-3.0; this release
intentionally focuses on the paper's proposed method.
