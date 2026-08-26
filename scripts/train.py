#!/usr/bin/env python3
"""Train using the paper-era optimization and validation protocol."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from emccd_denoising.config import load_config
from emccd_denoising.data import PairedDataset, SyntheticDataset
from emccd_denoising.model import build_uformer, load_checkpoint
from emccd_denoising.noise import Calibration, EMCCDNoiseModel
from emccd_denoising.scheduler import GradualWarmupScheduler


def batch_psnr(output, target):
    errors = (output.clamp(0, 1) - target.clamp(0, 1)).square().flatten(1).mean(1)
    return (-10 * torch.log10(errors.clamp_min(1e-12))).sum()


def checkpoint_state(epoch, model, optimizer, validation_psnr):
    return {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "validation_psnr": validation_psnr,
    }


def validate(model, loader, dataset_size, device):
    model.eval()
    score_sum = 0.0
    with torch.inference_mode():
        for target, noisy, _ in loader:
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                restored = model(noisy.to(device)).clamp(0, 1)
            score_sum += batch_psnr(restored, target.to(device)).item()
    return score_sum / dataset_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/paper.yaml"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--epochs", type=int, help="Override epochs (smoke tests only)")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--output-dir", type=Path, help="Override output directory")
    parser.add_argument("--resume-from", type=Path, help="Override the config checkpoint and resume optimizer state")
    parser.add_argument("--max-train-steps", type=int, help="Limit batches per epoch for integration tests")
    parser.add_argument("--max-validation-items", type=int, help="Limit validation images for integration tests")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
    noise_cfg, train_cfg, data_cfg = cfg["noise"], cfg["training"], cfg["data"]
    epochs = args.epochs or train_cfg["epochs"]
    batch_size = args.batch_size or train_cfg["batch_size"]
    workers = train_cfg.get("workers", 4) if args.workers is None else args.workers
    output_dir = args.output_dir or Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    calibration = Calibration.load(noise_cfg["calibration_dir"])
    noise = EMCCDNoiseModel(calibration, noise_cfg["min_ratio"], noise_cfg["max_ratio"])
    train_set = SyntheticDataset(
        data_cfg["train_dir"], noise, train_cfg["patch_size"], seed,
        full_frame_noise=train_cfg.get("historical_full_frame_noise", True),
    )
    val_set = PairedDataset(
        data_cfg["validation_input_dir"], data_cfg["validation_gt_dir"], train_cfg["patch_size"],
        random_crop=True,
    )
    if args.max_validation_items:
        val_set = Subset(val_set, range(min(args.max_validation_items, len(val_set))))
    train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=workers, drop_last=False)
    val_loader = DataLoader(
        val_set, batch_size, shuffle=False, num_workers=workers,
        pin_memory=True, drop_last=False,
    )

    model = build_uformer(train_cfg["patch_size"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"],
        betas=(0.9, 0.999), eps=1e-8,
    )
    start_epoch = 1
    resume_from = args.resume_from or train_cfg.get("resume_from")
    if resume_from:
        start_epoch = (load_checkpoint(model, resume_from, "cpu") or 0) + 1
        saved = torch.load(resume_from, map_location="cpu")
        if "optimizer" in saved:
            optimizer.load_state_dict(saved["optimizer"])

    device = torch.device(args.device)
    model = model.to(device)
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, train_cfg["epochs"] - train_cfg["warmup_epochs"], eta_min=1e-6,
    )
    scheduler = GradualWarmupScheduler(
        optimizer, multiplier=1, total_epoch=train_cfg["warmup_epochs"], after_scheduler=cosine,
    )
    scheduler.step()
    for _ in range(1, start_epoch):
        scheduler.step()
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    best = float("-inf")
    initial_score = validate(model, val_loader, len(val_set), device)
    print(json.dumps({"initial_validation_psnr": initial_score}), flush=True)
    if not train_cfg.get("historical_first_epoch_eval_mode", True):
        model.train()

    for epoch in range(start_epoch, epochs + 1):
        epoch_loss = 0.0
        for step, (target, noisy) in enumerate(train_loader, 1):
            target, noisy = target.to(device), noisy.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                restored = model(noisy)
                loss = torch.sqrt((restored - target).square() + 1e-6).mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            if args.max_train_steps and step >= args.max_train_steps:
                break

        score = validate(model, val_loader, len(val_set), device)
        model.train()  # Historical trainer first did this after epoch-1 validation.
        state = checkpoint_state(epoch, model, optimizer, score)
        torch.save(state, output_dir / "model_latest.pth")
        if score > best:
            best = score
            torch.save(state, output_dir / "model_best.pth")
        scheduler.step()
        record = {
            "epoch": epoch, "loss_sum": epoch_loss, "validation_psnr": score,
            "best_psnr": best, "learning_rate": scheduler.get_last_lr()[0],
        }
        with (output_dir / "training.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
