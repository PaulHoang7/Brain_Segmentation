#!/usr/bin/env python3
"""
[07] Train SAM ViT-B + LoRA for one target region (WT / TC / ET).

Freezes SAM base weights, injects LoRA into attention + mask decoder,
trains with Dice + BCE loss using jittered GT bboxes as prompts.

Usage:
    python scripts/07_train_sam_lora.py --target WT
    python scripts/07_train_sam_lora.py --target WT --epochs 5 --max_batches 50  # smoke
    python scripts/07_train_sam_lora.py --target WT --resume --epochs 30

Output:
    {OUTPUT_ROOT}/ckpt/sam_{target}/best.pth
    {OUTPUT_ROOT}/ckpt/sam_{target}/last.pth
    {OUTPUT_ROOT}/logs/sam_{target}/train_log.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datn.config import CKPT_DIR, LOGS_DIR, SEED, ensure_dirs, load_hparams, seed_everything
from datn.datasets import SAMDataset
from datn.lora import inject_lora_sam, lora_state_dict, count_params
from datn.losses import sam_seg_loss


# ── Cosine scheduler with linear warmup ──────────────────────────
def build_scheduler(optimizer, total_steps: int, warmup_steps: int):
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


# ── SAM forward with bbox prompt ─────────────────────────────────
def sam_forward(sam, image, bbox, device):
    """
    Run SAM forward pass with a bbox prompt.
    image: (B, 3, 1024, 1024)
    bbox:  (B, 4) in 1024 space
    Returns: mask logits (B, 1, 256, 256)
    """
    with torch.no_grad():
        image_embeddings = sam.image_encoder(image)

    # Prepare sparse (bbox) + dense (none) prompts
    B = image.shape[0]
    # SAM expects bbox as (B, 4) -> needs to be point format for prompt encoder
    # bbox prompt: 2 points (top-left, bottom-right) with labels [2, 3]
    boxes = bbox.reshape(B, 1, 4)  # (B, 1, 4)
    # Convert to point pairs: (B, 2, 2) with (B, 2) labels
    box_coords = torch.zeros(B, 2, 2, device=device)
    box_coords[:, 0, 0] = boxes[:, 0, 0]  # x1
    box_coords[:, 0, 1] = boxes[:, 0, 1]  # y1
    box_coords[:, 1, 0] = boxes[:, 0, 2]  # x2
    box_coords[:, 1, 1] = boxes[:, 0, 3]  # y2
    box_labels = torch.tensor([[2, 3]], device=device).expand(B, -1)  # box prompt labels

    sparse_embeddings, dense_embeddings = sam.prompt_encoder(
        points=(box_coords, box_labels),
        boxes=None,
        masks=None,
    )

    low_res_masks, iou_predictions = sam.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )

    return low_res_masks  # (B, 1, 256, 256)


# ── Train one epoch ──────────────────────────────────────────────
def train_epoch(sam, loader, optimizer, scheduler, device,
                epoch: int, total_epochs: int, max_batches: int = 0):
    sam.train()
    # Only LoRA params are unfrozen, but we set train() for dropout
    total_loss = 0.0
    n_batches = 0
    n_total = max_batches if max_batches else len(loader)
    log_every = max(1, min(n_total // 5, 50))
    t_ep = time.time()

    for batch in loader:
        img = batch["image"].to(device)       # (B, 3, 1024, 1024)
        mask_gt = batch["mask"].to(device)     # (B, 1, 256, 256)
        bbox = batch["bbox"].to(device)        # (B, 4)

        mask_logits = sam_forward(sam, img, bbox, device)  # (B, 1, 256, 256)
        loss = sam_seg_loss(mask_logits, mask_gt)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in sam.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        n_batches += 1

        if n_batches % log_every == 0 or n_batches == n_total:
            elapsed = time.time() - t_ep
            avg_loss = total_loss / n_batches
            print(f"  Ep {epoch}/{total_epochs}  "
                  f"batch {n_batches}/{n_total}  "
                  f"loss={avg_loss:.4f}  "
                  f"lr={optimizer.param_groups[0]['lr']:.6f}  "
                  f"{elapsed:.0f}s", flush=True)

        if max_batches and n_batches >= max_batches:
            break

    return total_loss / n_batches


# ── Validate ─────────────────────────────────────────────────────
@torch.no_grad()
def validate(sam, loader, device, max_batches: int = 0):
    sam.eval()
    total_loss = 0.0
    total_dice = 0.0
    n_batches = 0

    for batch in loader:
        img = batch["image"].to(device)
        mask_gt = batch["mask"].to(device)
        bbox = batch["bbox"].to(device)

        mask_logits = sam_forward(sam, img, bbox, device)
        loss = sam_seg_loss(mask_logits, mask_gt)
        total_loss += loss.item()

        # Compute Dice
        pred = (torch.sigmoid(mask_logits) > 0.5).float()
        intersection = (pred * mask_gt).sum(dim=(1, 2, 3))
        union = pred.sum(dim=(1, 2, 3)) + mask_gt.sum(dim=(1, 2, 3))
        dice = (2 * intersection + 1.0) / (union + 1.0)
        total_dice += dice.mean().item()

        n_batches += 1
        if max_batches and n_batches >= max_batches:
            break

    return {
        "loss": total_loss / n_batches,
        "dice": total_dice / n_batches,
    }


# ── Main ─────────────────────────────────────────────────────────
def main(target: str = "WT", epochs: int = None, batch_size: int = None,
         lr: float = None, num_workers: int = 0, max_batches: int = 0,
         resume: bool = False):
    t0 = time.time()
    ensure_dirs()
    seed_everything(SEED)

    hp = load_hparams()
    sam_hp = hp.get("sam_train", {})
    lora_hp = hp.get("lora", {})
    sam_cfg = hp.get("sam", {})

    epochs = epochs or sam_hp.get("epochs", 30)
    batch_size = batch_size or sam_hp.get("batch_size", 4)
    lr = lr or sam_hp.get("lr", 1e-4)
    wd = sam_hp.get("weight_decay", 0.01)
    warmup_ratio = sam_hp.get("warmup_ratio", 0.1)
    jitter_shift = sam_hp.get("bbox_jitter_shift", 0.15)
    jitter_scale = sam_hp.get("bbox_jitter_scale", 0.15)

    target = target.upper()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Output dirs
    tag = target.lower()
    ckpt_dir = CKPT_DIR / f"sam_{tag}"
    log_dir = LOGS_DIR / f"sam_{tag}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"[SAM+LoRA {target}] device={device}  epochs={epochs}  bs={batch_size}  lr={lr}")
    print(f"[SAM+LoRA {target}] jitter_shift={jitter_shift}  jitter_scale={jitter_scale}")
    print(f"[SAM+LoRA {target}] ckpt → {ckpt_dir}")
    print()

    # ── Load SAM ViT-B ───────────────────────────────────────────
    sam_ckpt = CKPT_DIR / sam_cfg.get("checkpoint", "sam_vit_b_01ec64.pth")
    if not sam_ckpt.exists():
        print(f"ERROR: SAM checkpoint not found at {sam_ckpt}")
        print("Download from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
        print(f"Place it at: {sam_ckpt}")
        sys.exit(1)

    from segment_anything import sam_model_registry
    sam = sam_model_registry["vit_b"](checkpoint=str(sam_ckpt))

    # ── Inject LoRA ──────────────────────────────────────────────
    rank = lora_hp.get("rank", 16)
    alpha = lora_hp.get("alpha", 32)
    dropout = lora_hp.get("dropout", 0.05)

    # Map config target names to SAM module patterns
    target_modules = []
    for t in lora_hp.get("targets", ["attn.qkv", "attn.proj", "mask_decoder"]):
        if "qkv" in t:
            target_modules.append("attn.qkv")
        elif "proj" in t:
            target_modules.append("attn.proj")
        elif "mask_decoder" in t:
            target_modules.append("mask_decoder")
        else:
            target_modules.append(t)

    sam = inject_lora_sam(sam, rank=rank, alpha=alpha, dropout=dropout,
                          target_modules=target_modules)
    sam = sam.to(device)

    params = count_params(sam)
    print(f"[SAM+LoRA {target}] params: total={params['total']:,}  "
          f"trainable={params['trainable']:,}  frozen={params['frozen']:,}")
    print(f"[SAM+LoRA {target}] LoRA ratio: {params['trainable']/params['total']*100:.2f}%")
    print()

    # ── Data ─────────────────────────────────────────────────────
    train_ds = SAMDataset(split="train", target=target,
                          jitter=True,
                          jitter_shift=jitter_shift,
                          jitter_scale=jitter_scale)
    val_ds = SAMDataset(split="val", target=target,
                        jitter=False)

    nw = num_workers
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=nw, pin_memory=(nw > 0), drop_last=True,
        persistent_workers=(nw > 0))
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=nw, pin_memory=(nw > 0),
        persistent_workers=(nw > 0))

    print(f"[SAM+LoRA {target}] train: {len(train_ds)} slices  "
          f"val: {len(val_ds)} slices")
    print(f"[SAM+LoRA {target}] train batches: {len(train_loader)}  "
          f"val batches: {len(val_loader)}")
    print()

    # ── Optimizer + Scheduler ────────────────────────────────────
    trainable_params = [p for p in sam.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=wd)

    total_steps = epochs * len(train_loader)
    warmup_steps = int(warmup_ratio * total_steps)
    scheduler = build_scheduler(optimizer, total_steps, warmup_steps)

    # ── Resume ───────────────────────────────────────────────────
    start_epoch = 1
    best_val_loss = float("inf")
    best_epoch = -1

    if resume:
        last_ckpt = ckpt_dir / "last.pth"
        if not last_ckpt.exists():
            print(f"[SAM+LoRA {target}] WARNING: --resume but {last_ckpt} not found, training from scratch")
        else:
            ckpt_state = torch.load(last_ckpt, map_location=device, weights_only=False)
            # Load LoRA weights
            sam.load_state_dict(ckpt_state["model_state_dict"], strict=False)
            optimizer.load_state_dict(ckpt_state["optimizer_state_dict"])
            start_epoch = ckpt_state["epoch"] + 1

            if "scheduler_state_dict" in ckpt_state:
                scheduler.load_state_dict(ckpt_state["scheduler_state_dict"])
            else:
                resumed_steps = ckpt_state["epoch"] * len(train_loader)
                for _ in range(resumed_steps):
                    scheduler.step()

            best_ckpt = ckpt_dir / "best.pth"
            if best_ckpt.exists():
                best_state = torch.load(best_ckpt, map_location="cpu", weights_only=False)
                best_val_loss = best_state.get("val_loss", float("inf"))
                best_epoch = best_state.get("epoch", -1)

            print(f"[SAM+LoRA {target}] Resumed from epoch {ckpt_state['epoch']}  "
                  f"(best_epoch={best_epoch}  best_val_loss={best_val_loss:.4f})")

    if start_epoch > epochs:
        print(f"[SAM+LoRA {target}] Already trained {start_epoch - 1} epochs. Nothing to do.")
        return

    print(f"[SAM+LoRA {target}] total_steps={total_steps}  warmup_steps={warmup_steps}")
    print(f"[SAM+LoRA {target}] Training epochs {start_epoch}→{epochs}")
    print()

    # ── CSV log ──────────────────────────────────────────────────
    csv_path = log_dir / "train_log.csv"
    csv_fields = ["epoch", "train_loss", "val_loss", "val_dice", "lr"]
    csv_mode = "a" if (resume and csv_path.exists()) else "w"
    csv_file = open(csv_path, csv_mode, newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    if csv_mode == "w":
        csv_writer.writeheader()

    # ── Training loop ────────────────────────────────────────────
    for epoch in range(start_epoch, epochs + 1):
        ep_t0 = time.time()

        train_loss = train_epoch(sam, train_loader, optimizer, scheduler,
                                 device, epoch=epoch, total_epochs=epochs,
                                 max_batches=max_batches)
        val_metrics = validate(sam, val_loader, device,
                               max_batches=max_batches)

        cur_lr = optimizer.param_groups[0]["lr"]
        ep_time = time.time() - ep_t0

        row = {
            "epoch": epoch,
            "train_loss": f"{train_loss:.4f}",
            "val_loss": f"{val_metrics['loss']:.4f}",
            "val_dice": f"{val_metrics['dice']:.4f}",
            "lr": f"{cur_lr:.6f}",
        }
        csv_writer.writerow(row)
        csv_file.flush()

        is_best = val_metrics["loss"] < best_val_loss
        marker = " *BEST*" if is_best else ""
        print(f"Ep {epoch:3d}/{epochs}  "
              f"train={train_loss:.4f}  "
              f"val={val_metrics['loss']:.4f}  "
              f"dice={val_metrics['dice']:.3f}  "
              f"lr={cur_lr:.6f}  {ep_time:.0f}s{marker}")

        # Save checkpoints (LoRA weights only + optimizer)
        state = {
            "epoch": epoch,
            "model_state_dict": lora_state_dict(sam),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_metrics["loss"],
            "val_dice": val_metrics["dice"],
            "target": target,
            "lora_config": {"rank": rank, "alpha": alpha, "dropout": dropout,
                            "target_modules": target_modules},
        }

        torch.save(state, ckpt_dir / "last.pth")

        if is_best:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            torch.save(state, ckpt_dir / "best.pth")

    csv_file.close()

    elapsed = time.time() - t0
    print(f"\n[SAM+LoRA {target}] Done in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"[SAM+LoRA {target}] Best epoch: {best_epoch}  "
          f"val_loss: {best_val_loss:.4f}")
    print(f"[SAM+LoRA {target}] Checkpoints: {ckpt_dir}/{{best,last}}.pth")
    print(f"[SAM+LoRA {target}] Log: {csv_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train SAM+LoRA for target region")
    p.add_argument("--target", required=True, choices=["WT", "TC", "ET"])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--max_batches", type=int, default=0,
                   help="Stop each epoch after N batches (0=full epoch)")
    p.add_argument("--resume", action="store_true",
                   help="Resume from last.pth checkpoint")
    main(**vars(p.parse_args()))
