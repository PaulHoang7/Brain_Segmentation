#!/usr/bin/env python3
"""
[05] Train the Prompt Generator (PG).

Model:   ResNet18 (9-ch conv1) → objectness logit + bbox [0,1]
Loss:    Focal(obj) + SmoothL1(bbox) + GIoU(bbox) + Temporal(z-consistency)
Data:    PGDataset2p5D with weighted tumour oversampling
Sched:   Cosine annealing with linear warmup

Usage:
    python scripts/05_train_pg.py                          # from hparams.yaml
    python scripts/05_train_pg.py --epochs 2 --batch_size 16  # override

Output:
    {OUTPUT_ROOT}/ckpt/pg/best.pth
    {OUTPUT_ROOT}/ckpt/pg/last.pth
    {OUTPUT_ROOT}/logs/pg/train_log.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datn.config import CKPT_DIR, LOGS_DIR, SEED, ensure_dirs, load_hparams, seed_everything
from datn.datasets import PGDataset2p5D
from datn.losses import focal_loss, giou_loss, temporal_bbox_loss
from datn.metrics import pg_bbox_iou, pg_detection_metrics
from datn.pg_model import PromptGenerator
from datn.samplers import make_pg_sampler


# ── Cosine scheduler with linear warmup ──────────────────────────
def build_scheduler(optimizer, total_steps: int, warmup_steps: int):
    """Linear warmup → cosine decay to 0."""
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


# ── Within-batch temporal loss ───────────────────────────────────
def compute_temporal_loss(case_ids, zs, bbox_preds, has_tumors, device):
    """
    Find same-case consecutive z-pairs within a batch and compute
    SmoothL1 between their bbox predictions.
    """
    # Group indices by case_id
    case_groups = defaultdict(list)
    for i in range(len(case_ids)):
        if has_tumors[i]:
            case_groups[case_ids[i]].append((int(zs[i]), i))

    pairs = []
    for cid, items in case_groups.items():
        items.sort(key=lambda x: x[0])
        for k in range(len(items) - 1):
            z_a, idx_a = items[k]
            z_b, idx_b = items[k + 1]
            if z_b - z_a == 1:  # consecutive
                pairs.append((idx_a, idx_b))

    if not pairs:
        return torch.tensor(0.0, device=device)

    idx_a = [p[0] for p in pairs]
    idx_b = [p[1] for p in pairs]
    return temporal_bbox_loss(bbox_preds[idx_a], bbox_preds[idx_b])


# ── Train one epoch ──────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler, device, hp,
                epoch: int = 0, total_epochs: int = 0,
                max_batches: int = 0):
    model.train()
    total_loss = 0.0
    loss_parts = defaultdict(float)
    n_batches = 0
    n_total = max_batches if max_batches else len(loader)
    log_every = max(1, min(n_total // 5, 200))  # ~5 prints per epoch

    temporal_w = hp.get("temporal_weight", 0.5)
    t_ep = time.time()

    for batch in loader:
        img = batch["image"].to(device)
        obj_gt = batch["objectness"].to(device)
        bbox_gt = batch["bbox"].to(device)
        case_ids = batch["case_id"]
        zs = batch["z"]

        out = model(img)
        obj_logit = out["objectness"]
        bbox_pred = out["bbox"]

        # Core losses
        has_tumor = obj_gt.bool()
        l_obj = focal_loss(obj_logit, obj_gt, gamma=hp.get("focal_gamma", 2.0))

        if has_tumor.any():
            bp = bbox_pred[has_tumor]
            bg = bbox_gt[has_tumor]
            l_reg = F.smooth_l1_loss(bp, bg)
            l_giou = giou_loss(bp, bg)
        else:
            l_reg = torch.tensor(0.0, device=device)
            l_giou = torch.tensor(0.0, device=device)

        # Temporal consistency
        l_temp = compute_temporal_loss(
            case_ids, zs, bbox_pred, obj_gt.cpu().numpy(), device)

        loss = l_obj + l_reg + l_giou + temporal_w * l_temp

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        loss_parts["obj"] += l_obj.item()
        loss_parts["reg"] += l_reg.item()
        loss_parts["giou"] += l_giou.item()
        loss_parts["temp"] += l_temp.item()
        n_batches += 1

        if n_batches % log_every == 0 or n_batches == n_total:
            elapsed = time.time() - t_ep
            avg_loss = total_loss / n_batches
            print(f"  Ep {epoch}/{total_epochs}  "
                  f"batch {n_batches}/{n_total}  "
                  f"loss={avg_loss:.4f}  "
                  f"lr={optimizer.param_groups[0]['lr']:.5f}  "
                  f"{elapsed:.0f}s", flush=True)

        if max_batches and n_batches >= max_batches:
            break

    avg = {k: v / n_batches for k, v in loss_parts.items()}
    avg["total"] = total_loss / n_batches
    return avg


# ── Validate ─────────────────────────────────────────────────────
@torch.no_grad()
def validate(model, loader, device, hp, max_batches: int = 0):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_obj_pred = []
    all_obj_gt = []
    all_bbox_pred = []
    all_bbox_gt = []
    all_has_tumor = []

    for batch in loader:
        img = batch["image"].to(device)
        obj_gt = batch["objectness"].to(device)
        bbox_gt = batch["bbox"].to(device)

        out = model(img)
        obj_logit = out["objectness"]
        bbox_pred = out["bbox"]

        has_tumor = obj_gt.bool()
        l_obj = focal_loss(obj_logit, obj_gt, gamma=hp.get("focal_gamma", 2.0))

        if has_tumor.any():
            bp = bbox_pred[has_tumor]
            bg = bbox_gt[has_tumor]
            l_reg = F.smooth_l1_loss(bp, bg)
            l_giou = giou_loss(bp, bg)
        else:
            l_reg = torch.tensor(0.0, device=device)
            l_giou = torch.tensor(0.0, device=device)

        loss = l_obj + l_reg + l_giou
        total_loss += loss.item()
        n_batches += 1

        obj_prob = torch.sigmoid(obj_logit).cpu().numpy()
        all_obj_pred.append(obj_prob)
        all_obj_gt.append(obj_gt.cpu().numpy())
        all_bbox_pred.append(bbox_pred.cpu().numpy())
        all_bbox_gt.append(bbox_gt.cpu().numpy())
        all_has_tumor.append(has_tumor.cpu().numpy())

        if max_batches and n_batches >= max_batches:
            break

    avg_loss = total_loss / n_batches

    # Metrics
    obj_pred = np.concatenate(all_obj_pred)
    obj_gt = np.concatenate(all_obj_gt)
    det = pg_detection_metrics(obj_pred, obj_gt)

    # Bbox IoU on tumour slices only
    has_t = np.concatenate(all_has_tumor).astype(bool)
    if has_t.any():
        bp = np.concatenate(all_bbox_pred)[has_t]
        bg = np.concatenate(all_bbox_gt)[has_t]
        iou = pg_bbox_iou(bp, bg)
    else:
        iou = 0.0

    return {
        "loss": avg_loss,
        "precision": det["precision"],
        "recall": det["recall"],
        "f1": det["f1"],
        "bbox_iou": iou,
    }


# ── Main ─────────────────────────────────────────────────────────
def main(epochs: int = None, batch_size: int = None, lr: float = None,
         num_workers: int = 0, max_batches: int = 0):
    t0 = time.time()
    ensure_dirs()
    hp = load_hparams().get("pg", {})

    # CLI overrides yaml
    epochs = epochs or hp.get("epochs", 50)
    batch_size = batch_size or hp.get("batch_size", 64)
    lr = lr or hp.get("lr", 1e-3)
    img_size = hp.get("img_size", 224)
    pos_weight = hp.get("pos_sample_weight", 3.0)
    wd = hp.get("weight_decay", 0.01)
    warmup_ratio = hp.get("warmup_ratio", 0.1)

    seed_everything(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Output dirs
    ckpt_dir = CKPT_DIR / "pg"
    log_dir = LOGS_DIR / "pg"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"[PG Train] device={device}  epochs={epochs}  bs={batch_size}  lr={lr}")
    print(f"[PG Train] img_size={img_size}  pos_weight={pos_weight}  "
          f"num_workers={num_workers}")
    print(f"[PG Train] ckpt → {ckpt_dir}")
    print(f"[PG Train] logs → {log_dir}")
    print()

    # ── Data ─────────────────────────────────────────────────────
    train_ds = PGDataset2p5D(split="train", img_size=img_size)
    val_ds = PGDataset2p5D(split="val", img_size=img_size)
    sampler = make_pg_sampler("train", pos_weight=pos_weight)

    # NFS note: num_workers=0 is fastest with CaseGroupedSampler
    # because _VolumeCache is shared in the main process.
    # With workers>0 each fork gets its own cache → redundant NFS reads.
    nw = num_workers
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=nw, pin_memory=(nw > 0), drop_last=True,
        persistent_workers=(nw > 0))
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=nw, pin_memory=(nw > 0),
        persistent_workers=(nw > 0))

    # ── Model + Optimizer + Scheduler ────────────────────────────
    model = PromptGenerator(
        in_channels=hp.get("input_channels", 9),
        pretrained_backbone=True,
    ).to(device)
    print(f"[PG Train] params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    total_steps = epochs * len(train_loader)
    warmup_steps = int(warmup_ratio * total_steps)
    scheduler = build_scheduler(optimizer, total_steps, warmup_steps)
    print(f"[PG Train] total_steps={total_steps}  warmup_steps={warmup_steps}")
    print()

    # ── CSV log ──────────────────────────────────────────────────
    csv_path = log_dir / "train_log.csv"
    csv_fields = [
        "epoch", "train_loss", "train_obj", "train_reg", "train_giou", "train_temp",
        "val_loss", "val_prec", "val_rec", "val_f1", "val_iou", "lr",
    ]
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    csv_writer.writeheader()

    # ── Training loop ────────────────────────────────────────────
    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(1, epochs + 1):
        ep_t0 = time.time()

        train_metrics = train_epoch(model, train_loader, optimizer, scheduler,
                                    device, hp, epoch=epoch,
                                    total_epochs=epochs,
                                    max_batches=max_batches)
        val_metrics = validate(model, val_loader, device, hp,
                               max_batches=max_batches)

        cur_lr = optimizer.param_groups[0]["lr"]
        ep_time = time.time() - ep_t0

        # Log
        row = {
            "epoch": epoch,
            "train_loss": f"{train_metrics['total']:.4f}",
            "train_obj": f"{train_metrics['obj']:.4f}",
            "train_reg": f"{train_metrics['reg']:.4f}",
            "train_giou": f"{train_metrics['giou']:.4f}",
            "train_temp": f"{train_metrics['temp']:.4f}",
            "val_loss": f"{val_metrics['loss']:.4f}",
            "val_prec": f"{val_metrics['precision']:.4f}",
            "val_rec": f"{val_metrics['recall']:.4f}",
            "val_f1": f"{val_metrics['f1']:.4f}",
            "val_iou": f"{val_metrics['bbox_iou']:.4f}",
            "lr": f"{cur_lr:.6f}",
        }
        csv_writer.writerow(row)
        csv_file.flush()

        # Print
        is_best = val_metrics["loss"] < best_val_loss
        marker = " *BEST*" if is_best else ""
        print(f"Ep {epoch:3d}/{epochs}  "
              f"train={train_metrics['total']:.4f} "
              f"(obj={train_metrics['obj']:.3f} reg={train_metrics['reg']:.3f} "
              f"giou={train_metrics['giou']:.3f} temp={train_metrics['temp']:.3f})  "
              f"val={val_metrics['loss']:.4f}  "
              f"F1={val_metrics['f1']:.3f}  IoU={val_metrics['bbox_iou']:.3f}  "
              f"lr={cur_lr:.5f}  {ep_time:.0f}s{marker}")

        # Save checkpoints
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_metrics["loss"],
            "val_f1": val_metrics["f1"],
            "val_iou": val_metrics["bbox_iou"],
            "hparams": hp,
        }

        torch.save(state, ckpt_dir / "last.pth")

        if is_best:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            torch.save(state, ckpt_dir / "best.pth")

    csv_file.close()

    elapsed = time.time() - t0
    print(f"\n[PG Train] Done in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"[PG Train] Best epoch: {best_epoch}  val_loss: {best_val_loss:.4f}")
    print(f"[PG Train] Checkpoints: {ckpt_dir}/{{best,last}}.pth")
    print(f"[PG Train] Log: {csv_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train Prompt Generator")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--num_workers", type=int, default=0,
                   help="DataLoader workers (0=main process, best for NFS)")
    p.add_argument("--max_batches", type=int, default=0,
                   help="Stop each epoch after N batches (0=full epoch)")
    main(**vars(p.parse_args()))
