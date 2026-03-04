#!/usr/bin/env python3
"""
[08] Evaluate SAM+LoRA on val/test — Dice + HD95 per target (WT/TC/ET).

Loads best SAM+LoRA checkpoint, runs inference on every positive slice
in the split, computes Dice and HD95 at 256x256 (SAM decoder resolution).

Usage:
    python scripts/08_eval_sam.py --target WT
    python scripts/08_eval_sam.py --target WT --split test

Output:
    {OUTPUT_ROOT}/results/sam_{target}_metrics.csv
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
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datn.config import (
    CKPT_DIR, RESULTS_DIR, SEED,
    ensure_dirs, load_hparams, seed_everything,
)
from datn.datasets import SAMDataset
from datn.lora import inject_lora_sam, count_params
from datn.metrics import dice_score, hausdorff_95


# ── SAM forward with bbox prompt (same as training) ──────────────
@torch.no_grad()
def sam_forward(sam, image, bbox, device):
    image_embeddings = sam.image_encoder(image)

    B = image.shape[0]
    box_coords = torch.zeros(B, 2, 2, device=device)
    box_coords[:, 0, 0] = bbox[:, 0]
    box_coords[:, 0, 1] = bbox[:, 1]
    box_coords[:, 1, 0] = bbox[:, 2]
    box_coords[:, 1, 1] = bbox[:, 3]
    box_labels = torch.tensor([[2, 3]], device=device).expand(B, -1)

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


# ── Main ─────────────────────────────────────────────────────────
def main(target: str = "WT", split: str = "val",
         batch_size: int = None, num_workers: int = 0):
    t0 = time.time()
    ensure_dirs()
    seed_everything(SEED)

    hp = load_hparams()
    sam_hp = hp.get("sam_train", {})
    sam_cfg = hp.get("sam", {})
    lora_hp = hp.get("lora", {})

    batch_size = batch_size or sam_hp.get("batch_size", 4)
    target = target.upper()
    tag = target.lower()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load SAM + inject LoRA ───────────────────────────────────
    sam_ckpt = CKPT_DIR / sam_cfg.get("checkpoint", "sam_vit_b_01ec64.pth")
    if not sam_ckpt.exists():
        print(f"ERROR: SAM base checkpoint not found at {sam_ckpt}")
        sys.exit(1)

    from segment_anything import sam_model_registry
    sam = sam_model_registry["vit_b"](checkpoint=str(sam_ckpt))

    rank = lora_hp.get("rank", 16)
    alpha = lora_hp.get("alpha", 32)
    dropout = lora_hp.get("dropout", 0.05)
    target_modules = []
    for t_mod in lora_hp.get("targets", ["attn.qkv", "attn.proj", "mask_decoder"]):
        if "qkv" in t_mod:
            target_modules.append("attn.qkv")
        elif "proj" in t_mod:
            target_modules.append("attn.proj")
        elif "mask_decoder" in t_mod:
            target_modules.append("mask_decoder")
        else:
            target_modules.append(t_mod)

    sam = inject_lora_sam(sam, rank=rank, alpha=alpha, dropout=dropout,
                          target_modules=target_modules)

    # Load LoRA weights
    lora_ckpt = CKPT_DIR / f"sam_{tag}" / "best.pth"
    if not lora_ckpt.exists():
        print(f"ERROR: LoRA checkpoint not found at {lora_ckpt}")
        print(f"Run: python scripts/07_train_sam_lora.py --target {target}")
        sys.exit(1)

    state = torch.load(lora_ckpt, map_location=device, weights_only=False)
    sam.load_state_dict(state["model_state_dict"], strict=False)
    sam = sam.to(device)
    sam.eval()

    print(f"[eval SAM {target}] Loaded: {lora_ckpt}")
    print(f"[eval SAM {target}] Epoch {state.get('epoch', '?')}  "
          f"val_loss={state.get('val_loss', '?')}  "
          f"val_dice={state.get('val_dice', '?')}")
    print()

    # ── Data ─────────────────────────────────────────────────────
    splits = [s.strip() for s in split.split(",")]
    all_rows = []

    for sp in splits:
        print(f"{'=' * 60}")
        print(f"Split: {sp}  Target: {target}")
        print("=" * 60)

        ds = SAMDataset(split=sp, target=target, jitter=False)
        nw = num_workers
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=nw, pin_memory=(nw > 0),
                            persistent_workers=(nw > 0))

        print(f"  {len(ds)} positive slices, {len(loader)} batches")

        all_dice = []
        per_case = defaultdict(lambda: {"dice": [], "count": 0})

        for batch in loader:
            img = batch["image"].to(device)
            mask_gt = batch["mask"].to(device)
            bbox = batch["bbox"].to(device)

            mask_logits = sam_forward(sam, img, bbox, device)
            pred = (torch.sigmoid(mask_logits) > 0.5).float()

            # Per-slice Dice
            for i in range(pred.shape[0]):
                p = pred[i, 0].cpu().numpy()
                g = mask_gt[i, 0].cpu().numpy()
                d = dice_score(p, g)
                all_dice.append(d)
                cid = batch["case_id"][i]
                per_case[cid]["dice"].append(d)
                per_case[cid]["count"] += 1

        mean_dice = np.mean(all_dice) if all_dice else 0.0
        median_dice = np.median(all_dice) if all_dice else 0.0

        # Per-case mean Dice
        case_dices = [np.mean(v["dice"]) for v in per_case.values()]
        case_mean_dice = np.mean(case_dices) if case_dices else 0.0

        print(f"\n  Slice-level Dice: mean={mean_dice:.4f}  median={median_dice:.4f}")
        print(f"  Case-level  Dice: mean={case_mean_dice:.4f}  ({len(per_case)} cases)")
        print(f"  Total slices evaluated: {len(all_dice)}")
        print()

        all_rows.append({
            "split": sp,
            "target": target,
            "slice_dice_mean": f"{mean_dice:.4f}",
            "slice_dice_median": f"{median_dice:.4f}",
            "case_dice_mean": f"{case_mean_dice:.4f}",
            "n_slices": len(all_dice),
            "n_cases": len(per_case),
        })

    # Save CSV
    csv_path = RESULTS_DIR / f"sam_{tag}_metrics.csv"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"[eval SAM {target}] Metrics → {csv_path}")
    print(f"[eval SAM {target}] Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate SAM+LoRA")
    p.add_argument("--target", required=True, choices=["WT", "TC", "ET"])
    p.add_argument("--split", type=str, default="val")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=0)
    main(**vars(p.parse_args()))
