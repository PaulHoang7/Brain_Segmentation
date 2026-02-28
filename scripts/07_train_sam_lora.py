#!/usr/bin/env python3
"""
[07] Train SAM ViT-B + LoRA for one target region (WT / TC / ET).

Freezes SAM base weights, injects LoRA into attention + mask decoder,
trains with Dice + BCE loss using jittered GT bboxes.

Usage:
    python scripts/07_train_sam_lora.py --target WT [--epochs 30]
    python scripts/07_train_sam_lora.py --target TC
    python scripts/07_train_sam_lora.py --target ET

Output:
    OUTPUT_ROOT/ckpt/sam_lora_{target}_best.pth
    OUTPUT_ROOT/logs/sam_lora_{target}_{timestamp}/

Quality gate:
    - Val Dice(WT) > 0.80
    - Val Dice(TC) > 0.70
    - Val Dice(ET) > 0.60
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datn.config import CKPT_DIR, LOGS_DIR, ensure_dirs, load_hparams
from datn.datasets import SAMDataset
from datn.lora import inject_lora_sam, lora_state_dict, count_params
from datn.losses import sam_seg_loss


def main(target: str = "WT", epochs: int = 30, batch_size: int = 4,
         lr: float = 1e-4):
    ensure_dirs()
    hp = load_hparams()
    sam_hp  = hp.get("sam_train", {})
    lora_hp = hp.get("lora", {})

    epochs     = sam_hp.get("epochs", epochs)
    batch_size = sam_hp.get("batch_size", batch_size)
    lr         = sam_hp.get("lr", lr)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Target: {target} | Device: {device}")

    # ── Load SAM ──
    ckpt_path = CKPT_DIR / hp.get("sam", {}).get("checkpoint", "sam_vit_b_01ec64.pth")
    if not ckpt_path.exists():
        print(f"ERROR: SAM checkpoint not found at {ckpt_path}")
        sys.exit(1)

    # TODO: implement after SAM is installed
    # from segment_anything import sam_model_registry
    # sam = sam_model_registry["vit_b"](checkpoint=str(ckpt_path))
    # sam = inject_lora_sam(sam, rank=lora_hp["rank"], alpha=lora_hp["alpha"],
    #                       dropout=lora_hp["dropout"])
    # sam = sam.to(device)
    # print(count_params(sam))

    # ── Data ──
    # train_ds = SAMDataset(split="train", target=target,
    #                       jitter=True,
    #                       jitter_shift=sam_hp.get("bbox_jitter_shift", 0.1),
    #                       jitter_scale=sam_hp.get("bbox_jitter_scale", 0.1))

    # TODO: training loop, validation, save best LoRA weights
    raise NotImplementedError(
        "Install segment-anything, then implement training loop with "
        "LoRA injection, Dice+BCE loss, cosine scheduler."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, choices=["WT", "TC", "ET"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    main(**vars(parser.parse_args()))
