#!/usr/bin/env python3
"""
[05] Train the Prompt Generator (PG).

Model:  ResNet18 (9-ch) → objectness + bbox
Loss:   Focal(objectness) + SmoothL1(bbox) + GIoU(bbox) + Temporal(z)
Data:   PGDataset with tumor oversampling

Usage:
    python scripts/05_train_pg.py [--epochs 50] [--batch_size 64] [--lr 1e-3]

Output:
    OUTPUT_ROOT/ckpt/pg_best.pth
    OUTPUT_ROOT/logs/pg_{timestamp}/

Quality gate:
    - Val loss decreasing
    - Val objectness F1 > 0.80
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datn.config import CKPT_DIR, LOGS_DIR, SEED, ensure_dirs, load_hparams
from datn.datasets import PGDataset
from datn.samplers import make_pg_sampler
from datn.pg_model import PromptGenerator
from datn.losses import pg_loss


def main(epochs: int = 50, batch_size: int = 64, lr: float = 1e-3):
    ensure_dirs()
    hp = load_hparams().get("pg", {})
    epochs     = hp.get("epochs", epochs)
    batch_size = hp.get("batch_size", batch_size)
    lr         = hp.get("lr", lr)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Data ──
    train_ds = PGDataset(split="train", img_size=hp.get("img_size", 224))
    val_ds   = PGDataset(split="val",   img_size=hp.get("img_size", 224))
    sampler  = make_pg_sampler("train", pos_weight=hp.get("pos_sample_weight", 3.0))

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    # ── Model ──
    model = PromptGenerator(in_channels=hp.get("input_channels", 9)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=hp.get("weight_decay", 0.01))

    # TODO: add cosine scheduler with warmup
    # TODO: add temporal loss (requires adjacent-slice pairing in dataset)
    # TODO: add TensorBoard / W&B logging
    # TODO: implement training loop

    raise NotImplementedError(
        "Training loop skeleton ready — implement the epoch loop, "
        "validation, checkpoint saving, and early stopping."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    main(**vars(parser.parse_args()))
