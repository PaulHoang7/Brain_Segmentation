#!/usr/bin/env python3
"""
[07a] Pre-compute SAM ViT-B encoder embeddings for all positive slices.

The encoder is frozen during LoRA training, so we only need to run it once.
Saves per-case .npz files containing embeddings (fp16), masks, bboxes.

This reduces per-epoch training time from ~8h to ~40min by skipping the
encoder forward pass entirely during training.

Usage:
    python scripts/07a_precompute_embeddings.py --target WT
    python scripts/07a_precompute_embeddings.py --target WT --split train,val

Output:
    {OUTPUT_ROOT}/embeddings/{target}/{split}/{case_id}.npz
    Each .npz contains:
      - embeddings: (N, 256, 64, 64) float16
      - masks:      (N, 256, 256) uint8
      - bboxes:     (N, 4) float32  (in 1024 space)
      - z_indices:  (N,) int32
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datn.config import (
    CKPT_DIR, OUTPUT_ROOT, SEED,
    ensure_dirs, load_hparams, seed_everything,
)
from datn.datasets import SAMDataset
from datn.samplers import DatasetCaseGroupedSampler
from torch.utils.data import DataLoader


def main(target: str = "WT", splits: str = "train,val",
         batch_size: int = 12):
    t0 = time.time()
    ensure_dirs()
    seed_everything(SEED)

    hp = load_hparams()
    sam_cfg = hp.get("sam", {})
    target = target.upper()
    tag = target.lower()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load SAM encoder only
    sam_ckpt = CKPT_DIR / sam_cfg.get("checkpoint", "sam_vit_b_01ec64.pth")
    if not sam_ckpt.exists():
        print(f"ERROR: SAM checkpoint not found at {sam_ckpt}")
        sys.exit(1)

    from segment_anything import sam_model_registry
    sam = sam_model_registry["vit_b"](checkpoint=str(sam_ckpt))
    sam = sam.to(device)
    sam.eval()

    # Freeze everything
    for p in sam.parameters():
        p.requires_grad = False

    print(f"[precompute] target={target}  device={device}  bs={batch_size}")
    print(f"[precompute] SAM encoder loaded from {sam_ckpt}")
    print()

    split_list = [s.strip() for s in splits.split(",")]

    for sp in split_list:
        out_dir = OUTPUT_ROOT / "embeddings" / tag / sp
        out_dir.mkdir(parents=True, exist_ok=True)

        ds = SAMDataset(split=sp, target=target, jitter=False)
        sampler = DatasetCaseGroupedSampler(ds.rows, seed=SEED)
        loader = DataLoader(ds, batch_size=batch_size, sampler=sampler,
                            num_workers=0)

        print(f"[precompute] {sp}: {len(ds)} slices, {len(loader)} batches")

        # Collect per-case data
        case_data = defaultdict(lambda: {
            "embeddings": [], "masks": [], "bboxes": [], "z_indices": []
        })

        n_done = 0
        n_total = len(loader)
        log_every = max(1, n_total // 20)
        t_sp = time.time()

        for batch in loader:
            img = batch["image"].to(device)

            with torch.no_grad(), torch.amp.autocast("cuda"):
                emb = sam.image_encoder(img)  # (B, 256, 64, 64)

            emb_fp16 = emb.cpu().half().numpy()
            masks = batch["mask"][:, 0].numpy().astype(np.uint8)  # (B, 256, 256)
            bboxes = batch["bbox"].numpy()  # (B, 4)

            for i in range(img.shape[0]):
                cid = batch["case_id"][i]
                case_data[cid]["embeddings"].append(emb_fp16[i])
                case_data[cid]["masks"].append(masks[i])
                case_data[cid]["bboxes"].append(bboxes[i])
                case_data[cid]["z_indices"].append(int(batch["z"][i]))

            n_done += 1
            if n_done % log_every == 0:
                elapsed = time.time() - t_sp
                pct = n_done / n_total * 100
                eta = elapsed / n_done * (n_total - n_done)
                print(f"  {sp} {n_done}/{n_total} ({pct:.0f}%)  "
                      f"{elapsed:.0f}s elapsed  ETA {eta:.0f}s", flush=True)

        # Save per-case .npz files
        for cid, data in case_data.items():
            np.savez_compressed(
                out_dir / f"{cid}.npz",
                embeddings=np.stack(data["embeddings"]),  # (N, 256, 64, 64) fp16
                masks=np.stack(data["masks"]),              # (N, 256, 256) uint8
                bboxes=np.stack(data["bboxes"]),            # (N, 4) float32
                z_indices=np.array(data["z_indices"], dtype=np.int32),
            )

        sp_time = time.time() - t_sp
        total_size = sum(f.stat().st_size for f in out_dir.glob("*.npz"))
        print(f"  {sp}: {len(case_data)} cases saved → {out_dir}")
        print(f"  {sp}: {total_size / 1024**3:.1f} GB  ({sp_time:.0f}s)")
        print()

    print(f"[precompute] Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Pre-compute SAM encoder embeddings")
    p.add_argument("--target", required=True, choices=["WT", "TC", "ET"])
    p.add_argument("--splits", type=str, default="train,val")
    p.add_argument("--batch_size", type=int, default=12)
    main(**vars(p.parse_args()))
