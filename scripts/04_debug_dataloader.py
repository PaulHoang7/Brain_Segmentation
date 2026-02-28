#!/usr/bin/env python3
"""
[04] Debug Dataloader — smoke-test PGDataset2p5D + SAMDataset + samplers.

For each dataset:
  1. Instantiate dataset + sampler
  2. Create DataLoader, iterate 1 batch
  3. Print tensor shapes / dtypes / value ranges
  4. Save debug visualizations for N samples

Usage:
    python scripts/04_debug_dataloader.py              # 3 samples
    python scripts/04_debug_dataloader.py --n 5        # 5 samples

Output:
    {OUTPUT_ROOT}/debug/dataloader/pg_sample_{i}.png
    {OUTPUT_ROOT}/debug/dataloader/sam_{target}_sample_{i}.png
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datn.config import DEBUG_DIR, SEED, ensure_dirs, seed_everything
from datn.datasets import PGDataset2p5D, SAMDataset
from datn.samplers import make_pg_sampler, make_tumor_oversampler
from datn.vis import fig_pg_sample, fig_sam_sample


def print_batch(name: str, batch: dict) -> None:
    """Print shape/dtype/range for every tensor in a batch dict."""
    print(f"\n  [{name}] Batch contents:")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"    {k:12s}  shape={str(list(v.shape)):18s}  "
                  f"dtype={str(v.dtype):15s}  "
                  f"range=[{v.min().item():.3f}, {v.max().item():.3f}]")
        elif isinstance(v, (list, tuple)):
            print(f"    {k:12s}  len={len(v)}  sample={v[0]}")
        else:
            print(f"    {k:12s}  type={type(v).__name__}  value={v}")


def main(n: int = 3, seed: int = SEED):
    t0 = time.time()
    ensure_dirs()
    seed_everything(seed)

    out_dir = DEBUG_DIR / "dataloader"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  PGDataset2p5D
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("=" * 60)
    print("PGDataset2p5D")
    print("=" * 60)

    pg_ds = PGDataset2p5D(split="train")
    print(f"  Dataset size: {len(pg_ds)}")

    pg_sampler = make_pg_sampler("train", pos_weight=3.0)
    pg_loader = DataLoader(pg_ds, batch_size=4, sampler=pg_sampler,
                           num_workers=0)

    batch = next(iter(pg_loader))
    print_batch("PG", batch)

    # Visualise N individual samples (not batched)
    rng = np.random.default_rng(seed)
    # Pick some positive and some negative
    pos_indices = [i for i, r in enumerate(pg_ds.rows) if r["has_tumor_wt"]]
    neg_indices = [i for i, r in enumerate(pg_ds.rows) if not r["has_tumor_wt"]]
    n_pos = min(n, len(pos_indices))
    n_neg = min(max(1, n // 3), len(neg_indices))  # ~1/3 negative

    sample_indices = list(rng.choice(pos_indices, size=n_pos, replace=False))
    sample_indices += list(rng.choice(neg_indices, size=n_neg, replace=False))

    print(f"\n  Saving {len(sample_indices)} PG samples ({n_pos} pos + {n_neg} neg):")
    for i, idx in enumerate(sample_indices):
        sample = pg_ds[idx]
        fig = fig_pg_sample(sample, idx=idx)
        path = out_dir / f"pg_sample_{i}.png"
        fig.savefig(str(path), dpi=100, bbox_inches="tight")
        plt.close(fig)
        obj = float(sample["objectness"])
        print(f"    [{i}] idx={idx}  obj={obj:.0f}  → {path.name}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  SAMDataset (WT, TC, ET)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    for target in ("WT", "TC", "ET"):
        print(f"\n{'=' * 60}")
        print(f"SAMDataset  target={target}")
        print("=" * 60)

        sam_ds = SAMDataset(split="train", target=target, jitter=True,
                            seed=seed)
        print(f"  Dataset size: {len(sam_ds)} ({target}-positive slices)")

        sam_loader = DataLoader(sam_ds, batch_size=2, shuffle=True,
                                num_workers=0)
        batch = next(iter(sam_loader))
        print_batch(f"SAM-{target}", batch)

        # Visualise N samples
        sample_indices = rng.choice(len(sam_ds), size=min(n, len(sam_ds)),
                                    replace=False)
        print(f"\n  Saving {len(sample_indices)} SAM-{target} samples:")
        for i, idx in enumerate(sample_indices):
            sample = sam_ds[int(idx)]
            fig = fig_sam_sample(sample, idx=int(idx))
            path = out_dir / f"sam_{target.lower()}_sample_{i}.png"
            fig.savefig(str(path), dpi=100, bbox_inches="tight")
            plt.close(fig)
            mask_area = float(sample["mask"].sum())
            print(f"    [{i}] idx={idx}  mask_area={mask_area:.0f}px  → {path.name}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Sampler distribution check
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print(f"\n{'=' * 60}")
    print("Sampler distribution check (1000 draws)")
    print("=" * 60)
    for target_tag in ("wt", "tc", "et"):
        sampler = make_tumor_oversampler("train", target=target_tag,
                                         pos_weight=3.0)
        draws = list(sampler)[:1000]
        # Check how many drawn indices correspond to positive slices
        key = f"has_tumor_{target_tag}"
        all_rows = pg_ds.rows
        n_drawn_pos = sum(1 for d in draws if all_rows[d].get(key, 0))
        print(f"  {target_tag.upper()}: {n_drawn_pos}/1000 positive "
              f"({n_drawn_pos/10:.1f}%)")

    elapsed = time.time() - t0
    print(f"\n[debug] Done in {elapsed:.1f}s.  Images → {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Debug dataloader smoke-test")
    p.add_argument("--n", type=int, default=3, help="samples per dataset")
    p.add_argument("--seed", type=int, default=42, help="RNG seed")
    main(**vars(p.parse_args()))
