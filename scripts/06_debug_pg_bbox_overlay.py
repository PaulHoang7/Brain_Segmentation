#!/usr/bin/env python3
"""
[06-debug] PG Bbox Overlay — draw GT and predicted bboxes on input slices.

Samples 20 GT-positive slices, runs PG inference, and saves overlay images
showing GT bbox (green) and predicted bbox (red).

Usage:
    python scripts/06_debug_pg_bbox_overlay.py
    python scripts/06_debug_pg_bbox_overlay.py --n 30 --ckpt /path/to/model.pth

Output:
    {OUTPUT_ROOT}/debug/pg_bbox_overlay/overlay_{i}.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datn.config import (
    CKPT_DIR, DEBUG_DIR, SEED,
    ensure_dirs, load_hparams, seed_everything,
)
from datn.datasets import PGDataset2p5D
from datn.metrics import _single_iou
from datn.pg_model import PromptGenerator


def draw_bbox_rect(ax, bbox_norm, img_size, color, label, linewidth=2):
    """Draw a [x1,y1,x2,y2] normalised bbox on axes."""
    x1, y1, x2, y2 = bbox_norm
    # Convert normalised → pixel coords in img_size space
    px1, py1 = x1 * img_size, y1 * img_size
    px2, py2 = x2 * img_size, y2 * img_size
    w, h = px2 - px1, py2 - py1
    rect = patches.Rectangle(
        (px1, py1), w, h,
        linewidth=linewidth, edgecolor=color, facecolor="none",
        linestyle="-" if "GT" in label else "--",
        label=label,
    )
    ax.add_patch(rect)


def main(n: int = 20, ckpt: str = None, seed: int = SEED):
    ensure_dirs()
    seed_everything(seed)

    hp = load_hparams().get("pg", {})
    img_size = hp.get("img_size", 224)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    ckpt_path = Path(ckpt) if ckpt else CKPT_DIR / "pg" / "best.pth"
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found at {ckpt_path}")
        sys.exit(1)

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = PromptGenerator(
        in_channels=hp.get("input_channels", 9)).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    print(f"[overlay] Loaded: {ckpt_path}  (epoch {state.get('epoch', '?')})")

    # Dataset
    ds = PGDataset2p5D(split="val", img_size=img_size)

    # Find positive slices
    pos_indices = [i for i, r in enumerate(ds.rows) if r["has_tumor_wt"]]
    rng = np.random.default_rng(seed)
    chosen = rng.choice(pos_indices, size=min(n, len(pos_indices)), replace=False)

    out_dir = DEBUG_DIR / "pg_bbox_overlay"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[overlay] Sampling {len(chosen)} GT-positive slices from val split")
    print(f"[overlay] Output → {out_dir}")

    ious = []
    for i, idx in enumerate(chosen):
        sample = ds[int(idx)]
        img_t = sample["image"].unsqueeze(0).to(device)       # (1, 9, H, W)
        bbox_gt = sample["bbox"].numpy()                       # (4,)
        obj_gt = float(sample["objectness"])

        with torch.no_grad():
            out = model(img_t)
            obj_prob = torch.sigmoid(out["objectness"]).item()
            bbox_pred = out["bbox"][0].cpu().numpy()            # (4,)

        iou = _single_iou(bbox_pred, bbox_gt)
        ious.append(iou)

        # Visualise: middle channel of first modality (index 1 = z-center)
        vis_img = sample["image"][1].numpy()  # (224, 224), z-center of first mod

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(vis_img, cmap="gray", vmin=vis_img.min(), vmax=vis_img.max())
        draw_bbox_rect(ax, bbox_gt, img_size, "lime", "GT bbox")
        draw_bbox_rect(ax, bbox_pred, img_size, "red", "Pred bbox")

        row = ds.rows[int(idx)]
        ax.set_title(
            f"{row['case_id']}  z={row['z']}\n"
            f"obj: GT={obj_gt:.0f} pred={obj_prob:.3f}  "
            f"IoU={iou:.3f}",
            fontsize=10,
        )
        ax.legend(loc="upper right", fontsize=8)
        ax.set_xlim(0, img_size)
        ax.set_ylim(img_size, 0)

        path = out_dir / f"overlay_{i:02d}.png"
        fig.savefig(str(path), dpi=120, bbox_inches="tight")
        plt.close(fig)

        bp_str = ",".join(f"{v:.3f}" for v in bbox_pred)
        bg_str = ",".join(f"{v:.3f}" for v in bbox_gt)
        print(f"  [{i:2d}] {row['case_id']} z={row['z']:3d}  "
              f"obj={obj_prob:.3f}  IoU={iou:.3f}  "
              f"pred=[{bp_str}]  GT=[{bg_str}]")

    mean_iou = np.mean(ious) if ious else 0.0
    print(f"\n[overlay] Mean IoU over {len(ious)} samples: {mean_iou:.4f}")
    print(f"[overlay] Images → {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="PG bbox overlay debug")
    p.add_argument("--n", type=int, default=20, help="Number of samples")
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--seed", type=int, default=SEED)
    main(**vars(p.parse_args()))
