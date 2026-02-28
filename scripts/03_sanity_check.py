#!/usr/bin/env python3
"""
[03] SAM Preprocess Sanity Check.

For N random tumour slices from train.jsonl:
  1. Load preprocessed .npy volumes (t2f, t1c, t2w) + seg
  2. Show original 240×240 with mask overlays + GT bboxes (WT/TC/ET)
  3. Apply SAM resize-longest-side(1024) + pad
  4. Transform bboxes to SAM space
  5. Show SAM 1024×1024 with transformed bboxes + resized seg overlay
  6. Print numerical verification (scale factor, bbox coords)

Usage:
    python scripts/03_sanity_check.py              # 3 samples
    python scripts/03_sanity_check.py --n 5        # 5 samples

Output:
    {OUTPUT_ROOT}/debug/sam_preprocess/sample_{i}_{case_id}_z{z}.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datn.config import (
    DEBUG_DIR,
    INDEX_DIR,
    LABEL_MAP_JSON,
    MODALITIES,
    SAM_IMG_SIZE,
    ensure_dirs,
)
from datn.sam_preprocess import (
    get_preprocess_shape,
    preprocess_image_for_sam,
    resize_mask,
    transform_bbox,
)
from datn.vis import fig_sam_preprocess_sanity


# ── Helpers ──────────────────────────────────────────────────────
def load_slice_from_paths(paths: dict, z: int) -> dict[str, np.ndarray]:
    """Load a single z-slice from preprocessed .npy volumes + seg."""
    slices = {}
    for mod in MODALITIES:
        vol = np.load(paths[mod])           # (H, W, D)
        slices[mod] = vol[:, :, z]
    seg_vol = nib.load(paths["seg"])
    slices["seg"] = np.asarray(seg_vol.dataobj, dtype=np.int16)[:, :, z]
    return slices


def make_3ch(slices: dict) -> np.ndarray:
    """Stack 3 modalities into (H, W, 3) float32."""
    return np.stack([slices[m] for m in MODALITIES], axis=-1)


def masks_from_seg(seg_2d: np.ndarray, lmap: dict) -> dict[str, np.ndarray]:
    """Compute WT/TC/ET binary masks from seg slice."""
    return {
        tag: np.isin(seg_2d, lmap[tag]).astype(np.uint8)
        for tag in ("WT", "TC", "ET")
    }


# ── Main ─────────────────────────────────────────────────────────
def main(n: int = 3, seed: int = 42):
    ensure_dirs()

    out_dir = DEBUG_DIR / "sam_preprocess"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(LABEL_MAP_JSON) as f:
        lmap = json.load(f)

    # Load train index, filter tumour slices
    train_path = INDEX_DIR / "train.jsonl"
    if not train_path.exists():
        print(f"ERROR: {train_path} not found. Run 02_make_index.sh first.")
        sys.exit(1)

    tumor_rows = []
    with open(train_path) as f:
        for line in f:
            row = json.loads(line)
            if row["has_tumor_wt"]:
                tumor_rows.append(row)

    print(f"[sanity] Train tumour slices: {len(tumor_rows)}")
    print(f"[sanity] Sampling {n} for visualization")
    print(f"[sanity] SAM target size: {SAM_IMG_SIZE}")
    print()

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(tumor_rows), size=min(n, len(tumor_rows)),
                         replace=False)

    for i, idx in enumerate(indices):
        row = tumor_rows[idx]
        cid, z = row["case_id"], row["z"]
        H, W = row["img_shape"]
        paths = row["paths"]

        print(f"--- Sample {i} : {cid}  z={z}  shape=({H},{W}) ---")

        # Load preprocessed slices
        slices = load_slice_from_paths(paths, z)
        seg_orig = slices["seg"]
        img3_orig = make_3ch(slices)

        # Original bboxes from index
        bboxes_orig = {}
        for tag in ("WT", "TC", "ET"):
            key = f"bbox_gt_{tag.lower()}"
            b = row.get(key)
            bboxes_orig[tag] = tuple(b) if b else None

        # Binary masks
        masks_orig = masks_from_seg(seg_orig, lmap)

        # ── SAM preprocess ───────────────────────────────────────
        img3_sam, orig_h, orig_w = preprocess_image_for_sam(img3_orig)
        seg_sam = resize_mask(seg_orig, SAM_IMG_SIZE)

        # Transform bboxes
        bboxes_sam = {}
        for tag in ("WT", "TC", "ET"):
            if bboxes_orig[tag] is not None:
                bboxes_sam[tag] = transform_bbox(
                    bboxes_orig[tag], orig_h, orig_w, SAM_IMG_SIZE)
            else:
                bboxes_sam[tag] = None

        # ── Numerical verification ───────────────────────────────
        newh, neww = get_preprocess_shape(H, W, SAM_IMG_SIZE)
        scale = SAM_IMG_SIZE / max(H, W)
        print(f"  resize: ({H},{W}) → ({newh},{neww})  scale={scale:.4f}")
        print(f"  padded: ({SAM_IMG_SIZE},{SAM_IMG_SIZE})")
        for tag in ("WT", "TC", "ET"):
            bo = bboxes_orig[tag]
            bs = bboxes_sam[tag]
            if bo and bs:
                print(f"  bbox_{tag}: orig={bo} → sam={bs}")
                # Verify: transformed coords ≈ orig * scale
                expected = tuple(int(c * scale) for c in bo)
                print(f"    expected(orig*scale)={expected}  "
                      f"actual={bs}  match={expected == bs}")

        print(f"  seg_sam labels: {np.unique(seg_sam).tolist()}")

        # ── Draw ─────────────────────────────────────────────────
        fig = fig_sam_preprocess_sanity(
            slice_orig=slices[MODALITIES[0]],
            img3_orig=img3_orig,
            seg_orig=seg_orig,
            img3_sam=img3_sam,
            seg_sam=seg_sam,
            bboxes_orig=bboxes_orig,
            bboxes_sam=bboxes_sam,
            masks_orig=masks_orig,
            title=f"{cid}  z={z}  ({H}×{W} → {SAM_IMG_SIZE}×{SAM_IMG_SIZE})",
        )

        out_path = out_dir / f"sample_{i}_{cid}_z{z}.png"
        fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
        import matplotlib.pyplot as plt
        plt.close(fig)
        print(f"  Saved → {out_path}")
        print()

    print(f"[sanity] All {n} samples saved to {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="SAM preprocess sanity check")
    p.add_argument("--n", type=int, default=3, help="samples to visualise")
    p.add_argument("--seed", type=int, default=42, help="RNG seed")
    main(**vars(p.parse_args()))
