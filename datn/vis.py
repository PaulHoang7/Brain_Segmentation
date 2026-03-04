"""
Visualization utilities for debug / sanity-check plots.

All functions return matplotlib Figure objects (caller saves).
Uses Agg backend — no display needed.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


# ── Colour definitions ───────────────────────────────────────────
BBOX_COLORS = {"WT": "#00ff00", "TC": "#ffff00", "ET": "#ff3333"}
MASK_CMAPS  = {"WT": "Greens", "TC": "Oranges", "ET": "Reds"}
LABEL_COLORS = {0: [0, 0, 0], 1: [255, 0, 0], 2: [0, 255, 0], 3: [0, 0, 255]}


# ── Low-level drawing ────────────────────────────────────────────
def draw_bbox(ax: plt.Axes, bbox: Sequence[int], color: str,
              label: str = "", linewidth: float = 2.0) -> None:
    """Draw (x1, y1, x2, y2) rectangle on axis."""
    x1, y1, x2, y2 = bbox
    rect = patches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        linewidth=linewidth, edgecolor=color, facecolor="none",
    )
    ax.add_patch(rect)
    if label:
        ax.text(x1, max(y1 - 3, 0), label,
                fontsize=8, color=color, weight="bold",
                va="bottom", ha="left")


def overlay_mask(ax: plt.Axes, mask: np.ndarray,
                 cmap: str = "Reds", alpha: float = 0.35) -> None:
    """Overlay a binary mask on axis with transparency."""
    masked = np.ma.masked_where(mask == 0, mask.astype(np.float32))
    ax.imshow(masked, cmap=cmap, alpha=alpha, interpolation="nearest")


def norm_for_display(img: np.ndarray) -> np.ndarray:
    """Normalise a float image to [0,1] per-channel for display."""
    out = img.copy().astype(np.float32)
    if out.ndim == 2:
        mn, mx = out.min(), out.max()
        return (out - mn) / (mx - mn + 1e-8)
    for c in range(out.shape[-1]):
        ch = out[..., c]
        mn, mx = ch.min(), ch.max()
        out[..., c] = (ch - mn) / (mx - mn + 1e-8)
    return out


def seg_to_rgb(seg: np.ndarray) -> np.ndarray:
    """Convert integer seg labels to RGB (H,W,3) uint8."""
    H, W = seg.shape[:2]
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for lab, col in LABEL_COLORS.items():
        rgb[seg == lab] = col
    return rgb


# ── High-level figure builders ───────────────────────────────────
def fig_sam_preprocess_sanity(
    *,
    slice_orig: np.ndarray,           # (H, W) float32  — single modality
    img3_orig: np.ndarray,            # (H, W, 3) float32  — 3-mod composite
    seg_orig: np.ndarray,             # (H, W) int
    img3_sam: np.ndarray,             # (1024,1024,3) after resize+pad
    seg_sam: np.ndarray,              # (1024,1024) after resize+pad
    bboxes_orig: dict[str, Optional[Tuple[int, int, int, int]]],
    bboxes_sam: dict[str, Optional[Tuple[int, int, int, int]]],
    masks_orig: dict[str, np.ndarray],
    title: str = "",
) -> plt.Figure:
    """
    2-row × 3-col sanity figure.

    Row 1 (original 240×240):
        [0] single-mod grayscale
        [1] 3-mod composite + mask overlays + bboxes
        [2] seg label map (coloured)

    Row 2 (SAM 1024×1024):
        [3] 3-mod after resize+pad
        [4] same + bboxes transformed
        [5] seg after resize+pad (coloured)
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    if title:
        fig.suptitle(title, fontsize=14, y=0.98)

    # ── Row 1: original ──────────────────────────────────────────
    ax = axes[0, 0]
    ax.imshow(slice_orig, cmap="gray")
    ax.set_title("Original (t2f)")

    ax = axes[0, 1]
    ax.imshow(norm_for_display(img3_orig))
    for tag in ("WT", "TC", "ET"):
        if tag in masks_orig and masks_orig[tag] is not None:
            overlay_mask(ax, masks_orig[tag], cmap=MASK_CMAPS[tag], alpha=0.25)
        if tag in bboxes_orig and bboxes_orig[tag] is not None:
            draw_bbox(ax, bboxes_orig[tag], BBOX_COLORS[tag], label=tag)
    ax.set_title("Original + masks + bboxes")

    ax = axes[0, 2]
    ax.imshow(seg_to_rgb(seg_orig))
    ax.set_title("Seg labels (R=NCR G=ED B=ET)")

    # ── Row 2: SAM 1024 ─────────────────────────────────────────
    ax = axes[1, 0]
    ax.imshow(norm_for_display(img3_sam))
    ax.set_title("SAM 1024 (resize+pad)")

    ax = axes[1, 1]
    ax.imshow(norm_for_display(img3_sam))
    for tag in ("WT", "TC", "ET"):
        if tag in bboxes_sam and bboxes_sam[tag] is not None:
            draw_bbox(ax, bboxes_sam[tag], BBOX_COLORS[tag], label=tag)
    # Also overlay resized seg as contour check
    seg_wt = (seg_sam > 0).astype(np.float32)
    overlay_mask(ax, seg_wt, cmap="Greens", alpha=0.2)
    ax.set_title("SAM 1024 + bboxes transformed")

    ax = axes[1, 2]
    ax.imshow(seg_to_rgb(seg_sam))
    ax.set_title("Seg 1024 (resize+pad)")

    for ax in axes.flat:
        ax.axis("off")

    fig.tight_layout()
    return fig


# ── Dataloader debug figures ──────────────────────────────────────
def fig_pg_sample(sample: dict, idx: int = 0) -> plt.Figure:
    """
    Visualise one PGDataset2p5D sample.

    Shows 9 channels arranged as 3 rows (mods) × 3 cols (z-1, z, z+1),
    plus objectness/bbox annotation.
    """
    img = sample["image"]  # (9, H, W) tensor
    if hasattr(img, "numpy"):
        img = img.numpy()
    obj = float(sample["objectness"])
    bbox = sample["bbox"]
    if hasattr(bbox, "numpy"):
        bbox = bbox.numpy()
    cid = sample.get("case_id", "?")
    z = sample.get("z", "?")

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle(f"PG sample #{idx}  {cid} z={z}  obj={obj:.0f}  "
                 f"bbox=[{bbox[0]:.2f},{bbox[1]:.2f},{bbox[2]:.2f},{bbox[3]:.2f}]",
                 fontsize=12)

    mod_names = ["t2f", "t1c", "t2w"]
    slice_names = ["z-1", "z", "z+1"]
    ch = 0
    for row in range(3):
        for col in range(3):
            ax = axes[row, col]
            ax.imshow(img[ch], cmap="gray")
            ax.set_title(f"{mod_names[row]} ({slice_names[col]})", fontsize=9)
            # Draw bbox on centre slice (col=1)
            if col == 1 and obj > 0.5:
                H, W = img[ch].shape
                x1, y1 = bbox[0] * W, bbox[1] * H
                x2, y2 = bbox[2] * W, bbox[3] * H
                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor="lime", facecolor="none")
                ax.add_patch(rect)
            ax.axis("off")
            ch += 1

    fig.tight_layout()
    return fig


def fig_sam_sample(sample: dict, idx: int = 0) -> plt.Figure:
    """
    Visualise one SAMDataset sample.

    Col 1: SAM image (1024×1024, 3-ch composite)
    Col 2: same + bbox prompt overlay
    Col 3: target mask (256×256)
    """
    img = sample["image"]    # (3, 1024, 1024)
    mask = sample["mask"]    # (1, 256, 256)
    bbox = sample["bbox"]    # (4,)
    if hasattr(img, "numpy"):
        img = img.numpy()
    if hasattr(mask, "numpy"):
        mask = mask.numpy()
    if hasattr(bbox, "numpy"):
        bbox = bbox.numpy()
    cid = sample.get("case_id", "?")
    z = sample.get("z", "?")

    # (3,H,W) → (H,W,3)
    disp = norm_for_display(img.transpose(1, 2, 0))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"SAM sample #{idx}  {cid} z={z}  "
                 f"bbox=[{bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f}]",
                 fontsize=12)

    axes[0].imshow(disp)
    axes[0].set_title("SAM image (1024)")

    axes[1].imshow(disp)
    x1, y1, x2, y2 = bbox
    rect = patches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        linewidth=2, edgecolor="lime", facecolor="none")
    axes[1].add_patch(rect)
    axes[1].set_title("+ bbox prompt")

    axes[2].imshow(mask[0], cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(f"Target mask (256)  area={mask.sum():.0f}px")

    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    return fig
