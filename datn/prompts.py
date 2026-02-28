"""Bounding-box prompt utilities: tight bbox, padding, jitter, point sampling."""
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np


def tight_bbox(mask_2d: np.ndarray,
               pad: int = 0) -> Optional[Tuple[int, int, int, int]]:
    """
    Compute tight bbox (x1, y1, x2, y2) for a binary 2-D mask.
    Returns None if mask is all-zero.
    pad: extra pixels added on each side (clamped to image bounds).
    """
    ys, xs = np.where(mask_2d > 0)
    if len(xs) == 0:
        return None
    H, W = mask_2d.shape
    x1 = max(int(xs.min()) - pad, 0)
    y1 = max(int(ys.min()) - pad, 0)
    x2 = min(int(xs.max()) + pad, W - 1)
    y2 = min(int(ys.max()) + pad, H - 1)
    return (x1, y1, x2, y2)


def jitter_bbox(bbox: Tuple[int, int, int, int],
                img_h: int, img_w: int,
                shift_ratio: float = 0.1,
                scale_ratio: float = 0.1,
                rng: Optional[np.random.Generator] = None
                ) -> Tuple[int, int, int, int]:
    """
    Randomly shift + scale a bbox for SAM training robustness.
    shift_ratio / scale_ratio are relative to bbox size.
    """
    if rng is None:
        rng = np.random.default_rng()

    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1

    # random shift
    dx = rng.uniform(-shift_ratio, shift_ratio) * w
    dy = rng.uniform(-shift_ratio, shift_ratio) * h

    # random scale
    sw = rng.uniform(-scale_ratio, scale_ratio) * w
    sh = rng.uniform(-scale_ratio, scale_ratio) * h

    nx1 = int(np.clip(x1 + dx - sw / 2, 0, img_w - 1))
    ny1 = int(np.clip(y1 + dy - sh / 2, 0, img_h - 1))
    nx2 = int(np.clip(x2 + dx + sw / 2, 0, img_w - 1))
    ny2 = int(np.clip(y2 + dy + sh / 2, 0, img_h - 1))

    # ensure valid box (at least 1px)
    if nx2 <= nx1:
        nx2 = min(nx1 + 1, img_w - 1)
    if ny2 <= ny1:
        ny2 = min(ny1 + 1, img_h - 1)
    return (nx1, ny1, nx2, ny2)


def bbox_area(bbox: Tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def bbox_iou(a: Tuple[int, int, int, int],
             b: Tuple[int, int, int, int]) -> float:
    """IoU between two (x1,y1,x2,y2) boxes."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = bbox_area(a)
    area_b = bbox_area(b)
    union = area_a + area_b - inter
    if union == 0:
        return 0.0
    return inter / union


def sample_point_in_mask(mask_2d: np.ndarray,
                         rng: Optional[np.random.Generator] = None
                         ) -> Optional[Tuple[int, int]]:
    """Sample a random (x, y) point inside the foreground mask."""
    if rng is None:
        rng = np.random.default_rng()
    ys, xs = np.where(mask_2d > 0)
    if len(xs) == 0:
        return None
    idx = rng.integers(len(xs))
    return (int(xs[idx]), int(ys[idx]))
