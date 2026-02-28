"""
3D post-processing for cascade predictions.

- Hierarchy enforcement: ET ⊂ TC ⊂ WT
- Connected-component removal (small islands)
- Bbox propagation along z (fill PG misses)
- EMA smoothing of bboxes along z
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from scipy.ndimage import label as ndlabel, binary_closing


# ── Hierarchy enforcement ────────────────────────────────────────
def enforce_hierarchy(wt: np.ndarray,
                      tc: np.ndarray,
                      et: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Enforce ET ⊂ TC ⊂ WT on binary 3-D masks.
    Operates in-place and returns the corrected masks.
    """
    et = et & tc          # ET must be inside TC
    tc = (tc | et) & wt   # TC (+ ET) must be inside WT
    wt = wt | tc          # WT must contain TC
    return wt.astype(np.uint8), tc.astype(np.uint8), et.astype(np.uint8)


# ── Connected-component cleanup ─────────────────────────────────
def remove_small_cc(mask_3d: np.ndarray, min_size: int = 50) -> np.ndarray:
    """Remove connected components smaller than min_size voxels."""
    labeled, n_cc = ndlabel(mask_3d.astype(bool))
    if n_cc == 0:
        return mask_3d
    out = np.zeros_like(mask_3d)
    for i in range(1, n_cc + 1):
        cc = labeled == i
        if cc.sum() >= min_size:
            out[cc] = 1
    return out


# ── Light morphological closing ──────────────────────────────────
def closing_3d(mask_3d: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Binary closing to fill small holes."""
    return binary_closing(mask_3d, iterations=iterations).astype(np.uint8)


# ── Bbox EMA smoothing along z ───────────────────────────────────
def smooth_boxes_ema(boxes: List[Optional[Tuple[int, int, int, int]]],
                     alpha: float = 0.8
                     ) -> List[Optional[Tuple[int, int, int, int]]]:
    """
    Smooth predicted bboxes along the z axis using exponential moving average.
    boxes: length-D list, None for slices with no predicted tumor.
    alpha: weight for current frame (higher = less smoothing).
    """
    smoothed: List[Optional[Tuple[int, int, int, int]]] = [None] * len(boxes)
    prev = None

    for z in range(len(boxes)):
        if boxes[z] is None:
            smoothed[z] = None
            continue
        cur = np.array(boxes[z], dtype=np.float64)
        if prev is None:
            smoothed[z] = boxes[z]
        else:
            s = alpha * cur + (1 - alpha) * prev
            smoothed[z] = tuple(int(round(v)) for v in s)
        prev = np.array(smoothed[z], dtype=np.float64)

    return smoothed


# ── Bbox propagation (fill gaps) ────────────────────────────────
def propagate_bbox(boxes: List[Optional[Tuple[int, int, int, int]]],
                   objectness: List[float],
                   obj_threshold: float = 0.3
                   ) -> List[Optional[Tuple[int, int, int, int]]]:
    """
    If PG misses a slice (box=None) but neighbours have boxes,
    interpolate from nearest positive slice.
    Only propagates if objectness is above obj_threshold.
    """
    D = len(boxes)
    result = list(boxes)

    for z in range(D):
        if result[z] is not None:
            continue
        if objectness[z] < obj_threshold:
            continue

        # Find nearest non-None neighbours
        left = right = None
        for dz in range(1, D):
            if z - dz >= 0 and result[z - dz] is not None:
                left = result[z - dz]
                break
        for dz in range(1, D):
            if z + dz < D and result[z + dz] is not None:
                right = result[z + dz]
                break

        if left is not None and right is not None:
            # simple mean interpolation
            result[z] = tuple(int((l + r) / 2) for l, r in zip(left, right))
        elif left is not None:
            result[z] = left
        elif right is not None:
            result[z] = right

    return result
