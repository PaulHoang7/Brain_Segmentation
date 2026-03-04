"""
EMA smoothing along the z-axis for PG bbox predictions.

Given per-slice predictions for one case (ordered by z),
applies exponential moving average for temporal consistency.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def ema_smooth_boxes(
    boxes: np.ndarray,
    has_tumor: np.ndarray,
    alpha: float = 0.8,
) -> np.ndarray:
    """
    EMA-smooth bbox predictions along z for one case.

    Args:
        boxes:      (D, 4) predicted bboxes [x1, y1, x2, y2], normalised [0,1].
        has_tumor:  (D,) binary — 1 if slice is predicted positive.
        alpha:      smoothing factor.  output = alpha * current + (1-alpha) * prev.
                    Higher alpha → more weight on current slice.

    Returns:
        (D, 4) smoothed boxes.  Non-tumour slices are unchanged (zeros).
    """
    D = len(boxes)
    smoothed = boxes.copy().astype(np.float64)

    # Forward pass
    for z in range(1, D):
        if has_tumor[z] and has_tumor[z - 1]:
            smoothed[z] = alpha * smoothed[z] + (1 - alpha) * smoothed[z - 1]

    # Backward pass (bidirectional EMA)
    backward = boxes.copy().astype(np.float64)
    for z in range(D - 2, -1, -1):
        if has_tumor[z] and has_tumor[z + 1]:
            backward[z] = alpha * backward[z] + (1 - alpha) * backward[z + 1]

    # Average forward and backward
    for z in range(D):
        if has_tumor[z]:
            smoothed[z] = 0.5 * (smoothed[z] + backward[z])

    return smoothed.astype(np.float32)


def ema_smooth_objectness(
    obj_probs: np.ndarray,
    alpha: float = 0.8,
) -> np.ndarray:
    """
    EMA-smooth objectness probabilities along z for one case.

    Args:
        obj_probs: (D,) predicted objectness probabilities [0,1].
        alpha:     smoothing factor.

    Returns:
        (D,) smoothed probabilities.
    """
    D = len(obj_probs)

    # Forward
    fwd = obj_probs.copy().astype(np.float64)
    for z in range(1, D):
        fwd[z] = alpha * fwd[z] + (1 - alpha) * fwd[z - 1]

    # Backward
    bwd = obj_probs.copy().astype(np.float64)
    for z in range(D - 2, -1, -1):
        bwd[z] = alpha * bwd[z] + (1 - alpha) * bwd[z + 1]

    return (0.5 * (fwd + bwd)).astype(np.float32)
