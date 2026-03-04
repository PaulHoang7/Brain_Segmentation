"""
Evaluation metrics for segmentation and prompt generation.

Segmentation: Dice coefficient, Hausdorff distance 95th percentile.
PG: detection P/R/F1, bbox IoU, temporal stability.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import label as ndlabel
from scipy.spatial.distance import directed_hausdorff


# ── Dice ─────────────────────────────────────────────────────────
def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Binary Dice coefficient. Returns 1.0 if both are empty."""
    pred = pred.astype(bool)
    gt   = gt.astype(bool)
    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0
    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0
    intersection = (pred & gt).sum()
    return 2.0 * intersection / (pred.sum() + gt.sum())


# ── Hausdorff 95 ────────────────────────────────────────────────
def hausdorff_95(pred: np.ndarray, gt: np.ndarray,
                 spacing: Tuple[float, ...] = (1.0, 1.0, 1.0)) -> float:
    """
    95th percentile Hausdorff distance.
    pred, gt: binary 3D arrays.
    spacing:  voxel spacing (mm).
    Returns inf if one mask is empty and the other is not.
    """
    pred = pred.astype(bool)
    gt   = gt.astype(bool)
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return float("inf")

    pred_pts = np.argwhere(pred) * np.array(spacing)
    gt_pts   = np.argwhere(gt) * np.array(spacing)

    # directed distances
    from scipy.spatial import cKDTree
    tree_gt   = cKDTree(gt_pts)
    tree_pred = cKDTree(pred_pts)
    d_p2g, _ = tree_gt.query(pred_pts)
    d_g2p, _ = tree_pred.query(gt_pts)

    return max(np.percentile(d_p2g, 95), np.percentile(d_g2p, 95))


# ── Per-case segmentation metrics ────────────────────────────────
def compute_seg_metrics(pred_3d: np.ndarray, gt_3d: np.ndarray,
                        label_map: dict,
                        spacing: Tuple[float, ...] = (1.0, 1.0, 1.0)
                        ) -> Dict[str, Dict[str, float]]:
    """
    Compute Dice + HD95 for WT, TC, ET given raw seg volumes.
    Returns: {"WT": {"dice": .., "hd95": ..}, "TC": ..., "ET": ...}
    """
    results: Dict[str, Dict[str, float]] = {}
    for region in ("WT", "TC", "ET"):
        labels = label_map[region]
        p = np.isin(pred_3d, labels)
        g = np.isin(gt_3d, labels)
        results[region] = {
            "dice": dice_score(p, g),
            "hd95": hausdorff_95(p, g, spacing),
        }
    return results


# ── PG detection metrics ────────────────────────────────────────
def pg_detection_metrics(pred_obj: np.ndarray,
                         gt_obj: np.ndarray,
                         threshold: float = 0.5
                         ) -> Dict[str, float]:
    """
    Precision / Recall / F1 for per-slice tumor detection.
    pred_obj: (N,) predicted objectness probabilities.
    gt_obj:   (N,) binary ground truth.
    """
    pred_bin = (pred_obj >= threshold).astype(int)
    gt_bin   = gt_obj.astype(int)

    tp = ((pred_bin == 1) & (gt_bin == 1)).sum()
    fp = ((pred_bin == 1) & (gt_bin == 0)).sum()
    fn = ((pred_bin == 0) & (gt_bin == 1)).sum()

    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    return {"precision": float(prec), "recall": float(rec), "f1": float(f1)}


# ── PG bbox IoU ──────────────────────────────────────────────────
def _single_iou(pb: np.ndarray, gb: np.ndarray) -> float:
    """IoU between two [x1, y1, x2, y2] boxes (any coord system)."""
    ix1 = max(pb[0], gb[0])
    iy1 = max(pb[1], gb[1])
    ix2 = min(pb[2], gb[2])
    iy2 = min(pb[3], gb[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a1 = max(0, pb[2] - pb[0]) * max(0, pb[3] - pb[1])
    a2 = max(0, gb[2] - gb[0]) * max(0, gb[3] - gb[1])
    union = a1 + a2 - inter
    return inter / (union + 1e-9)


def pg_bbox_iou(pred_boxes: np.ndarray,
                gt_boxes: np.ndarray) -> float:
    """Mean IoU over paired boxes. boxes: (N, 4) [x1,y1,x2,y2] normalised."""
    assert len(pred_boxes) == len(gt_boxes)
    if len(pred_boxes) == 0:
        return 0.0
    ious = [_single_iou(pb, gb) for pb, gb in zip(pred_boxes, gt_boxes)]
    return float(np.mean(ious))


# ── PG stability ────────────────────────────────────────────────
def pg_stability(boxes_per_z: np.ndarray,
                 has_tumor: np.ndarray) -> Dict[str, float]:
    """
    Compute mean delta-center and delta-area between consecutive slices.
    boxes_per_z: (D, 4) predicted boxes for one case.
    has_tumor:   (D,) binary.
    """
    centers, areas = [], []
    for z in range(len(boxes_per_z)):
        if has_tumor[z]:
            b = boxes_per_z[z]
            centers.append(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2))
            areas.append((b[2] - b[0]) * (b[3] - b[1]))
        else:
            centers.append(None)
            areas.append(None)

    d_center, d_area = [], []
    for z in range(len(centers) - 1):
        if centers[z] is not None and centers[z + 1] is not None:
            dc = np.sqrt((centers[z][0] - centers[z+1][0])**2 +
                         (centers[z][1] - centers[z+1][1])**2)
            da = abs(areas[z] - areas[z+1])
            d_center.append(dc)
            d_area.append(da)

    return {
        "mean_delta_center": float(np.mean(d_center)) if d_center else 0.0,
        "mean_delta_area":   float(np.mean(d_area))   if d_area   else 0.0,
    }
