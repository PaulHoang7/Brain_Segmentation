"""
Loss functions for PG and SAM+LoRA training.

PG losses:  focal (objectness), smoothl1 + giou (bbox), temporal (z-consistency)
SAM losses: dice + bce (or focal for ET imbalance)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Dice Loss ────────────────────────────────────────────────────
def dice_loss(pred: torch.Tensor, target: torch.Tensor,
              smooth: float = 1.0) -> torch.Tensor:
    """
    Soft Dice loss.
    pred:   (B, 1, H, W) sigmoid probabilities
    target: (B, 1, H, W) binary
    """
    pred = pred.flatten(1)
    target = target.flatten(1)
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return (1.0 - dice).mean()


# ── Focal Loss ───────────────────────────────────────────────────
def focal_loss(pred: torch.Tensor, target: torch.Tensor,
               gamma: float = 2.0, alpha: float = 0.25) -> torch.Tensor:
    """
    Binary focal loss for class imbalance.
    pred:   (B,) or (B,1) logits (pre-sigmoid)
    target: (B,) or (B,1) binary labels
    """
    pred = pred.view(-1)
    target = target.view(-1).float()
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    p_t = torch.exp(-bce)
    loss = alpha * (1 - p_t) ** gamma * bce
    return loss.mean()


# ── GIoU Loss ────────────────────────────────────────────────────
def giou_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Generalised IoU loss for bounding boxes.
    pred, target: (B, 4) as [x1, y1, x2, y2] normalised to [0,1].
    Returns: mean (1 - GIoU).
    """
    px1, py1, px2, py2 = pred.unbind(dim=-1)
    tx1, ty1, tx2, ty2 = target.unbind(dim=-1)

    # intersection
    ix1 = torch.max(px1, tx1)
    iy1 = torch.max(py1, ty1)
    ix2 = torch.min(px2, tx2)
    iy2 = torch.min(py2, ty2)
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)

    # union
    area_p = (px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0)
    area_t = (tx2 - tx1).clamp(min=0) * (ty2 - ty1).clamp(min=0)
    union = area_p + area_t - inter

    iou = inter / (union + 1e-7)

    # enclosing box
    ex1 = torch.min(px1, tx1)
    ey1 = torch.min(py1, ty1)
    ex2 = torch.max(px2, tx2)
    ey2 = torch.max(py2, ty2)
    area_e = (ex2 - ex1).clamp(min=0) * (ey2 - ey1).clamp(min=0)

    giou = iou - (area_e - union) / (area_e + 1e-7)
    return (1.0 - giou).mean()


# ── Temporal consistency loss ────────────────────────────────────
def temporal_bbox_loss(bbox_z: torch.Tensor,
                       bbox_z1: torch.Tensor) -> torch.Tensor:
    """
    SmoothL1 between predicted bbox at slice z and z+1.
    Only applied when both slices have tumor (caller filters).
    bbox_z, bbox_z1: (B, 4) normalised coords.
    """
    return F.smooth_l1_loss(bbox_z, bbox_z1)


# ── Composite PG loss ────────────────────────────────────────────
def pg_loss(obj_logit: torch.Tensor, obj_gt: torch.Tensor,
            bbox_pred: torch.Tensor, bbox_gt: torch.Tensor,
            has_tumor: torch.Tensor,
            gamma: float = 2.0,
            bbox_weight: float = 1.0,
            giou_weight: float = 1.0) -> dict[str, torch.Tensor]:
    """Combined PG loss. Returns dict of individual + total."""
    l_obj = focal_loss(obj_logit, obj_gt, gamma=gamma)

    # bbox losses only where has_tumor==1
    mask = has_tumor.bool()
    if mask.any():
        bp = bbox_pred[mask]
        bg = bbox_gt[mask]
        l_reg = F.smooth_l1_loss(bp, bg)
        l_giou = giou_loss(bp, bg)
    else:
        l_reg = torch.tensor(0.0, device=obj_logit.device)
        l_giou = torch.tensor(0.0, device=obj_logit.device)

    total = l_obj + bbox_weight * l_reg + giou_weight * l_giou
    return {"total": total, "obj": l_obj, "reg": l_reg, "giou": l_giou}


# ── Composite SAM loss ───────────────────────────────────────────
def sam_seg_loss(pred: torch.Tensor, target: torch.Tensor,
                 dice_w: float = 1.0, bce_w: float = 1.0) -> torch.Tensor:
    """Dice + BCE for SAM mask prediction."""
    l_dice = dice_loss(torch.sigmoid(pred), target)
    l_bce  = F.binary_cross_entropy_with_logits(pred, target)
    return dice_w * l_dice + bce_w * l_bce
