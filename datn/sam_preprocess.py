"""
SAM-compatible preprocessing:
  - ResizeLongestSide(1024) keeping aspect ratio
  - Pad to 1024×1024
  - Coordinate transforms for bboxes / points
"""
from __future__ import annotations
from typing import Tuple, Optional
import numpy as np

from .config import SAM_IMG_SIZE


def get_preprocess_shape(oldh: int, oldw: int,
                         long_side: int = SAM_IMG_SIZE
                         ) -> Tuple[int, int]:
    """Compute new (h, w) after resizing so longest side == long_side."""
    scale = long_side / max(oldh, oldw)
    newh = int(oldh * scale + 0.5)
    neww = int(oldw * scale + 0.5)
    return (newh, neww)


def resize_longest_side(image: np.ndarray,
                        long_side: int = SAM_IMG_SIZE) -> np.ndarray:
    """
    Resize image so longest side == long_side.
    image: (H, W) or (H, W, C), float32.
    Uses bilinear interpolation via numpy (no cv2 dependency).
    """
    import torch
    import torch.nn.functional as F

    if image.ndim == 2:
        image = image[..., None]
    H, W, C = image.shape
    newh, neww = get_preprocess_shape(H, W, long_side)

    # Use torch interpolate for quality
    t = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
    t = F.interpolate(t, size=(newh, neww), mode="bilinear",
                      align_corners=False)
    out = t.squeeze(0).permute(1, 2, 0).numpy()
    if C == 1:
        out = out[..., 0]
    return out


def pad_to_square(image: np.ndarray,
                  size: int = SAM_IMG_SIZE) -> np.ndarray:
    """
    Pad image to (size, size) with zeros (bottom-right padding).
    image: (H, W) or (H, W, C).
    """
    if image.ndim == 2:
        h, w = image.shape
        out = np.zeros((size, size), dtype=image.dtype)
        out[:h, :w] = image
    else:
        h, w, c = image.shape
        out = np.zeros((size, size, c), dtype=image.dtype)
        out[:h, :w, :] = image
    return out


def transform_coords(coords: np.ndarray,
                     orig_h: int, orig_w: int,
                     long_side: int = SAM_IMG_SIZE) -> np.ndarray:
    """
    Transform (x, y) coordinates from original image space
    to resized+padded space.
    coords: (..., 2) array of (x, y) pairs.
    """
    newh, neww = get_preprocess_shape(orig_h, orig_w, long_side)
    scale = long_side / max(orig_h, orig_w)
    return (coords.astype(np.float64) * scale).astype(np.float64)


def transform_bbox(bbox: Tuple[int, int, int, int],
                   orig_h: int, orig_w: int,
                   long_side: int = SAM_IMG_SIZE
                   ) -> Tuple[int, int, int, int]:
    """Transform (x1, y1, x2, y2) bbox to resized+padded space."""
    coords = np.array([[bbox[0], bbox[1]],
                       [bbox[2], bbox[3]]], dtype=np.float64)
    tc = transform_coords(coords, orig_h, orig_w, long_side)
    return (int(tc[0, 0]), int(tc[0, 1]), int(tc[1, 0]), int(tc[1, 1]))


def resize_mask(mask_2d: np.ndarray,
                long_side: int = SAM_IMG_SIZE) -> np.ndarray:
    """
    Resize a 2-D label/mask with nearest-neighbour interpolation,
    then pad to (long_side, long_side).
    mask_2d: (H, W) int/uint8.
    Returns: (long_side, long_side) same dtype.
    """
    import torch
    import torch.nn.functional as F

    H, W = mask_2d.shape
    newh, neww = get_preprocess_shape(H, W, long_side)
    t = torch.from_numpy(mask_2d.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(newh, neww), mode="nearest")
    resized = t.squeeze().numpy().astype(mask_2d.dtype)
    out = np.zeros((long_side, long_side), dtype=mask_2d.dtype)
    out[:newh, :neww] = resized
    return out


def preprocess_image_for_sam(image_3ch: np.ndarray,
                             long_side: int = SAM_IMG_SIZE
                             ) -> Tuple[np.ndarray, int, int]:
    """
    Full SAM preprocess: resize + pad.
    image_3ch: (H, W, 3) float32 (already z-scored / scaled).
    Returns: (1024, 1024, 3) float32, orig_h, orig_w.
    """
    H, W = image_3ch.shape[:2]
    resized = resize_longest_side(image_3ch, long_side)
    padded  = pad_to_square(resized, long_side)
    return padded, H, W
