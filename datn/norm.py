"""MRI intensity normalization: z-score per volume over non-zero voxels."""
from __future__ import annotations
import numpy as np


def zscore_volume(vol: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Z-score normalise a 3-D MRI volume **over non-zero voxels only**.
    Zero voxels (background) stay zero.
    Returns float32.
    """
    vol = vol.astype(np.float32)
    mask = vol > 0
    if mask.sum() == 0:
        return vol                       # all-zero volume → no-op
    mu  = vol[mask].mean()
    std = vol[mask].std() + eps
    out = np.zeros_like(vol)
    out[mask] = (vol[mask] - mu) / std
    return out


def zscore_slice(slc: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Same as zscore_volume but for a 2-D slice (H, W)."""
    return zscore_volume(slc, eps)       # logic is identical


def clip_percentile(vol: np.ndarray,
                    p_low: float = 0.5,
                    p_high: float = 99.5) -> np.ndarray:
    """
    Clip intensities to [p_low, p_high] percentile range.
    Only considers non-zero voxels for percentile computation.
    Zero voxels remain zero.  Returns float32.
    """
    vol = vol.astype(np.float32)
    mask = vol != 0
    if mask.sum() == 0:
        return vol
    vals = vol[mask]
    lo = np.percentile(vals, p_low)
    hi = np.percentile(vals, p_high)
    out = vol.copy()
    out[mask] = np.clip(vals, lo, hi)
    return out


def normalize_volume(vol: np.ndarray,
                     p_low: float = 0.5,
                     p_high: float = 99.5,
                     eps: float = 1e-8) -> np.ndarray:
    """Full pipeline: clip percentile → z-score.  Returns float32."""
    vol = clip_percentile(vol, p_low, p_high)
    vol = zscore_volume(vol, eps)
    return vol
