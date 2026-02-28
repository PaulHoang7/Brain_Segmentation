"""
Volume-level preprocessing for SAM/PG pipeline.

Pipeline per modality:
  1. Load NIfTI → float32
  2. Clip to [p0.5, p99.5] (non-zero voxels)
  3. Z-score (non-zero voxels)
  4. Save as .npy  →  {PROCESSED_DIR}/{case_id}/{mod}.npy

Seg is copied as-is  →  {SEG_DIR}/{case_id}.nii.gz
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import DATASET_ROOT, MODALITIES, PROCESSED_DIR, SEG_DIR
from .io import case_dir, load_seg, load_volume
from .norm import normalize_volume


def preprocess_case(
    case_id: str,
    modalities: Sequence[str] = MODALITIES,
    processed_dir: Path = PROCESSED_DIR,
    seg_dir: Path = SEG_DIR,
    root: Path = DATASET_ROOT,
    p_low: float = 0.5,
    p_high: float = 99.5,
    overwrite: bool = False,
) -> dict:
    """
    Preprocess one case: normalise modalities + copy seg.

    Returns a stats dict:
        {case_id, modalities: {mod: {min, max, mean, std, shape}}, seg_path}
    """
    out_dir = processed_dir / case_id
    seg_out = seg_dir / f"{case_id}.nii.gz"

    # Skip if already done
    if not overwrite and out_dir.exists() and seg_out.exists():
        expected = [out_dir / f"{m}.npy" for m in modalities]
        if all(p.exists() for p in expected):
            return {"case_id": case_id, "status": "skipped"}

    out_dir.mkdir(parents=True, exist_ok=True)
    seg_dir.mkdir(parents=True, exist_ok=True)

    stats: dict = {"case_id": case_id, "modalities": {}, "status": "ok"}

    # ── Process each modality ────────────────────────────────────
    for mod in modalities:
        vol = load_volume(case_id, mod, root)          # float32 (H,W,D)
        vol = normalize_volume(vol, p_low, p_high)     # clip → zscore

        np.save(str(out_dir / f"{mod}.npy"), vol)

        nz = vol[vol != 0]
        stats["modalities"][mod] = {
            "shape": list(vol.shape),
            "min": float(vol.min()),
            "max": float(vol.max()),
            "mean": float(nz.mean()) if nz.size > 0 else 0.0,
            "std": float(nz.std()) if nz.size > 0 else 0.0,
        }

    # ── Copy seg ─────────────────────────────────────────────────
    src_seg = case_dir(case_id, root) / f"{case_id}-seg.nii.gz"
    shutil.copy2(str(src_seg), str(seg_out))
    stats["seg_path"] = str(seg_out)

    return stats
