"""
Torch Datasets for:
  1) PG (Prompt Generator) — 2.5D context (9-ch input)
  2) SAM fine-tuning       — 3-ch input + bbox prompt
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import (DATASET_ROOT, INDEX_DIR, MODALITIES, SAM_IMG_SIZE,
                     PG_CONTEXT_SLICES, BBOX_PAD, LABEL_MAP_JSON)
from .io import load_volume, load_seg
from .norm import zscore_volume
from .prompts import tight_bbox, jitter_bbox
from .sam_preprocess import (resize_longest_side, pad_to_square,
                             transform_bbox, get_preprocess_shape)


def _load_index(split: str) -> List[dict]:
    path = INDEX_DIR / f"{split}.jsonl"
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def _load_label_map() -> dict:
    with open(LABEL_MAP_JSON) as f:
        return json.load(f)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Volume cache (per-case, lazy, keeps one case in memory)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class _VolumeCache:
    """Caches z-scored volumes for the current case to avoid re-loading."""

    def __init__(self):
        self._cid: Optional[str] = None
        self._vols: Dict[str, np.ndarray] = {}
        self._seg:  Optional[np.ndarray] = None

    def get(self, case_id: str, modalities: Tuple[str, ...] = MODALITIES
            ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        if case_id != self._cid:
            self._vols = {}
            for m in modalities:
                vol = load_volume(case_id, m)
                self._vols[m] = zscore_volume(vol)
            self._seg = load_seg(case_id)
            self._cid = case_id
        return self._vols, self._seg


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PG Dataset (2.5D, 9 channels)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class PGDataset(Dataset):
    """
    Prompt Generator dataset.
    Input:  9-ch image  (3 modalities × 3 consecutive slices)
    Target: objectness (0/1) + bbox (x1,y1,x2,y2) normalised to [0,1].
    """

    def __init__(self, split: str = "train",
                 modalities: Tuple[str, ...] = MODALITIES,
                 img_size: int = 224):
        self.rows = _load_index(split)
        self.mods = modalities
        self.img_size = img_size
        self._cache = _VolumeCache()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[idx]
        cid, z, D = row["case_id"], row["z"], row["num_slices"]
        H, W = row["img_shape"]

        vols, seg = self._cache.get(cid, self.mods)

        # Stack 2.5D context: z-1, z, z+1 (clamp at boundaries)
        slices_idx = [max(0, z - 1), z, min(D - 1, z + 1)]
        channels = []
        for m in self.mods:
            for si in slices_idx:
                channels.append(vols[m][:, :, si])
        img = np.stack(channels, axis=0)  # (9, H, W)

        # Simple resize to img_size (PG uses smaller resolution)
        import torch.nn.functional as F
        t = torch.from_numpy(img).unsqueeze(0).float()
        t = F.interpolate(t, size=(self.img_size, self.img_size),
                          mode="bilinear", align_corners=False)
        img_t = t.squeeze(0)  # (9, img_size, img_size)

        # Target
        has_tumor = float(row["has_tumor"])

        if row["bbox_wt"] is not None:
            x1, y1, x2, y2 = row["bbox_wt"]
            bbox_norm = torch.tensor([
                x1 / W, y1 / H, x2 / W, y2 / H
            ], dtype=torch.float32)
        else:
            bbox_norm = torch.zeros(4, dtype=torch.float32)

        return {
            "image":      img_t,                                    # (9, 224, 224)
            "objectness": torch.tensor(has_tumor, dtype=torch.float32),
            "bbox":       bbox_norm,                                # (4,)
            "has_tumor":  torch.tensor(has_tumor, dtype=torch.float32),
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SAM Dataset (3-ch, 1024 resize+pad, bbox prompt)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SAMDataset(Dataset):
    """
    SAM fine-tuning dataset.
    Input:  3-ch image (1024×1024) + bbox prompt
    Target: binary mask (256×256 — SAM output resolution)
    """

    def __init__(self, split: str = "train",
                 target: str = "WT",
                 modalities: Tuple[str, ...] = MODALITIES,
                 jitter: bool = True,
                 jitter_shift: float = 0.1,
                 jitter_scale: float = 0.1,
                 seed: int = 42):
        self.rows = [r for r in _load_index(split) if r["has_tumor"]]
        self.mods = modalities
        self.target = target
        self.jitter = jitter
        self.jitter_shift = jitter_shift
        self.jitter_scale = jitter_scale
        self._cache = _VolumeCache()
        self._lmap = _load_label_map()
        self._rng = np.random.default_rng(seed)

    def _target_labels(self) -> list[int]:
        return self._lmap[self.target]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[idx]
        cid, z = row["case_id"], row["z"]
        H, W = row["img_shape"]

        vols, seg_vol = self._cache.get(cid, self.mods)

        # Build 3-ch image from current slice
        channels = [vols[m][:, :, z] for m in self.mods]
        img = np.stack(channels, axis=-1)  # (H, W, 3)

        # Resize + pad for SAM
        resized = resize_longest_side(img, SAM_IMG_SIZE)
        padded  = pad_to_square(resized, SAM_IMG_SIZE)  # (1024, 1024, 3)

        # Segmentation mask for target region
        seg_slice = seg_vol[:, :, z]
        target_labels = self._target_labels()
        mask = np.isin(seg_slice, target_labels).astype(np.float32)

        # Resize mask to SAM output resolution (256×256)
        mask_resized = resize_longest_side(mask, 256)
        mask_padded  = pad_to_square(mask_resized, 256)
        mask_padded  = (mask_padded > 0.5).astype(np.float32)

        # Bbox prompt (in 1024 space)
        bbox_key = f"bbox_{self.target.lower()}"
        raw_bbox = row.get(bbox_key)
        if raw_bbox is None:
            # Fallback to WT bbox
            raw_bbox = row["bbox_wt"]

        bbox_1024 = transform_bbox(tuple(raw_bbox), H, W, SAM_IMG_SIZE)

        if self.jitter:
            bbox_1024 = jitter_bbox(bbox_1024, SAM_IMG_SIZE, SAM_IMG_SIZE,
                                    self.jitter_shift, self.jitter_scale,
                                    self._rng)

        img_t  = torch.from_numpy(padded).permute(2, 0, 1).float()   # (3, 1024, 1024)
        mask_t = torch.from_numpy(mask_padded).unsqueeze(0).float()   # (1, 256, 256)
        bbox_t = torch.tensor(bbox_1024, dtype=torch.float32)         # (4,)

        return {
            "image":    img_t,
            "mask":     mask_t,
            "bbox":     bbox_t,
            "case_id":  cid,
            "z":        z,
        }
