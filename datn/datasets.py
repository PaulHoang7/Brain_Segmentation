"""
Torch Datasets for:
  1) PGDataset2p5D  — 2.5D context (9-ch) for Prompt Generator
  2) SAMDataset     — 3-ch (1024×1024) + mask + bbox prompt for SAM LoRA

Both datasets read **preprocessed** .npy volumes (z-scored + clipped)
and the new JSONL index (has_tumor_wt/tc/et, bbox_gt_wt/tc/et, paths).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .config import (
    BBOX_PAD,
    INDEX_DIR,
    LABEL_MAP_JSON,
    MODALITIES,
    PG_IMG_SIZE,
    SAM_IMG_SIZE,
)
from .prompts import jitter_bbox
from .sam_preprocess import (
    get_preprocess_shape,
    pad_to_square,
    resize_longest_side,
    resize_mask,
    transform_bbox,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Index + label-map loaders
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _load_index(split: str) -> List[dict]:
    path = INDEX_DIR / f"{split}.jsonl"
    with open(path) as f:
        return [json.loads(line) for line in f]


def _load_label_map() -> dict:
    with open(LABEL_MAP_JSON) as f:
        return json.load(f)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Volume cache  — loads from preprocessed .npy, keeps one case
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class _VolumeCache:
    """
    Caches preprocessed volumes for the current case_id.
    Reads .npy (already z-scored+clipped) and seg .nii.gz.
    """
    def __init__(self):
        self._cid: Optional[str] = None
        self._vols: Dict[str, np.ndarray] = {}
        self._seg: Optional[np.ndarray] = None

    def get(self, case_id: str, paths: dict,
            modalities: Tuple[str, ...] = MODALITIES,
            ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        if case_id != self._cid:
            self._vols = {}
            for m in modalities:
                self._vols[m] = np.load(paths[m])          # (H, W, D) float32
            img = nib.load(paths["seg"])
            self._seg = np.asarray(img.dataobj, dtype=np.int16)
            self._cid = case_id
        return self._vols, self._seg


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PGDataset2p5D  — 9 channels, objectness + bbox
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class PGDataset2p5D(Dataset):
    """
    Prompt Generator dataset (2.5D context).

    Input:  (9, PG_IMG_SIZE, PG_IMG_SIZE)
            = 3 modalities × 3 consecutive slices (z-1, z, z+1)
    Target: objectness (0/1)  — has any WT tumour
            bbox (4,)         — normalised [0,1]  (x1/W, y1/H, x2/W, y2/H)
    """

    def __init__(self, split: str = "train",
                 modalities: Tuple[str, ...] = MODALITIES,
                 img_size: int = PG_IMG_SIZE):
        self.rows = _load_index(split)
        self.mods = modalities
        self.img_size = img_size
        self._cache = _VolumeCache()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[idx]
        cid = row["case_id"]
        z = row["z"]
        D = row["num_slices"]
        H, W = row["img_shape"]
        paths = row["paths"]

        vols, _ = self._cache.get(cid, paths, self.mods)

        # Stack 2.5D:  [z-1, z, z+1] × 3 modalities → 9 channels
        zs = [max(0, z - 1), z, min(D - 1, z + 1)]
        channels = []
        for m in self.mods:
            for zi in zs:
                channels.append(vols[m][:, :, zi])
        img = np.stack(channels, axis=0)                # (9, H, W)

        # Resize to PG resolution
        t = torch.from_numpy(img).unsqueeze(0).float()  # (1, 9, H, W)
        t = F.interpolate(t, size=(self.img_size, self.img_size),
                          mode="bilinear", align_corners=False)
        img_t = t.squeeze(0)                            # (9, img_size, img_size)

        # Objectness: does WT tumour exist in this slice?
        has_tumor = float(row["has_tumor_wt"])

        # Bbox: normalised to [0,1]
        bbox_raw = row["bbox_gt_wt"]
        if bbox_raw is not None:
            x1, y1, x2, y2 = bbox_raw
            bbox_norm = torch.tensor(
                [x1 / W, y1 / H, x2 / W, y2 / H], dtype=torch.float32)
        else:
            bbox_norm = torch.zeros(4, dtype=torch.float32)

        return {
            "image":      img_t,                                       # (9, 224, 224)
            "objectness": torch.tensor(has_tumor, dtype=torch.float32),
            "bbox":       bbox_norm,                                   # (4,)
            "case_id":    cid,
            "z":          z,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SAMDataset  — 3-ch 1024×1024 + mask 256×256 + bbox prompt
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SAMDataset(Dataset):
    """
    SAM LoRA fine-tuning dataset.

    Filters index to tumour-positive slices for the target class.

    Input:  image (3, 1024, 1024)  — SAM-preprocessed 3-mod composite
            bbox  (4,)             — in 1024-space, optionally jittered
    Target: mask  (1, 256, 256)    — binary, SAM decoder output resolution
    """

    SAM_MASK_SIZE = 256   # SAM decoder output resolution

    def __init__(self, split: str = "train",
                 target: str = "WT",
                 modalities: Tuple[str, ...] = MODALITIES,
                 jitter: bool = True,
                 jitter_shift: float = 0.1,
                 jitter_scale: float = 0.1,
                 seed: int = 42):
        all_rows = _load_index(split)
        # Filter: only slices positive for the target class
        target_key = f"has_tumor_{target.lower()}"
        self.rows = [r for r in all_rows if r.get(target_key, 0)]

        self.mods = modalities
        self.target = target.upper()
        self.jitter = jitter
        self.jitter_shift = jitter_shift
        self.jitter_scale = jitter_scale
        self._cache = _VolumeCache()
        self._lmap = _load_label_map()
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[idx]
        cid = row["case_id"]
        z = row["z"]
        H, W = row["img_shape"]
        paths = row["paths"]

        vols, seg_vol = self._cache.get(cid, paths, self.mods)

        # ── 3-channel image (current slice) ──────────────────────
        channels = [vols[m][:, :, z] for m in self.mods]
        img_3ch = np.stack(channels, axis=-1)               # (H, W, 3)

        # SAM preprocess: resize longest side → 1024, pad
        resized = resize_longest_side(img_3ch, SAM_IMG_SIZE)
        padded = pad_to_square(resized, SAM_IMG_SIZE)       # (1024, 1024, 3)

        # ── Target mask ──────────────────────────────────────────
        seg_slice = seg_vol[:, :, z]
        target_labels = self._lmap[self.target]
        mask_orig = np.isin(seg_slice, target_labels).astype(np.uint8)

        # Resize mask to 256×256 (SAM decoder output resolution)
        mask_256 = resize_mask(mask_orig, self.SAM_MASK_SIZE)
        mask_256 = (mask_256 > 0).astype(np.float32)

        # ── Bbox prompt (in 1024 space) ──────────────────────────
        bbox_key = f"bbox_gt_{self.target.lower()}"
        raw_bbox = row.get(bbox_key)
        if raw_bbox is None:
            # Fallback to WT bbox
            raw_bbox = row["bbox_gt_wt"]

        bbox_1024 = transform_bbox(tuple(raw_bbox), H, W, SAM_IMG_SIZE)

        if self.jitter:
            bbox_1024 = jitter_bbox(
                bbox_1024, SAM_IMG_SIZE, SAM_IMG_SIZE,
                self.jitter_shift, self.jitter_scale, self._rng,
            )

        # ── To tensors ───────────────────────────────────────────
        img_t = torch.from_numpy(padded).permute(2, 0, 1).float()    # (3, 1024, 1024)
        mask_t = torch.from_numpy(mask_256).unsqueeze(0).float()      # (1, 256, 256)
        bbox_t = torch.tensor(bbox_1024, dtype=torch.float32)         # (4,)

        return {
            "image":   img_t,
            "mask":    mask_t,
            "bbox":    bbox_t,
            "case_id": cid,
            "z":       z,
        }
