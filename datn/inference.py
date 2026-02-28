"""
End-to-end cascade inference pipeline.

Flow per case:
  1. PG predicts objectness + bbox for every slice → smooth along z
  2. SAM_WT(smoothed bbox) → WT mask per slice
  3. Refine bbox from WT mask → SAM_TC → TC mask
  4. Refine bbox from TC mask → SAM_ET → ET mask
  5. Stack slices → 3D → enforce hierarchy → remove small CC
  6. Save as NIfTI
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import nibabel as nib
import torch

from .config import (DATASET_ROOT, SAM_IMG_SIZE, MODALITIES,
                     CKPT_DIR, PREDS_DIR, ensure_dirs, load_hparams)
from .io import load_volume, load_seg, nifti_path
from .norm import zscore_volume
from .prompts import tight_bbox
from .sam_preprocess import (resize_longest_side, pad_to_square,
                             transform_bbox, get_preprocess_shape)
from .postprocess import (enforce_hierarchy, remove_small_cc,
                          smooth_boxes_ema, propagate_bbox)


class CascadeInference:
    """
    Holds PG + 3 SAM+LoRA adapters and runs full cascade.

    Usage:
        ci = CascadeInference(pg_model, sam_wt, sam_tc, sam_et, device)
        pred_wt, pred_tc, pred_et = ci.predict_case(case_id)
    """

    def __init__(self,
                 pg_model: torch.nn.Module,
                 sam_wt: torch.nn.Module,
                 sam_tc: torch.nn.Module,
                 sam_et: torch.nn.Module,
                 device: str = "cuda",
                 smooth_alpha: float = 0.8,
                 min_cc: int = 50):
        self.pg = pg_model.eval().to(device)
        self.sam_wt = sam_wt.eval().to(device)
        self.sam_tc = sam_tc.eval().to(device)
        self.sam_et = sam_et.eval().to(device)
        self.device = device
        self.smooth_alpha = smooth_alpha
        self.min_cc = min_cc

    @torch.no_grad()
    def predict_case(self, case_id: str
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run full cascade for one patient.
        Returns: (wt_3d, tc_3d, et_3d) binary uint8 arrays (H, W, D).
        """
        # Load and normalize volumes
        vols = {}
        for m in MODALITIES:
            vols[m] = zscore_volume(load_volume(case_id, m))
        H, W, D = vols[MODALITIES[0]].shape

        # Step 1: PG predictions for all slices
        pg_boxes, pg_obj = self._pg_all_slices(vols, D)

        # Smooth + propagate
        pg_boxes = smooth_boxes_ema(pg_boxes, self.smooth_alpha)
        pg_boxes = propagate_bbox(pg_boxes, pg_obj)

        # Step 2-4: Cascade SAM per slice
        wt_3d = np.zeros((H, W, D), dtype=np.uint8)
        tc_3d = np.zeros((H, W, D), dtype=np.uint8)
        et_3d = np.zeros((H, W, D), dtype=np.uint8)

        for z in range(D):
            if pg_boxes[z] is None:
                continue

            # Build 3-ch image for this slice
            img_3ch = np.stack([vols[m][:, :, z] for m in MODALITIES], axis=-1)

            # SAM WT
            wt_mask = self._sam_predict(img_3ch, pg_boxes[z], self.sam_wt)
            wt_3d[:, :, z] = wt_mask

            # Refine bbox from WT → SAM TC
            tc_bbox = tight_bbox(wt_mask, pad=5)
            if tc_bbox is not None:
                tc_mask = self._sam_predict(img_3ch, tc_bbox, self.sam_tc)
                tc_3d[:, :, z] = tc_mask

                # Refine bbox from TC → SAM ET
                et_bbox = tight_bbox(tc_mask, pad=5)
                if et_bbox is not None:
                    et_mask = self._sam_predict(img_3ch, et_bbox, self.sam_et)
                    et_3d[:, :, z] = et_mask

        # Step 5: Post-process
        wt_3d, tc_3d, et_3d = enforce_hierarchy(wt_3d, tc_3d, et_3d)
        wt_3d = remove_small_cc(wt_3d, self.min_cc)
        tc_3d = remove_small_cc(tc_3d, self.min_cc)
        et_3d = remove_small_cc(et_3d, self.min_cc)

        return wt_3d, tc_3d, et_3d

    def _pg_all_slices(self, vols: dict, D: int):
        """Run PG on every slice, return boxes and objectness lists."""
        raise NotImplementedError("Wire in after PG training — script 05")

    def _sam_predict(self, img_3ch: np.ndarray,
                     bbox: Tuple[int, int, int, int],
                     sam_model: torch.nn.Module) -> np.ndarray:
        """Run SAM forward on one slice with bbox prompt → binary mask (H,W)."""
        raise NotImplementedError("Wire in after SAM+LoRA training — script 07")

    @staticmethod
    def save_nifti(mask_3d: np.ndarray, case_id: str,
                   output_dir: Path, suffix: str = "pred") -> Path:
        """Save 3D prediction as NIfTI, copying affine from original."""
        ref_path = nifti_path(case_id, "t2f")
        ref_img  = nib.load(str(ref_path))
        out_img  = nib.Nifti1Image(mask_3d.astype(np.int16),
                                   affine=ref_img.affine, header=ref_img.header)
        out_path = output_dir / f"{case_id}-{suffix}.nii.gz"
        nib.save(out_img, str(out_path))
        return out_path
