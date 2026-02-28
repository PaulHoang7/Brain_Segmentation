#!/usr/bin/env python3
"""
[04] Vanilla SAM baseline — zero-shot with GT bboxes.

Runs SAM ViT-B (no fine-tuning) on the test set using ground-truth
bounding boxes as prompts.  Reports Dice + HD95 for WT / TC / ET.

Usage:
    python scripts/04_baseline_vanilla_sam.py [--split test]

Prerequisites:
    - SAM checkpoint at OUTPUT_ROOT/ckpt/sam_vit_b_01ec64.pth
    - pip install git+https://github.com/facebookresearch/segment-anything.git

Output:
    OUTPUT_ROOT/results/baseline_vanilla_sam.csv
    OUTPUT_ROOT/preds/vanilla_sam/{case_id}.nii.gz

Quality gate:
    - Dice WT > 0.5 (sanity — GT bbox should give decent results)
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datn.config import (CKPT_DIR, PREDS_DIR, RESULTS_DIR, SPLITS_JSON,
                          LABEL_MAP_JSON, SAM_IMG_SIZE, MODALITIES,
                          ensure_dirs, load_hparams)
from datn.io import load_volume, load_seg
from datn.norm import zscore_volume
from datn.sam_preprocess import (resize_longest_side, pad_to_square,
                                 transform_bbox)
from datn.prompts import tight_bbox
from datn.metrics import compute_seg_metrics


def main(split: str = "test"):
    ensure_dirs()

    # ── Load SAM ──
    ckpt_path = CKPT_DIR / "sam_vit_b_01ec64.pth"
    if not ckpt_path.exists():
        print(f"ERROR: SAM checkpoint not found at {ckpt_path}")
        print("Download from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
        print(f"Place at: {ckpt_path}")
        sys.exit(1)

    # TODO: implement after SAM is installed
    # from segment_anything import sam_model_registry, SamPredictor
    # sam = sam_model_registry["vit_b"](checkpoint=str(ckpt_path))
    # sam = sam.to("cuda" if torch.cuda.is_available() else "cpu")
    # predictor = SamPredictor(sam)
    raise NotImplementedError(
        "Install segment-anything and uncomment the SAM loading code above. "
        "Then implement the per-slice prediction loop."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    main(**vars(parser.parse_args()))
