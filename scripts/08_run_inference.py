#!/usr/bin/env python3
"""
[08] Full cascade inference: PG → SAM_WT → SAM_TC → SAM_ET → postprocess.

Usage:
    python scripts/08_run_inference.py [--split test]

Prerequisites:
    - Trained PG:        OUTPUT_ROOT/ckpt/pg_best.pth
    - Trained SAM+LoRA:  OUTPUT_ROOT/ckpt/sam_lora_{wt,tc,et}_best.pth

Output:
    OUTPUT_ROOT/preds/full_method/{case_id}.nii.gz

Quality gate:
    - Predictions are valid NIfTI with same shape as input
    - ET ⊂ TC ⊂ WT holds for every voxel
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datn.config import (CKPT_DIR, PREDS_DIR, SPLITS_JSON,
                          ensure_dirs, load_hparams)
from datn.inference import CascadeInference


def main(split: str = "test"):
    ensure_dirs()

    preds_dir = PREDS_DIR / "full_method"
    preds_dir.mkdir(parents=True, exist_ok=True)

    # TODO: load PG + 3 SAM+LoRA models
    # TODO: create CascadeInference
    # TODO: loop over cases, predict, save NIfTI
    raise NotImplementedError(
        "Load all checkpoints, instantiate CascadeInference, "
        "run per-case, save NIfTI predictions."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    main(**vars(parser.parse_args()))
