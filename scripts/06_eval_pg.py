#!/usr/bin/env python3
"""
[06] Evaluate trained Prompt Generator.

Reports:
    - Slice detection: Precision / Recall / F1
    - Bbox IoU (tumor slices only)
    - Temporal stability: delta-center, delta-area (before/after smoothing)

Usage:
    python scripts/06_eval_pg.py [--split val] [--ckpt pg_best.pth]

Output:
    OUTPUT_ROOT/results/pg_metrics.csv
    OUTPUT_ROOT/preds/pg_boxes/{split}.json

Quality gate:
    - Detection F1 > 0.85
    - Bbox IoU > 0.60
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datn.config import (CKPT_DIR, RESULTS_DIR, PREDS_DIR,
                          ensure_dirs, load_hparams)
from datn.pg_model import PromptGenerator
from datn.metrics import pg_detection_metrics, pg_bbox_iou, pg_stability


def main(split: str = "val", ckpt: str = "pg_best.pth"):
    ensure_dirs()

    ckpt_path = CKPT_DIR / ckpt
    if not ckpt_path.exists():
        print(f"ERROR: PG checkpoint not found at {ckpt_path}")
        print("Run scripts/05_train_pg.py first.")
        sys.exit(1)

    # TODO: load model, run inference on split, compute metrics
    raise NotImplementedError(
        "Load PG checkpoint, run on val/test, compute detection + bbox + "
        "stability metrics, save CSV."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val")
    parser.add_argument("--ckpt", default="pg_best.pth")
    main(**vars(parser.parse_args()))
