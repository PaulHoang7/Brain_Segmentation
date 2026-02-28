#!/usr/bin/env python3
"""
[09] Evaluate segmentation predictions — Dice + HD95 for WT / TC / ET.

Works for any prediction source (vanilla SAM, SAM+LoRA, full cascade).

Usage:
    python scripts/09_eval_full.py --pred_dir full_method [--split test]

Output:
    OUTPUT_ROOT/results/main_table.csv

Quality gate:
    - CSV has columns: case_id, WT_dice, WT_hd95, TC_dice, TC_hd95, ET_dice, ET_hd95
    - Mean Dice printed to stdout
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path

import numpy as np
import nibabel as nib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datn.config import (PREDS_DIR, RESULTS_DIR, SPLITS_JSON,
                          LABEL_MAP_JSON, ensure_dirs)
from datn.io import load_seg
from datn.metrics import compute_seg_metrics


def main(pred_dir: str = "full_method", split: str = "test"):
    ensure_dirs()

    preds_path = PREDS_DIR / pred_dir
    if not preds_path.exists():
        print(f"ERROR: prediction directory not found: {preds_path}")
        sys.exit(1)

    with open(SPLITS_JSON) as f:
        cases = json.load(f)[split]
    with open(LABEL_MAP_JSON) as f:
        lmap = json.load(f)

    out_csv = RESULTS_DIR / f"{pred_dir}_metrics.csv"
    rows = []

    for cid in cases:
        pred_file = preds_path / f"{cid}-pred.nii.gz"
        if not pred_file.exists():
            print(f"  SKIP {cid} — no prediction file")
            continue

        pred = nib.load(str(pred_file)).get_fdata().astype(np.int16)
        gt   = load_seg(cid)
        m    = compute_seg_metrics(pred, gt, lmap)

        row = {"case_id": cid}
        for region in ("WT", "TC", "ET"):
            row[f"{region}_dice"] = f"{m[region]['dice']:.4f}"
            row[f"{region}_hd95"] = f"{m[region]['hd95']:.2f}"
        rows.append(row)

    if not rows:
        print("No predictions evaluated.")
        return

    # Write CSV
    fieldnames = ["case_id", "WT_dice", "WT_hd95",
                  "TC_dice", "TC_hd95", "ET_dice", "ET_hd95"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Print summary
    for region in ("WT", "TC", "ET"):
        dices = [float(r[f"{region}_dice"]) for r in rows]
        hd95s = [float(r[f"{region}_hd95"]) for r in rows
                 if r[f"{region}_hd95"] != "inf"]
        print(f"  {region}: Dice={np.mean(dices):.4f}  "
              f"HD95={np.mean(hd95s):.2f}" if hd95s else
              f"  {region}: Dice={np.mean(dices):.4f}  HD95=N/A")

    print(f"\nSaved → {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", default="full_method")
    parser.add_argument("--split", default="test")
    main(**vars(parser.parse_args()))
