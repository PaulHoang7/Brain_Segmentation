#!/usr/bin/env python3
"""
[02] Build per-slice JSONL index for each split.

Each JSONL row contains:
    case_id, z, num_slices, img_shape,
    has_tumor_wt, has_tumor_tc, has_tumor_et,
    bbox_gt_wt, bbox_gt_tc, bbox_gt_et,     (tight + pad=5)
    area_wt, area_tc, area_et,              (pixel count)
    paths {t2f, t1c, t2w, seg}

Reads:
    - splits.json       (from 01_make_splits)
    - label_map.json    (from 00_label_probe)
    - seg from SEG_DIR  (from 01_preprocess)

Usage:
    python scripts/02_build_slice_index.py

Output:
    OUTPUT_ROOT/index/{train,val,test}.jsonl
    stdout: pos/neg counts + tumor area distribution
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datn.config import (
    BBOX_PAD,
    INDEX_DIR,
    LABEL_MAP_JSON,
    MODALITIES,
    PROCESSED_DIR,
    SEG_DIR,
    SPLITS_JSON,
    ensure_dirs,
)
from datn.prompts import tight_bbox


# ── Helpers ──────────────────────────────────────────────────────
def mask_from_labels(seg_slice: np.ndarray, label_list: list[int]) -> np.ndarray:
    return np.isin(seg_slice, label_list).astype(np.uint8)


def load_seg_from_dir(case_id: str) -> np.ndarray:
    """Load seg from preprocessed SEG_DIR."""
    path = SEG_DIR / f"{case_id}.nii.gz"
    img = nib.load(str(path))
    return np.asarray(img.dataobj, dtype=np.int16)


def case_paths(case_id: str) -> dict:
    """Return dict of paths to preprocessed files for one case."""
    p = {}
    for mod in MODALITIES:
        p[mod] = str(PROCESSED_DIR / case_id / f"{mod}.npy")
    p["seg"] = str(SEG_DIR / f"{case_id}.nii.gz")
    return p


# ── Main ─────────────────────────────────────────────────────────
def main():
    ensure_dirs()

    with open(SPLITS_JSON) as f:
        splits = json.load(f)
    with open(LABEL_MAP_JSON) as f:
        lmap = json.load(f)

    wt_labels = lmap["WT"]
    tc_labels = lmap["TC"]
    et_labels = lmap["ET"]

    print(f"[index] Label map: WT={wt_labels}  TC={tc_labels}  ET={et_labels}")
    print(f"[index] Bbox pad : {BBOX_PAD}")
    print(f"[index] Modalities: {MODALITIES}")
    print()

    grand_stats: dict[str, dict] = {}

    for split_name in ("train", "val", "test"):
        case_ids = splits[split_name]
        out_path = INDEX_DIR / f"{split_name}.jsonl"

        # Counters
        n_pos_wt = n_pos_tc = n_pos_et = 0
        n_neg = 0
        total_slices = 0
        areas_wt: list[int] = []
        areas_tc: list[int] = []
        areas_et: list[int] = []

        with open(out_path, "w") as fout:
            for ci, cid in enumerate(case_ids):
                seg = load_seg_from_dir(cid)
                H, W, D = seg.shape
                paths = case_paths(cid)

                for z in range(D):
                    seg_slice = seg[:, :, z]
                    total_slices += 1

                    # ── WT ────────────────────────────────────
                    wt_mask = mask_from_labels(seg_slice, wt_labels)
                    a_wt = int(wt_mask.sum())
                    has_wt = int(a_wt > 0)

                    # ── TC ────────────────────────────────────
                    tc_mask = mask_from_labels(seg_slice, tc_labels)
                    a_tc = int(tc_mask.sum())
                    has_tc = int(a_tc > 0)

                    # ── ET ────────────────────────────────────
                    et_mask = mask_from_labels(seg_slice, et_labels)
                    a_et = int(et_mask.sum())
                    has_et = int(a_et > 0)

                    # Bboxes (tight + pad)
                    bbox_wt = tight_bbox(wt_mask, pad=BBOX_PAD) if has_wt else None
                    bbox_tc = tight_bbox(tc_mask, pad=BBOX_PAD) if has_tc else None
                    bbox_et = tight_bbox(et_mask, pad=BBOX_PAD) if has_et else None

                    # Count
                    if has_wt:
                        n_pos_wt += 1
                        areas_wt.append(a_wt)
                    if has_tc:
                        n_pos_tc += 1
                        areas_tc.append(a_tc)
                    if has_et:
                        n_pos_et += 1
                        areas_et.append(a_et)
                    if not has_wt:
                        n_neg += 1

                    row = {
                        "case_id":       cid,
                        "z":             z,
                        "num_slices":    D,
                        "has_tumor_wt":  has_wt,
                        "has_tumor_tc":  has_tc,
                        "has_tumor_et":  has_et,
                        "bbox_gt_wt":    list(bbox_wt) if bbox_wt else None,
                        "bbox_gt_tc":    list(bbox_tc) if bbox_tc else None,
                        "bbox_gt_et":    list(bbox_et) if bbox_et else None,
                        "area_wt":       a_wt,
                        "area_tc":       a_tc,
                        "area_et":       a_et,
                        "img_shape":     [H, W],
                        "paths":         paths,
                    }
                    fout.write(json.dumps(row) + "\n")

                if (ci + 1) % 100 == 0:
                    print(f"  {split_name}: {ci+1}/{len(case_ids)} cases ...")

        # ── Print stats ──────────────────────────────────────────
        pct_wt = n_pos_wt / total_slices if total_slices else 0
        pct_tc = n_pos_tc / total_slices if total_slices else 0
        pct_et = n_pos_et / total_slices if total_slices else 0

        def area_stats(areas: list[int]) -> dict:
            if not areas:
                return {"count": 0}
            a = np.array(areas)
            return {
                "count":  len(a),
                "mean":   float(np.mean(a)),
                "median": float(np.median(a)),
                "min":    int(np.min(a)),
                "max":    int(np.max(a)),
                "p25":    float(np.percentile(a, 25)),
                "p75":    float(np.percentile(a, 75)),
            }

        stats = {
            "cases": len(case_ids),
            "total_slices": total_slices,
            "pos_wt": n_pos_wt, "pos_tc": n_pos_tc, "pos_et": n_pos_et,
            "neg": n_neg,
            "pct_wt": round(pct_wt, 4),
            "pct_tc": round(pct_tc, 4),
            "pct_et": round(pct_et, 4),
            "area_wt": area_stats(areas_wt),
            "area_tc": area_stats(areas_tc),
            "area_et": area_stats(areas_et),
        }
        grand_stats[split_name] = stats

        print(f"\n{'=' * 60}")
        print(f"[{split_name}]  {len(case_ids)} cases, {total_slices} slices")
        print(f"  pos_WT = {n_pos_wt:>6}  ({pct_wt:.1%})   "
              f"neg = {n_neg:>6}  ({1-pct_wt:.1%})")
        print(f"  pos_TC = {n_pos_tc:>6}  ({pct_tc:.1%})")
        print(f"  pos_ET = {n_pos_et:>6}  ({pct_et:.1%})")

        for tag, areas in [("WT", areas_wt), ("TC", areas_tc), ("ET", areas_et)]:
            if areas:
                a = np.array(areas)
                print(f"  area_{tag}: mean={np.mean(a):.0f}  "
                      f"median={np.median(a):.0f}  "
                      f"[{np.min(a)}, {np.max(a)}]  "
                      f"p25={np.percentile(a,25):.0f}  "
                      f"p75={np.percentile(a,75):.0f}")

        print(f"  → {out_path}")

    # ── Save summary ─────────────────────────────────────────────
    summary_path = INDEX_DIR / "index_stats.json"
    with open(summary_path, "w") as f:
        json.dump(grand_stats, f, indent=2)
    print(f"\n[index] Stats → {summary_path}")


if __name__ == "__main__":
    main()
