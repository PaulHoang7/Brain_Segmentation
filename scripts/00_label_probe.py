#!/usr/bin/env python3
"""
[00] Dataset Scan + Label Probe
=================================
1. Scan ALL cases → cases.csv  (case_id, modalities, shape, has_seg)
2. Probe N random segmentations → label distribution
3. Save label_map.json + label_probe_raw.json

Usage:
    python scripts/00_label_probe.py [--n 30] [--seed 42]

Outputs (under OUTPUT_ROOT/configs/):
    cases.csv            — full dataset manifest
    label_map.json       — {WT, TC, ET, label_names, ...}
    label_probe_raw.json — per-case probe details
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datn.config import CONFIGS_DIR, DATASET_ROOT, ensure_dirs
from datn.io import list_cases, load_seg, scan_case


# ── Step 1: Dataset scan ─────────────────────────────────────────
def scan_dataset() -> list[dict]:
    """Scan every case directory and return a list of metadata dicts."""
    cases = list_cases()
    print(f"[scan] Dataset root : {DATASET_ROOT}")
    print(f"[scan] Total cases  : {len(cases)}")

    rows = []
    for i, cid in enumerate(cases):
        meta = scan_case(cid)
        rows.append(meta)
        if (i + 1) % 200 == 0:
            print(f"  scanned {i + 1}/{len(cases)} ...")

    return rows


def save_cases_csv(rows: list[dict], path: Path) -> None:
    """Write cases.csv from list of metadata dicts."""
    fieldnames = [
        "case_id", "modalities", "n_modalities",
        "has_seg", "shape_H", "shape_W", "shape_D",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[scan] Saved → {path}  ({len(rows)} rows)")


# ── Step 2: Label probe ─────────────────────────────────────────
def probe_labels(n: int, seed: int) -> tuple[list[dict], Counter]:
    """Load seg for N random cases, collect label stats."""
    cases = list_cases()
    rng = np.random.default_rng(seed)
    sample = sorted(rng.choice(cases, size=min(n, len(cases)), replace=False))

    all_labels: Counter = Counter()
    per_case = []

    print(f"\n[probe] Sampling {len(sample)} / {len(cases)} cases (seed={seed})")
    for cid in sample:
        seg = load_seg(cid)
        uniq, counts = np.unique(seg, return_counts=True)
        lc = {int(u): int(c) for u, c in zip(uniq, counts)}
        all_labels.update(lc)
        per_case.append({
            "case_id": cid,
            "shape": list(seg.shape),
            "labels": lc,
        })
        print(f"  {cid}  shape={seg.shape}  labels={lc}")

    return per_case, all_labels


def build_label_map(all_labels: Counter, n_probed: int,
                    n_total: int) -> dict:
    """Determine BraTS label scheme from observed labels."""
    label_set = set(all_labels.keys()) - {0}

    print(f"\n[probe] === Aggregate label distribution ===")
    for lab in sorted(all_labels):
        print(f"  Label {lab}: {all_labels[lab]:>14,} voxels")
    print(f"[probe] Non-zero labels: {sorted(label_set)}")

    if label_set == {1, 2, 3}:
        mapping = {
            "WT": [1, 2, 3],
            "TC": [1, 3],
            "ET": [3],
            "label_names": {
                "0": "background", "1": "NCR", "2": "ED", "3": "ET",
            },
        }
        print("[probe] Detected BraTS-GLI 2023 scheme: {1:NCR, 2:ED, 3:ET}")
    elif label_set == {1, 2, 4}:
        mapping = {
            "WT": [1, 2, 4],
            "TC": [1, 4],
            "ET": [4],
            "label_names": {
                "0": "background", "1": "NCR", "2": "ED", "4": "ET",
            },
        }
        print("[probe] Detected BraTS 2021 scheme: {1:NCR, 2:ED, 4:ET}")
    else:
        mapping = {
            "WT": sorted(label_set),
            "TC": "UNKNOWN",
            "ET": "UNKNOWN",
            "label_names": {str(l): f"label_{l}" for l in sorted(label_set)},
        }
        print(f"[probe] WARNING: unexpected labels {label_set} — check manually!")

    mapping["probed_cases"] = n_probed
    mapping["total_cases"] = n_total
    return mapping


# ── Main ─────────────────────────────────────────────────────────
def main(n: int = 30, seed: int = 42):
    t0 = time.time()
    ensure_dirs()

    # Step 1 — full dataset scan → cases.csv
    print("=" * 60)
    print("STEP 1: Dataset scan")
    print("=" * 60)
    rows = scan_dataset()
    cases_csv_path = CONFIGS_DIR / "cases.csv"
    save_cases_csv(rows, cases_csv_path)

    # Quick summary
    n_with_seg = sum(1 for r in rows if r["has_seg"])
    shapes = set(
        (r["shape_H"], r["shape_W"], r["shape_D"]) for r in rows
    )
    print(f"[scan] Cases with seg : {n_with_seg}/{len(rows)}")
    print(f"[scan] Unique shapes  : {shapes}")

    # Step 2 — label probe (30 cases) → label_map.json
    print()
    print("=" * 60)
    print(f"STEP 2: Label probe (n={n})")
    print("=" * 60)
    per_case, all_labels = probe_labels(n, seed)
    mapping = build_label_map(all_labels, len(per_case), len(rows))

    # Save
    label_map_path = CONFIGS_DIR / "label_map.json"
    with open(label_map_path, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"\n[save] label_map.json → {label_map_path}")

    probe_raw_path = CONFIGS_DIR / "label_probe_raw.json"
    with open(probe_raw_path, "w") as f:
        json.dump(per_case, f, indent=2)
    print(f"[save] label_probe_raw.json → {probe_raw_path}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Dataset scan + label probe")
    p.add_argument("--n", type=int, default=30, help="cases to probe")
    p.add_argument("--seed", type=int, default=42, help="RNG seed")
    main(**vars(p.parse_args()))
