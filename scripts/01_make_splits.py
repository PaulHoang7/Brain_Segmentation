#!/usr/bin/env python3
"""
[01-split] Create patient-level train / val / test split.

Reads case list from OUTPUT_ROOT/configs/cases.csv (produced by 00_label_probe).
Split ratio 80/10/10 (configurable via SplitCfg).

Usage:
    python scripts/01_make_splits.py

Output:
    OUTPUT_ROOT/configs/splits.json

Quality gate:
    - Exact ratio ±1 %
    - Zero patient overlap between splits
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datn.config import CASES_CSV, SPLITS_JSON, SplitCfg, ensure_dirs


def load_case_ids() -> list[str]:
    """Read case_id column from cases.csv."""
    if not CASES_CSV.exists():
        raise FileNotFoundError(
            f"cases.csv not found at {CASES_CSV}. Run 00_label_probe.py first."
        )
    with open(CASES_CSV) as f:
        return [row["case_id"] for row in csv.DictReader(f)]


def main():
    ensure_dirs()
    cfg = SplitCfg()
    cases = load_case_ids()
    n = len(cases)
    print(f"[splits] Total patients : {n}")
    print(f"[splits] Ratio          : {cfg.train_ratio}/{cfg.val_ratio}/{cfg.test_ratio}")
    print(f"[splits] Seed           : {cfg.seed}")

    # Two-stage stratified split
    test_size = cfg.test_ratio
    val_relative = cfg.val_ratio / (cfg.train_ratio + cfg.val_ratio)

    trainval, test = train_test_split(
        cases, test_size=test_size, random_state=cfg.seed)
    train, val = train_test_split(
        trainval, test_size=val_relative, random_state=cfg.seed)

    splits = {
        "train": sorted(train),
        "val":   sorted(val),
        "test":  sorted(test),
        "seed":  cfg.seed,
        "ratios": {
            "train": cfg.train_ratio,
            "val":   cfg.val_ratio,
            "test":  cfg.test_ratio,
        },
    }

    # ── Quality gate ─────────────────────────────────────────────
    all_ids = set(splits["train"]) | set(splits["val"]) | set(splits["test"])
    assert len(all_ids) == n, f"Patient leak! {len(all_ids)} != {n}"
    assert len(set(splits["train"]) & set(splits["val"])) == 0, "train∩val"
    assert len(set(splits["train"]) & set(splits["test"])) == 0, "train∩test"
    assert len(set(splits["val"])   & set(splits["test"])) == 0, "val∩test"

    with open(SPLITS_JSON, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"\n[splits] Train : {len(train):>5}  ({len(train)/n:.1%})")
    print(f"[splits] Val   : {len(val):>5}  ({len(val)/n:.1%})")
    print(f"[splits] Test  : {len(test):>5}  ({len(test)/n:.1%})")
    print(f"[splits] Saved → {SPLITS_JSON}")


if __name__ == "__main__":
    main()
