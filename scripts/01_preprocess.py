#!/usr/bin/env python3
"""
[01] Preprocess — normalise modalities + copy seg for all cases.

Reads cases from OUTPUT_ROOT/configs/cases.csv.
For each case:
    1. Load t2f, t1c, t2w  (3 modalities for SAM/PG)
    2. Clip p0.5–p99.5 (non-zero) → z-score (non-zero)
    3. Save .npy  →  {OUTPUT_ROOT}/data/processed/{case_id}/{mod}.npy
    4. Copy seg   →  {OUTPUT_ROOT}/data/seg/{case_id}.nii.gz

Usage:
    python scripts/01_preprocess.py                    # all cases
    python scripts/01_preprocess.py --workers 8        # parallel
    python scripts/01_preprocess.py --overwrite        # force redo

Outputs:
    {OUTPUT_ROOT}/data/processed/{case_id}/{t2f,t1c,t2w}.npy
    {OUTPUT_ROOT}/data/seg/{case_id}.nii.gz
    {OUTPUT_ROOT}/logs/preprocess_log.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datn.config import (
    CASES_CSV,
    CONFIGS_DIR,
    LOGS_DIR,
    MODALITIES,
    PROCESSED_DIR,
    SEG_DIR,
    ensure_dirs,
)


# ── Worker function (top-level for pickling) ─────────────────────
def _process_one(case_id: str, modalities: tuple, processed_dir: str,
                 seg_dir: str, overwrite: bool) -> dict:
    """Wrapper that imports inside worker to avoid pickling issues."""
    from datn.preprocess import preprocess_case
    return preprocess_case(
        case_id,
        modalities=modalities,
        processed_dir=Path(processed_dir),
        seg_dir=Path(seg_dir),
        overwrite=overwrite,
    )


def load_case_ids() -> list[str]:
    """Read case_id column from cases.csv."""
    csv_path = CASES_CSV
    if not csv_path.exists():
        # Fallback: try configs dir
        csv_path = CONFIGS_DIR / "cases.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"cases.csv not found at {CASES_CSV}. Run 00_label_probe.py first."
        )
    ids = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            ids.append(row["case_id"])
    print(f"[preprocess] Loaded {len(ids)} cases from {csv_path}")
    return ids


def main(workers: int = 1, overwrite: bool = False):
    t0 = time.time()
    ensure_dirs()

    case_ids = load_case_ids()
    n = len(case_ids)
    mods = MODALITIES  # (t2f, t1c, t2w)

    print(f"[preprocess] Modalities : {mods}")
    print(f"[preprocess] Output     : {PROCESSED_DIR}")
    print(f"[preprocess] Seg dir    : {SEG_DIR}")
    print(f"[preprocess] Workers    : {workers}")
    print(f"[preprocess] Overwrite  : {overwrite}")
    print(f"[preprocess] Processing {n} cases ...")
    print()

    results = []
    done = 0
    skipped = 0
    errors = 0

    if workers <= 1:
        # Sequential
        for i, cid in enumerate(case_ids):
            try:
                stats = _process_one(
                    cid, mods, str(PROCESSED_DIR), str(SEG_DIR), overwrite
                )
                results.append(stats)
                if stats.get("status") == "skipped":
                    skipped += 1
                else:
                    done += 1
            except Exception as e:
                errors += 1
                results.append({"case_id": cid, "status": "error",
                                "error": str(e)})
                print(f"  ERROR {cid}: {e}")

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{n}] done={done} skipped={skipped} "
                      f"errors={errors}")
    else:
        # Parallel
        futures = {}
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for cid in case_ids:
                fut = pool.submit(
                    _process_one, cid, mods,
                    str(PROCESSED_DIR), str(SEG_DIR), overwrite,
                )
                futures[fut] = cid

            for i, fut in enumerate(as_completed(futures)):
                cid = futures[fut]
                try:
                    stats = fut.result()
                    results.append(stats)
                    if stats.get("status") == "skipped":
                        skipped += 1
                    else:
                        done += 1
                except Exception as e:
                    errors += 1
                    results.append({"case_id": cid, "status": "error",
                                    "error": str(e)})
                    print(f"  ERROR {cid}: {e}")

                if (i + 1) % 50 == 0:
                    print(f"  [{i+1}/{n}] done={done} skipped={skipped} "
                          f"errors={errors}")

    elapsed = time.time() - t0

    # ── Summary ──────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"[preprocess] DONE in {elapsed:.1f}s "
          f"({elapsed/n:.2f}s/case)")
    print(f"  Processed : {done}")
    print(f"  Skipped   : {skipped}")
    print(f"  Errors    : {errors}")
    print(f"  Total     : {n}")

    # ── Aggregate stats for processed cases ──────────────────────
    ok_results = [r for r in results if r.get("status") == "ok"]
    if ok_results:
        print()
        print("[preprocess] Normalisation stats (across processed cases):")
        for mod in mods:
            mins = [r["modalities"][mod]["min"] for r in ok_results]
            maxs = [r["modalities"][mod]["max"] for r in ok_results]
            means = [r["modalities"][mod]["mean"] for r in ok_results]
            stds = [r["modalities"][mod]["std"] for r in ok_results]
            import numpy as np
            print(f"  {mod}:")
            print(f"    min  range: [{np.min(mins):.3f}, {np.max(mins):.3f}]")
            print(f"    max  range: [{np.min(maxs):.3f}, {np.max(maxs):.3f}]")
            print(f"    mean range: [{np.min(means):.4f}, {np.max(means):.4f}]")
            print(f"    std  range: [{np.min(stds):.4f}, {np.max(stds):.4f}]")

    # ── Save log ─────────────────────────────────────────────────
    log_path = LOGS_DIR / "preprocess_log.json"
    log = {
        "n_total": n,
        "n_processed": done,
        "n_skipped": skipped,
        "n_errors": errors,
        "elapsed_s": round(elapsed, 1),
        "modalities": list(mods),
        "per_case": results,
    }
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\n[preprocess] Log → {log_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Preprocess BraTS volumes")
    p.add_argument("--workers", type=int, default=1,
                   help="parallel workers (default: 1)")
    p.add_argument("--overwrite", action="store_true",
                   help="re-process even if output exists")
    main(**vars(p.parse_args()))
