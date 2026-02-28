#!/usr/bin/env bash
# ── 01 — Preprocess: z-score + clip for SAM/PG modalities ───────
# Reads cases.csv, normalises t2f/t1c/t2w, copies seg.
#
# Outputs:
#   {OUTPUT_ROOT}/data/processed/{case_id}/{t2f,t1c,t2w}.npy
#   {OUTPUT_ROOT}/data/seg/{case_id}.nii.gz
#   {OUTPUT_ROOT}/logs/preprocess_log.json
#
# Usage:
#   bash scripts/01_preprocess.sh              # sequential
#   bash scripts/01_preprocess.sh 8            # 8 workers
#   bash scripts/01_preprocess.sh 4 --overwrite
set -euo pipefail
cd "$(dirname "$0")/.."

WORKERS=${1:-1}
shift 2>/dev/null || true
EXTRA_ARGS="$*"

echo ">>> Running 01_preprocess.py  (workers=$WORKERS $EXTRA_ARGS)"
python scripts/01_preprocess.py --workers "$WORKERS" $EXTRA_ARGS
echo ">>> Done."
