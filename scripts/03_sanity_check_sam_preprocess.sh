#!/usr/bin/env bash
# ── 03 — SAM Preprocess Sanity Check ────────────────────────────
# Picks random tumour slices, runs SAM resize+pad, transforms bboxes,
# and saves debug overlay images.
#
# Output:
#   {OUTPUT_ROOT}/debug/sam_preprocess/sample_*.png
#
# Usage:
#   bash scripts/03_sanity_check_sam_preprocess.sh        # 3 samples
#   bash scripts/03_sanity_check_sam_preprocess.sh 5      # 5 samples
set -euo pipefail
cd "$(dirname "$0")/.."

N=${1:-3}
SEED=${2:-42}

echo ">>> Running 03_sanity_check.py  (n=$N, seed=$SEED)"
python scripts/03_sanity_check.py --n "$N" --seed "$SEED"
echo ">>> Done."
