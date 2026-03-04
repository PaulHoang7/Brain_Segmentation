#!/usr/bin/env bash
# ── 04 — Debug Dataloader ────────────────────────────────────────
# Smoke-test PGDataset2p5D + SAMDataset + weighted samplers.
# Saves debug visualizations.
#
# Output:
#   {OUTPUT_ROOT}/debug/dataloader/pg_sample_*.png
#   {OUTPUT_ROOT}/debug/dataloader/sam_{wt,tc,et}_sample_*.png
#
# Usage:
#   bash scripts/04_debug_dataloader.sh        # 3 samples
#   bash scripts/04_debug_dataloader.sh 5      # 5 samples
set -euo pipefail
cd "$(dirname "$0")/.."

N=${1:-3}
SEED=${2:-42}

echo ">>> Running 04_debug_dataloader.py  (n=$N, seed=$SEED)"
python scripts/04_debug_dataloader.py --n "$N" --seed "$SEED"
echo ">>> Done."
