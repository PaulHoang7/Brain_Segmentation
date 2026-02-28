#!/usr/bin/env bash
# ── 02 — Make splits + build slice index ─────────────────────────
# Step 1: Patient-level 80/10/10 split → splits.json
# Step 2: Per-slice JSONL index       → train/val/test.jsonl
#
# Prereqs: 00_label_probe.py + 01_preprocess.py must have run.
#
# Outputs:
#   {OUTPUT_ROOT}/configs/splits.json
#   {OUTPUT_ROOT}/index/{train,val,test}.jsonl
#   {OUTPUT_ROOT}/index/index_stats.json
#
# Usage:
#   bash scripts/02_make_index.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo ">>> Step 1/2: Make patient-level splits (80/10/10)"
python scripts/01_make_splits.py
echo ""

echo ">>> Step 2/2: Build per-slice index"
python scripts/02_build_slice_index.py
echo ""

echo ">>> Done. All index files ready."
