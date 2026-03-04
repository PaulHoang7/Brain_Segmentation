#!/usr/bin/env bash
# ── 00 — Dataset Scan + Label Probe ──────────────────────────────
# Scans the full BraTS-GLI 2023 dataset and probes 30 random segs.
#
# Outputs (under OUTPUT_ROOT/configs/):
#   cases.csv            — full manifest (case_id, mods, shape, has_seg)
#   label_map.json       — WT/TC/ET mapping
#   label_probe_raw.json — per-case probe detail
#
# Usage:
#   bash scripts/00_label_probe.sh        # defaults: n=30, seed=42
#   bash scripts/00_label_probe.sh 50 99  # custom n and seed
set -euo pipefail
cd "$(dirname "$0")/.."

N=${1:-30}
SEED=${2:-42}

echo ">>> Running 00_label_probe.py  (n=$N, seed=$SEED)"
python scripts/00_label_probe.py --n "$N" --seed "$SEED"
echo ">>> Done."
