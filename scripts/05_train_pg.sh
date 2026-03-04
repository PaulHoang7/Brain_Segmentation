#!/usr/bin/env bash
# ── 05 — Train + Eval Prompt Generator ───────────────────────────
# Step 1: Train PG (ResNet18 9ch) with focal+SmoothL1+GIoU+temporal
# Step 2: Evaluate on val+test with EMA smoothing
#
# Output:
#   {OUTPUT_ROOT}/ckpt/pg/{best,last}.pth
#   {OUTPUT_ROOT}/logs/pg/train_log.csv
#   {OUTPUT_ROOT}/preds/pg/{val,test}/{case_id}.json
#   {OUTPUT_ROOT}/results/pg_metrics.csv
#
# Usage:
#   bash scripts/05_train_pg.sh                    # full training (from hparams.yaml)
#   bash scripts/05_train_pg.sh --epochs 2         # quick smoke test
#   EVAL_ONLY=1 bash scripts/05_train_pg.sh        # skip training, eval only
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "${EVAL_ONLY:-0}" != "1" ]; then
    echo ">>> Step 1/2: Train PG"
    python scripts/05_train_pg.py "$@"
    echo ""
fi

echo ">>> Step 2/2: Evaluate PG on val+test"
python scripts/06_eval_pg.py --split val,test
echo ""

echo ">>> Done."
