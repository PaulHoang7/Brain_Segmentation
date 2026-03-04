# BraTS-SAM-LoRA

Brain tumor segmentation on BraTS-GLI 2023 using SAM ViT-B + LoRA with a
learned Prompt Generator (2.5D bbox).

## Setup

```bash
pip install -r requirements.txt

# SAM (required from script 04 onwards)
pip install git+https://github.com/facebookresearch/segment-anything.git

# Download SAM ViT-B checkpoint
wget -P /mnt/nfs-data/tin_dataset/datn_outputs/ckpt/ \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

## Configuration

All paths are defined in **`configs/data.yaml`**:

```yaml
dataset_root: /mnt/nfs-data/tin_dataset/asnr-miccai-brats2023-gli-challenge-trainingdata
output_root:  /mnt/nfs-data/tin_dataset/datn_outputs
```

All hyperparameters are in **`configs/hparams.yaml`**.

Override per-machine with environment variables (yaml takes priority):
```bash
export OUTPUT_ROOT=/other/path
```

## Quickstart

Run scripts in order from the repo root:

```bash
# ── Phase 1: Preprocessing (Week 1-2) ──────────────────────────
python scripts/00_label_probe.py          # scan labels → configs/label_map.json
python scripts/01_make_splits.py          # patient split → configs/splits.json
python scripts/02_build_slice_index.py    # slice index  → index/{train,val,test}.jsonl
python scripts/03_sanity_check.py         # stats + debug images → debug/

# ── Phase 2: Vanilla SAM Baseline (Week 3-4) ───────────────────
python scripts/04_baseline_vanilla_sam.py # SAM + GT bbox → Dice/HD95

# ── Phase 3: Prompt Generator (Week 5-6) ───────────────────────
python scripts/05_train_pg.py             # train PG (ResNet18 9-ch)
python scripts/06_eval_pg.py              # P/R/F1, bbox IoU, stability

# ── Phase 4: SAM + LoRA (Week 7-10) ────────────────────────────
python scripts/07_train_sam_lora.py --target WT
python scripts/07_train_sam_lora.py --target TC
python scripts/07_train_sam_lora.py --target ET
python scripts/08_run_inference.py        # cascade PG→WT→TC→ET→postprocess
python scripts/09_eval_full.py            # Dice/HD95 on test set

# ── Phase 5: Export (Week 15-16) ───────────────────────────────
python scripts/10_export_results.py       # final tables + figures
```

## GitHub Tracking (Progress + Bugs)

This repo includes GitHub templates and a setup script for consistent tracking:

- Issue templates:
  - `.github/ISSUE_TEMPLATE/bug_report.yml`
  - `.github/ISSUE_TEMPLATE/progress_update.yml`
- PR template:
  - `.github/PULL_REQUEST_TEMPLATE.md`
- Workflow guide:
  - `docs/GITHUB_TRACKING.md`

Optional one-time setup for labels and milestones (requires GitHub CLI):

```bash
gh auth login
pwsh ./scripts/11_setup_github_tracking.ps1 -Repo "PaulHoang7/Brain_Segmentation"
```

## Project Structure

```
configs/
  data.yaml              # paths (dataset_root, output_root)
  hparams.yaml           # all hyperparameters

datn/                    # library package
  config.py              # central config (reads yaml → exports constants)
  io.py                  # NIfTI loading
  norm.py                # z-score normalisation
  prompts.py             # bbox utilities (tight, jitter, IoU)
  sam_preprocess.py      # ResizeLongestSide + pad + coord transforms
  datasets.py            # PGDataset (9-ch) + SAMDataset (3-ch 1024)
  samplers.py            # tumor oversampling
  pg_model.py            # Prompt Generator (ResNet18)
  lora.py                # LoRA injection for SAM
  losses.py              # Dice, Focal, GIoU, temporal, composite
  metrics.py             # Dice, HD95, PG detection/bbox/stability
  postprocess.py         # hierarchy enforce, CC removal, bbox smoothing
  inference.py           # cascade pipeline (PG → SAM_WT → TC → ET)

scripts/
  00_label_probe.py      # detect label scheme
  01_make_splits.py      # patient-level 70/15/15 split
  02_build_slice_index.py  # per-slice JSONL index
  03_sanity_check.py     # stats + visual QA
  04_baseline_vanilla_sam.py  # zero-shot SAM baseline
  05_train_pg.py         # train prompt generator
  06_eval_pg.py          # evaluate PG
  07_train_sam_lora.py   # train SAM+LoRA (per target)
  08_run_inference.py    # full cascade inference
  09_eval_full.py        # Dice/HD95 evaluation
  10_export_results.py   # paper tables + figures
```

## Output Layout (all under OUTPUT_ROOT)

```
datn_outputs/
  configs/     label_map.json, splits.json
  index/       train.jsonl, val.jsonl, test.jsonl
  logs/        TensorBoard logs per run
  ckpt/        sam_vit_b_01ec64.pth, pg_best.pth, sam_lora_*.pth
  preds/       vanilla_sam/, pg_boxes/, full_method/
  results/     main_table.csv, pg_metrics.csv, ablation_table.csv, ...
  figures/     qualitative_cases.png, robustness_plot.png
  debug/       sanity check images
```

## Key Conventions

- **Label mapping**: always read from `configs/label_map.json`, never hardcoded
- **Splits**: single `configs/splits.json` shared by all experiments
- **SAM input**: 3-ch `[FLAIR, T1c, T2w]` → ResizeLongestSide(1024) → pad 1024x1024
- **PG input**: 9-ch (3 mods x 3 slices) → resize 224x224
- **Normalisation**: z-score per volume over non-zero voxels
- **Seed**: 42 (fixed everywhere)
