"""
Central configuration — single source of truth.

Resolution order for paths:
  1. configs/data.yaml  (repo-level, checked into git)
  2. env vars DATASET_ROOT / OUTPUT_ROOT  (per-machine override)
  3. built-in defaults  (fallback)

All downstream code imports constants from this module.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

# ── Locate repo root & yaml ───────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parent.parent
_DATA_YAML = REPO_ROOT / "configs" / "data.yaml"
_HP_YAML   = REPO_ROOT / "configs" / "hparams.yaml"


def _load_yaml(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


_data_cfg = _load_yaml(_DATA_YAML)
_hp       = _load_yaml(_HP_YAML)

# ── Roots (yaml > env > default) ──────────────────────────────────
_ds_base = Path(
    _data_cfg.get("dataset_root",
                  os.environ.get("DATASET_ROOT",
                                 "/mnt/nfs-data/tin_dataset/"
                                 "asnr-miccai-brats2023-gli-challenge-trainingdata"))
)
_ds_sub = _data_cfg.get("dataset_subdir", "")
DATASET_ROOT = (_ds_base / _ds_sub) if _ds_sub else _ds_base

OUTPUT_ROOT = Path(
    _data_cfg.get("output_root",
                  os.environ.get("OUTPUT_ROOT",
                                 "/mnt/nfs-data/tin_dataset/datn_outputs"))
)

# ── Derived output dirs ───────────────────────────────────────────
CONFIGS_DIR   = OUTPUT_ROOT / "configs"
INDEX_DIR     = OUTPUT_ROOT / "index"
DATA_DIR      = OUTPUT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
SEG_DIR       = DATA_DIR / "seg"
LOGS_DIR      = OUTPUT_ROOT / "logs"
CKPT_DIR      = OUTPUT_ROOT / "ckpt"
PREDS_DIR     = OUTPUT_ROOT / "preds"
RESULTS_DIR   = OUTPUT_ROOT / "results"
FIGURES_DIR   = OUTPUT_ROOT / "figures"
DEBUG_DIR     = OUTPUT_ROOT / "debug"

CASES_CSV      = CONFIGS_DIR / "cases.csv"
SPLITS_JSON    = CONFIGS_DIR / "splits.json"
LABEL_MAP_JSON = CONFIGS_DIR / "label_map.json"

# ── Modalities ────────────────────────────────────────────────────
_data_hp       = _hp.get("data", {})
MODALITIES     = tuple(_data_hp.get("modalities", ["t2f", "t1c", "t2w"]))
ALL_MODALITIES = tuple(_data_hp.get("all_modalities", ["t1n", "t1c", "t2w", "t2f"]))
BBOX_PAD       = _data_hp.get("bbox_pad", 5)

# ── Reproducibility ──────────────────────────────────────────────
SEED = _hp.get("seed", 42)

# ── SAM defaults ─────────────────────────────────────────────────
_sam_hp      = _hp.get("sam", {})
SAM_IMG_SIZE = _sam_hp.get("img_size", 1024)
PIXEL_MEAN   = tuple(_sam_hp.get("pixel_mean", [123.675, 116.28, 103.53]))
PIXEL_STD    = tuple(_sam_hp.get("pixel_std", [58.395, 57.12, 57.375]))

# ── PG defaults ──────────────────────────────────────────────────
_pg_hp            = _hp.get("pg", {})
PG_CONTEXT_SLICES = 3   # always z-1, z, z+1
PG_IMG_SIZE       = _pg_hp.get("img_size", 224)


# ── Helpers ──────────────────────────────────────────────────────
@dataclass
class SplitCfg:
    train_ratio: float = 0.80
    val_ratio:   float = 0.10
    test_ratio:  float = 0.10
    seed:        int   = SEED


def ensure_dirs() -> None:
    """Create all output directories (idempotent)."""
    for d in (CONFIGS_DIR, INDEX_DIR, DATA_DIR, PROCESSED_DIR, SEG_DIR,
              LOGS_DIR, CKPT_DIR, PREDS_DIR, RESULTS_DIR, FIGURES_DIR, DEBUG_DIR):
        d.mkdir(parents=True, exist_ok=True)


def load_label_map() -> dict:
    """Load configs/label_map.json (must exist after label_probe)."""
    with open(LABEL_MAP_JSON) as f:
        return json.load(f)


def load_hparams() -> dict:
    """Return full hparams dict (from configs/hparams.yaml)."""
    return _hp.copy()


def seed_everything(seed: int = SEED) -> None:
    """Set seed for Python, NumPy, and PyTorch (CPU+CUDA) reproducibility."""
    import random
    random.seed(seed)

    import numpy as _np
    _np.random.seed(seed)

    try:
        import torch as _torch
        _torch.manual_seed(seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(seed)
            _torch.backends.cudnn.deterministic = True
            _torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
