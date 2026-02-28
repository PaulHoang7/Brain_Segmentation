"""
Weighted samplers for imbalanced slice data.

Supports oversampling by WT, TC, or ET tumour presence.
Works with the new JSONL schema: has_tumor_wt / has_tumor_tc / has_tumor_et.
"""
from __future__ import annotations

import json
from typing import List

import numpy as np
from torch.utils.data import WeightedRandomSampler

from .config import INDEX_DIR


def _load_flags(split: str, key: str) -> List[int]:
    """Load a binary flag column from the JSONL index."""
    path = INDEX_DIR / f"{split}.jsonl"
    flags: List[int] = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            flags.append(int(row.get(key, 0)))
    return flags


def make_tumor_oversampler(
    split: str = "train",
    target: str = "wt",
    pos_weight: float = 3.0,
) -> WeightedRandomSampler:
    """
    WeightedRandomSampler that oversamples tumour-positive slices.

    Args:
        split:      "train" / "val" / "test"
        target:     "wt", "tc", or "et"
        pos_weight: how much more likely a positive slice is drawn
                    (e.g. 3.0 → 3× oversampling of tumour slices)

    Returns:
        WeightedRandomSampler (replacement=True, len=dataset size)
    """
    key = f"has_tumor_{target.lower()}"
    flags = _load_flags(split, key)

    weights = np.array(
        [pos_weight if f else 1.0 for f in flags], dtype=np.float64)
    weights /= weights.sum()

    n_pos = sum(flags)
    n_neg = len(flags) - n_pos
    eff_ratio = (n_pos * pos_weight) / (n_pos * pos_weight + n_neg)
    print(f"[sampler] {split} target={target.upper()} "
          f"pos={n_pos} neg={n_neg} pos_weight={pos_weight:.1f} "
          f"eff_pos_ratio={eff_ratio:.1%}")

    return WeightedRandomSampler(
        weights=weights.tolist(),
        num_samples=len(flags),
        replacement=True,
    )


def make_pg_sampler(split: str = "train",
                    pos_weight: float = 3.0) -> WeightedRandomSampler:
    """PG sampler: oversample WT-positive slices."""
    return make_tumor_oversampler(split, target="wt", pos_weight=pos_weight)
