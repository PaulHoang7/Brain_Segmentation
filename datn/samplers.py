"""
Weighted samplers for imbalanced slice data.
Tumor slices are oversampled relative to non-tumor slices.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import List

import numpy as np
from torch.utils.data import WeightedRandomSampler

from .config import INDEX_DIR


def make_tumor_oversampler(split: str = "train",
                           pos_weight: float = 3.0
                           ) -> WeightedRandomSampler:
    """
    Creates a WeightedRandomSampler that oversamples tumor slices.
    pos_weight: how much more likely a tumor slice is to be drawn
                (e.g., 3.0 → 3× oversampling).
    """
    index_path = INDEX_DIR / f"{split}.jsonl"
    labels: List[int] = []
    with open(index_path) as f:
        for line in f:
            row = json.loads(line)
            labels.append(row["has_tumor"])

    weights = np.array([pos_weight if l == 1 else 1.0 for l in labels],
                       dtype=np.float64)
    weights /= weights.sum()

    return WeightedRandomSampler(
        weights=weights.tolist(),
        num_samples=len(labels),
        replacement=True,
    )


def make_pg_sampler(split: str = "train",
                    pos_weight: float = 3.0) -> WeightedRandomSampler:
    """Alias — PG uses both pos and neg slices with oversampling."""
    return make_tumor_oversampler(split, pos_weight)
