"""
Weighted samplers for imbalanced slice data.

Supports oversampling by WT, TC, or ET tumour presence.
Works with the new JSONL schema: has_tumor_wt / has_tumor_tc / has_tumor_et.

CaseGroupedSampler: NFS-friendly — yields all slices of one case
contiguously so the VolumeCache has near-zero miss rate.
"""
from __future__ import annotations

import json
from typing import Iterator, List

import numpy as np
from torch.utils.data import Sampler, WeightedRandomSampler

from .config import INDEX_DIR


def _load_index_meta(split: str) -> list[dict]:
    """Load case_id + flags from JSONL index (lightweight)."""
    path = INDEX_DIR / f"{split}.jsonl"
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            rows.append({
                "case_id": r["case_id"],
                "has_tumor_wt": int(r.get("has_tumor_wt", 0)),
                "has_tumor_tc": int(r.get("has_tumor_tc", 0)),
                "has_tumor_et": int(r.get("has_tumor_et", 0)),
            })
    return rows


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Case-grouped sampler (NFS-friendly)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class CaseGroupedSampler(Sampler[int]):
    """
    Yields indices grouped by case_id. Cases are shuffled each epoch,
    but slices within a case are in z-order.

    This ensures the VolumeCache in the Dataset gets maximal re-use:
    after loading a case's 3 .npy files, ALL its slices are processed
    before moving to the next case.

    Supports tumour oversampling: cases with more tumour slices
    are sampled more frequently (via case-level weights).
    """

    def __init__(self, split: str = "train",
                 target: str = "wt",
                 pos_weight: float = 3.0,
                 seed: int = 42):
        rows = _load_index_meta(split)
        key = f"has_tumor_{target.lower()}"

        # Group indices by case_id, preserving z-order
        from collections import OrderedDict
        self._case_indices: dict[str, list[int]] = OrderedDict()
        for i, r in enumerate(rows):
            cid = r["case_id"]
            if cid not in self._case_indices:
                self._case_indices[cid] = []
            self._case_indices[cid].append(i)

        self._total = len(rows)
        self._case_ids = list(self._case_indices.keys())

        # Case-level weights: proportion of positive slices in each case
        self._case_weights = np.ones(len(self._case_ids), dtype=np.float64)
        for ci, cid in enumerate(self._case_ids):
            idxs = self._case_indices[cid]
            n_pos = sum(rows[i][key] for i in idxs)
            if n_pos > 0:
                self._case_weights[ci] = pos_weight

        self._rng = np.random.default_rng(seed)
        n_pos_cases = int((self._case_weights > 1).sum())
        n_neg_cases = len(self._case_ids) - n_pos_cases
        print(f"[sampler] CaseGrouped {split} target={target.upper()} "
              f"cases={len(self._case_ids)} "
              f"(pos_cases={n_pos_cases} neg_cases={n_neg_cases}) "
              f"pos_weight={pos_weight:.1f} total_slices={self._total}")

    def __iter__(self) -> Iterator[int]:
        # Weighted case sampling with replacement, then flatten
        probs = self._case_weights / self._case_weights.sum()
        chosen = self._rng.choice(
            len(self._case_ids), size=len(self._case_ids),
            replace=True, p=probs)
        indices = []
        for ci in chosen:
            cid = self._case_ids[ci]
            indices.extend(self._case_indices[cid])
        return iter(indices)

    def __len__(self) -> int:
        return self._total


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Original WeightedRandomSampler (slice-level, for non-NFS use)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def make_tumor_oversampler(
    split: str = "train",
    target: str = "wt",
    pos_weight: float = 3.0,
) -> WeightedRandomSampler:
    """Slice-level WeightedRandomSampler (high NFS cache-miss rate)."""
    rows = _load_index_meta(split)
    key = f"has_tumor_{target.lower()}"
    flags = [r[key] for r in rows]

    weights = np.array(
        [pos_weight if f else 1.0 for f in flags], dtype=np.float64)
    weights /= weights.sum()

    return WeightedRandomSampler(
        weights=weights.tolist(),
        num_samples=len(flags),
        replacement=True,
    )


def make_pg_sampler(split: str = "train",
                    pos_weight: float = 3.0) -> CaseGroupedSampler:
    """PG sampler: case-grouped, NFS-friendly, tumour-oversampled."""
    return CaseGroupedSampler(split, target="wt", pos_weight=pos_weight)


class DatasetCaseGroupedSampler(Sampler[int]):
    """
    Case-grouped sampler that works on an already-filtered dataset (e.g. SAMDataset).
    Groups dataset indices by case_id so _VolumeCache stays effective.
    Cases are shuffled each epoch; slices within a case stay in order.
    """

    def __init__(self, rows: list[dict], seed: int = 42):
        from collections import OrderedDict
        self._case_indices: dict[str, list[int]] = OrderedDict()
        for i, r in enumerate(rows):
            cid = r["case_id"]
            if cid not in self._case_indices:
                self._case_indices[cid] = []
            self._case_indices[cid].append(i)

        self._case_ids = list(self._case_indices.keys())
        self._total = len(rows)
        self._rng = np.random.default_rng(seed)
        print(f"[sampler] DatasetCaseGrouped: {len(self._case_ids)} cases, "
              f"{self._total} slices")

    def __iter__(self) -> Iterator[int]:
        order = self._rng.permutation(len(self._case_ids))
        indices = []
        for ci in order:
            cid = self._case_ids[ci]
            indices.extend(self._case_indices[cid])
        return iter(indices)

    def __len__(self) -> int:
        return self._total
