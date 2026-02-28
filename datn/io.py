"""NIfTI loading + modality helpers."""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import nibabel as nib

from .config import DATASET_ROOT, ALL_MODALITIES


def case_dir(case_id: str, root: Path = DATASET_ROOT) -> Path:
    return root / case_id


def nifti_path(case_id: str, modality: str, root: Path = DATASET_ROOT) -> Path:
    return case_dir(case_id, root) / f"{case_id}-{modality}.nii.gz"


def load_volume(case_id: str, modality: str,
                root: Path = DATASET_ROOT) -> np.ndarray:
    """Load a single modality volume → float32 (H, W, D)."""
    path = nifti_path(case_id, modality, root)
    img = nib.load(str(path))
    return np.asarray(img.dataobj, dtype=np.float32)


def load_seg(case_id: str, root: Path = DATASET_ROOT) -> np.ndarray:
    """Load segmentation volume → int16 (H, W, D)."""
    path = case_dir(case_id, root) / f"{case_id}-seg.nii.gz"
    img = nib.load(str(path))
    return np.asarray(img.dataobj, dtype=np.int16)


def load_modalities(case_id: str,
                    modalities: tuple[str, ...] = ALL_MODALITIES,
                    root: Path = DATASET_ROOT) -> Dict[str, np.ndarray]:
    """Load multiple modalities as a dict {mod_name: volume}."""
    return {m: load_volume(case_id, m, root) for m in modalities}


def list_cases(root: Path = DATASET_ROOT) -> list[str]:
    """Return sorted list of case_id strings found in dataset root."""
    cases = sorted([
        d.name for d in root.iterdir()
        if d.is_dir() and d.name.startswith("BraTS-GLI-")
    ])
    return cases


def scan_case(case_id: str, root: Path = DATASET_ROOT) -> dict:
    """Return metadata dict for one case (modalities present, seg, shape)."""
    d = case_dir(case_id, root)
    mods_present = []
    for m in ALL_MODALITIES:
        if (d / f"{case_id}-{m}.nii.gz").exists():
            mods_present.append(m)
    has_seg = (d / f"{case_id}-seg.nii.gz").exists()

    # Get shape from first available modality (cheap header read)
    shape = None
    for m in mods_present:
        hdr = nib.load(str(d / f"{case_id}-{m}.nii.gz"))
        shape = tuple(int(s) for s in hdr.shape)
        break

    return {
        "case_id": case_id,
        "modalities": ",".join(mods_present),
        "n_modalities": len(mods_present),
        "has_seg": has_seg,
        "shape_H": shape[0] if shape else None,
        "shape_W": shape[1] if shape else None,
        "shape_D": shape[2] if shape else None,
    }
