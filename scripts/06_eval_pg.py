#!/usr/bin/env python3
"""
[06] Evaluate / Infer Prompt Generator (PG).

1. Load best PG checkpoint
2. Run inference on val/test split (all slices)
3. Apply EMA smoothing along z per case
4. Export per-case JSON predictions → {OUTPUT_ROOT}/preds/pg/{case_id}.json
5. Compute metrics (P/R/F1, bbox IoU, stability) → results/pg_metrics.csv

Usage:
    python scripts/06_eval_pg.py                           # eval on val+test
    python scripts/06_eval_pg.py --split test              # eval on test only
    python scripts/06_eval_pg.py --no-smooth               # skip EMA smoothing
    python scripts/06_eval_pg.py --ckpt path/to/model.pth  # specific checkpoint

Output:
    {OUTPUT_ROOT}/preds/pg/{split}/{case_id}.json
    {OUTPUT_ROOT}/results/pg_metrics.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datn.config import (
    CKPT_DIR, PREDS_DIR, RESULTS_DIR, SEED,
    ensure_dirs, load_hparams, seed_everything,
)
from datn.datasets import PGDataset2p5D
from datn.metrics import pg_bbox_iou, pg_detection_metrics, pg_stability
from datn.pg_model import PromptGenerator
from datn.smoothing import ema_smooth_boxes, ema_smooth_objectness


# ── Run inference on a full split ────────────────────────────────
@torch.no_grad()
def run_inference(model, loader, device) -> list[dict]:
    """Returns list of {case_id, z, obj_prob, bbox_pred, obj_gt, bbox_gt}."""
    model.eval()
    results = []
    for batch in loader:
        img = batch["image"].to(device)
        out = model(img)
        obj_prob = torch.sigmoid(out["objectness"]).cpu().numpy()
        bbox_pred = out["bbox"].cpu().numpy()

        for i in range(len(batch["case_id"])):
            results.append({
                "case_id":   batch["case_id"][i],
                "z":         int(batch["z"][i]),
                "obj_prob":  float(obj_prob[i]),
                "bbox_pred": bbox_pred[i].tolist(),
                "obj_gt":    float(batch["objectness"][i]),
                "bbox_gt":   batch["bbox"][i].numpy().tolist(),
            })
    return results


# ── Group by case and apply EMA smoothing ────────────────────────
def smooth_predictions(results: list[dict], alpha: float) -> list[dict]:
    """Apply per-case EMA smoothing to objectness + bbox."""
    cases = defaultdict(list)
    for r in results:
        cases[r["case_id"]].append(r)

    smoothed = []
    for cid, items in cases.items():
        items.sort(key=lambda x: x["z"])

        obj_probs = np.array([it["obj_prob"] for it in items])
        boxes = np.array([it["bbox_pred"] for it in items])

        obj_smooth = ema_smooth_objectness(obj_probs, alpha)
        has_tumor = (obj_smooth >= 0.5).astype(int)
        boxes_smooth = ema_smooth_boxes(boxes, has_tumor, alpha)

        for i, it in enumerate(items):
            it["obj_prob_raw"] = it["obj_prob"]
            it["obj_prob"] = float(obj_smooth[i])
            it["bbox_pred_raw"] = it["bbox_pred"]
            it["bbox_pred"] = boxes_smooth[i].tolist()
            it["has_tumor_pred"] = int(has_tumor[i])
            smoothed.append(it)

    return smoothed


# ── Export per-case JSON ─────────────────────────────────────────
def export_predictions(results: list[dict], out_dir: Path):
    cases = defaultdict(list)
    for r in results:
        cases[r["case_id"]].append(r)

    out_dir.mkdir(parents=True, exist_ok=True)
    for cid, items in cases.items():
        items.sort(key=lambda x: x["z"])
        with open(out_dir / f"{cid}.json", "w") as f:
            json.dump(items, f, indent=2)

    print(f"  Exported {len(cases)} case JSONs → {out_dir}")


# ── Bbox diagnostics ─────────────────────────────────────────────
def print_bbox_diagnostics(results: list[dict]):
    """Print bbox ranges and sample predictions for debugging."""
    all_bp = np.array([r["bbox_pred"] for r in results])
    pos = [r for r in results if r["obj_gt"] >= 0.5]
    all_bg = np.array([r["bbox_gt"] for r in pos]) if pos else None

    # Ordering check
    n_inv_x = int((all_bp[:, 0] > all_bp[:, 2]).sum())
    n_inv_y = int((all_bp[:, 1] > all_bp[:, 3]).sum())
    pred_areas = (np.maximum(0, all_bp[:, 2] - all_bp[:, 0]) *
                  np.maximum(0, all_bp[:, 3] - all_bp[:, 1]))

    print(f"\n  Bbox diagnostics ({len(results)} slices, {len(pos)} GT-pos):")
    print(f"    pred range: x1=[{all_bp[:,0].min():.3f},{all_bp[:,0].max():.3f}]  "
          f"y1=[{all_bp[:,1].min():.3f},{all_bp[:,1].max():.3f}]  "
          f"x2=[{all_bp[:,2].min():.3f},{all_bp[:,2].max():.3f}]  "
          f"y2=[{all_bp[:,3].min():.3f},{all_bp[:,3].max():.3f}]")
    print(f"    pred area : mean={pred_areas.mean():.4f}  "
          f"min={pred_areas.min():.4f}  max={pred_areas.max():.4f}")
    if n_inv_x or n_inv_y:
        print(f"    WARNING: inverted boxes — x1>x2: {n_inv_x}  y1>y2: {n_inv_y}")
    if all_bg is not None:
        gt_areas = ((all_bg[:, 2] - all_bg[:, 0]) *
                    (all_bg[:, 3] - all_bg[:, 1]))
        print(f"    GT   range: x1=[{all_bg[:,0].min():.3f},{all_bg[:,0].max():.3f}]  "
              f"y1=[{all_bg[:,1].min():.3f},{all_bg[:,1].max():.3f}]  "
              f"x2=[{all_bg[:,2].min():.3f},{all_bg[:,2].max():.3f}]  "
              f"y2=[{all_bg[:,3].min():.3f},{all_bg[:,3].max():.3f}]")
        print(f"    GT   area : mean={gt_areas.mean():.4f}  "
              f"min={gt_areas.min():.4f}  max={gt_areas.max():.4f}")

    # Print 10 sample pred vs GT for positive slices
    samples = pos[:10] if len(pos) >= 10 else pos
    if samples:
        print(f"\n    Sample predictions ({len(samples)} GT-positive slices):")
        for r in samples:
            bp = [f"{v:.3f}" for v in r["bbox_pred"]]
            bg = [f"{v:.3f}" for v in r["bbox_gt"]]
            print(f"      {r['case_id']} z={r['z']:3d}  "
                  f"pred=[{','.join(bp)}]  GT=[{','.join(bg)}]  "
                  f"obj={r['obj_prob']:.3f}")


# ── Compute all metrics ──────────────────────────────────────────
def compute_metrics(results: list[dict]) -> dict:
    obj_pred = np.array([r["obj_prob"] for r in results])
    obj_gt = np.array([r["obj_gt"] for r in results])

    det = pg_detection_metrics(obj_pred, obj_gt)

    all_bp = np.array([r["bbox_pred"] for r in results])
    all_bg = np.array([r["bbox_gt"] for r in results])

    # IoU on GT-positive slices (primary metric)
    pos_mask = obj_gt >= 0.5
    if pos_mask.any():
        iou_pos = pg_bbox_iou(all_bp[pos_mask], all_bg[pos_mask])
    else:
        iou_pos = 0.0

    # IoU on true-positive slices (detected correctly)
    tp_mask = (obj_pred >= 0.5) & (obj_gt >= 0.5)
    if tp_mask.any():
        iou_tp = pg_bbox_iou(all_bp[tp_mask], all_bg[tp_mask])
    else:
        iou_tp = 0.0

    # Stability per case
    cases = defaultdict(list)
    for r in results:
        cases[r["case_id"]].append(r)

    stab_centers, stab_areas = [], []
    for cid, items in cases.items():
        items.sort(key=lambda x: x["z"])
        boxes = np.array([it["bbox_pred"] for it in items])
        has_t = np.array([it["obj_prob"] >= 0.5 for it in items]).astype(int)
        s = pg_stability(boxes, has_t)
        stab_centers.append(s["mean_delta_center"])
        stab_areas.append(s["mean_delta_area"])

    return {
        **det,
        "bbox_iou_pos": iou_pos,
        "bbox_iou_tp": iou_tp,
        "mean_delta_center": float(np.mean(stab_centers)),
        "mean_delta_area": float(np.mean(stab_areas)),
        "n_slices": len(results),
        "n_cases": len(cases),
    }


# ── Main ─────────────────────────────────────────────────────────
def main(split: str = "val,test", ckpt: str = None,
         smooth: bool = True, batch_size: int = None,
         num_workers: int = 0):
    t0 = time.time()
    ensure_dirs()
    seed_everything(SEED)

    hp = load_hparams().get("pg", {})
    batch_size = batch_size or hp.get("batch_size", 64)
    alpha = hp.get("smoothing_alpha", 0.8)
    img_size = hp.get("img_size", 224)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load checkpoint
    ckpt_path = Path(ckpt) if ckpt else CKPT_DIR / "pg" / "best.pth"
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found at {ckpt_path}")
        print("Run scripts/05_train_pg.py first.")
        sys.exit(1)

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = PromptGenerator(
        in_channels=hp.get("input_channels", 9)).to(device)
    model.load_state_dict(state["model_state_dict"])
    print(f"[eval] Loaded: {ckpt_path}")
    print(f"[eval] Epoch {state.get('epoch', '?')}  "
          f"val_loss={state.get('val_loss', '?')}  "
          f"val_f1={state.get('val_f1', '?')}")
    print(f"[eval] Smooth={smooth}  alpha={alpha}  device={device}")
    print()

    pred_dir = PREDS_DIR / "pg"
    splits = [s.strip() for s in split.split(",")]
    all_rows = []

    for sp in splits:
        print(f"{'=' * 60}")
        print(f"Split: {sp}")
        print("=" * 60)

        ds = PGDataset2p5D(split=sp, img_size=img_size)
        nw = num_workers
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=nw, pin_memory=(nw > 0),
                            persistent_workers=(nw > 0))

        results = run_inference(model, loader, device)
        print(f"  Inferred {len(results)} slices")

        if smooth:
            results = smooth_predictions(results, alpha)
            print(f"  EMA smoothed (alpha={alpha})")

        print_bbox_diagnostics(results)

        export_predictions(results, pred_dir / sp)

        metrics = compute_metrics(results)
        print(f"\n  Precision      : {metrics['precision']:.4f}")
        print(f"  Recall         : {metrics['recall']:.4f}")
        print(f"  F1             : {metrics['f1']:.4f}")
        print(f"  Bbox IoU (pos) : {metrics['bbox_iou_pos']:.4f}  "
              f"(all GT-positive slices)")
        print(f"  Bbox IoU (TP)  : {metrics['bbox_iou_tp']:.4f}  "
              f"(true-positive slices only)")
        print(f"  Stability      : d_center={metrics['mean_delta_center']:.4f}  "
              f"d_area={metrics['mean_delta_area']:.6f}")
        print()

        all_rows.append({"split": sp, **metrics})

    # Save CSV
    csv_path = RESULTS_DIR / "pg_metrics.csv"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"[eval] Metrics → {csv_path}")
    print(f"[eval] Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate Prompt Generator")
    p.add_argument("--split", type=str, default="val,test")
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--no-smooth", dest="smooth", action="store_false")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=0)
    main(**vars(p.parse_args()))
