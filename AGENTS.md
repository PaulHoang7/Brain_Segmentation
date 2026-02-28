# AGENTS_MASTER.md — DATN BraTS2023 (GLI)  
## Research‑Grade Pipeline Spec + Engineering Scaffolding (4‑month plan)

> **One file to follow.** This merges your existing `AGENTS.md` (engineering + preprocessing scaffolding) with a full **research/thesis plan** (baselines, ablations, robustness, timeline, and paper‑level deliverables).  
> Main direction: **PG 2.5D (bbox + temporal consistency) + SAM ViT‑B + LoRA (encoder+decoder) + Cascade WT→TC→ET + nnU‑Net baseline**.

---

# PART I — ENGINEERING SPEC (from your AGENTS.md)

# AGENTS.md — DATN (BraTS-GLI) preprocessing & training scaffolding

## Project goal
We are building a research-grade pipeline for MRI brain tumor segmentation using:
- BraTS-GLI 2023 training data (NIfTI .nii.gz) with modalities: t1n, t1c, t2w, t2f(FLAIR), and seg.
- A SAM + LoRA segmentation backbone (2D slice-based).
- A Prompt Generator (PG) that predicts per-slice prompts (bbox + objectness) from 2.5D context.

Main deliverables:
1) Clean dataset/index pipeline (patient split, slice index, prompts).
2) Torch datasets + samplers for PG and SAM training.
3) SAM-compatible preprocessing (ResizeLongestSide + pad to 1024) and bbox coordinate transforms.
4) Training-ready dataloaders (not full training yet).

## Environment
- OS: Windows (paths like D:\DATN\...)
- Python: >=3.10 recommended
- GPU: RTX 5090 32GB (but code must run on CPU too for debugging)
- Prefer reproducibility: fixed seeds, deterministic splits.

## Dataset layout (current)
Dataset root contains folders like:
BraTS-GLI-xxxxx-xxx/
  BraTS-GLI-xxxxx-xxx-t1c.nii.gz
  BraTS-GLI-xxxxx-xxx-t1n.nii.gz
  BraTS-GLI-xxxxx-xxx-t2f.nii.gz
  BraTS-GLI-xxxxx-xxx-t2w.nii.gz
  BraTS-GLI-xxxxx-xxx-seg.nii.gz

We already create:
- splits.json (patient-level split)
- index/train.jsonl, index/val.jsonl, index/test.jsonl (slice-level rows)
Each jsonl row: {case_id, z, has_tumor, bbox|null, paths{...}}

## Conventions / definitions
- Whole tumor (WT) binary mask: seg > 0.
- Default SAM input channels: [FLAIR(t2f), T1c(t1c), T2w(t2w)] stacked as 3 channels.
- PG trains on 2.5D: slices z-1, z, z+1 (handle boundaries by clamping).
- PG label bbox should be tight bbox with optional fixed padding (e.g., pad=5).
- SAM training should use jittered bbox input for robustness (e.g., random shift/scale 5–20 px), but PG labels remain tight.

## Key preprocessing rules
- MRI intensity normalization: z-score per volume over non-zero voxels only.
- Never split by slice; always split by patient (already done).
- For SAM, use "ResizeLongestSide(1024)" behavior (scale keeping aspect ratio) then pad to 1024x1024.
- Bbox coordinates must be transformed consistently with image resize/pad.

## Coding style
- Keep code modular: put reusable utilities in `datn/` package.
- Type hints where easy.
- No hard-coded absolute paths inside library modules; accept root_dir / index paths via args/config.
- Avoid overengineering: focus on correctness + research reproducibility.

## Repo structure we want
datn/
  __init__.py
  io.py                # NIfTI loading, modality mapping
  norm.py              # z-score normalization
  prompts.py           # bbox, jitter, point sampling utilities
  sam_preprocess.py    # ResizeLongestSide + pad + coord transforms
  datasets.py          # Torch Dataset for PG and SAM
  samplers.py          # Weighted sampler / oversampling tumor slices
  config.py            # dataclass configs (paths, hyperparams)
scripts/
  make_splits.py
  build_slice_index.py
  debug_visualize_case.py
  sanity_check_index.py

## Testing / validation
Add a lightweight sanity script (no heavy tests):
- Print index stats (pos/neg ratio).
- Load 3 random samples and visualize: image channel + mask + bbox.
- Verify bbox transform after SAM preprocessing (draw on processed image).

## Output expectations
When implementing changes:
- Keep work reproducible (seeded).
- Provide a short usage example in README or in script docstring.
- Do not download datasets automatically.

---

# PART II — THESIS / RESEARCH MASTER PLAN (Paper‑level, 4 months)

## 1) Scope & Outcomes (what “done” looks like)

### Pipeline (1 line)
**BraTS2023 3D MRI → preprocess + slice index 2.5D → train Prompt Generator (PG) bbox 2.5D w/ temporal consistency → train SAM ViT‑B + LoRA (encoder+decoder) for WT/TC/ET → inference: PG bbox + z‑smoothing → SAM(WT)→refine bbox → SAM(TC)→refine bbox → SAM(ET) → enforce ET⊂TC⊂WT + 3D postprocess → metrics & ablation vs nnU‑Net.**

### Mandatory deliverables (paper‑level)
1) **Main results table** (WT/TC/ET, Dice + HD95): baseline ladder + nnU‑Net + full method  
2) **PG metrics table**: slice detection (P/R/F1), bbox IoU, stability (Δcenter/Δarea along z)  
3) **Ablation table** (minimum 6 ablations)  
4) **Robustness plot**: Dice vs bbox error (shift/scale)  
5) **Qualitative figure + error analysis**: overlay + failure modes

---

## 2) Definitions: WT / TC / ET (BraTS subregions)

### Meaning
- **WT (Whole Tumor)** = all tumor tissue (edema + core + enhancing)  
- **TC (Tumor Core)** = tumor core (no edema)  
- **ET (Enhancing Tumor)** = enhancing region

Hierarchy:
- **ET ⊂ TC ⊂ WT**

### Label mapping rule (MUST DO — avoid hardcoding)
Different BraTS variants may encode ET differently (e.g., 4 or 3).  
**Do not hardcode. Always run a Label Probe first.**

**Label Probe checklist**
- [ ] Sample 30 segmentation volumes  
- [ ] Print `unique labels` and voxel counts  
- [ ] Decide mapping for WT/TC/ET  
- [ ] Save to `configs/label_map.json`  
- [ ] All preprocess/train/eval reads mapping only from that json

Safe mapping patterns (after probe):
- `WT = seg > 0` (or seg in tumor label set)
- `TC = seg in core labels`
- `ET = seg == enhancing label`

---

## 3) Data protocol (to avoid “unfair comparison”)

- **Patient‑level split** only.  
- Single `splits.json` used by: PG, SAM+LoRA, nnU‑Net, all ablations.  
- Keep a stable evaluation script for Dice/HD95 WT/TC/ET.

Optional (nice for paper):
- 3 seeds for confidence intervals, if time.

---

## 4) Prompt Generator (PG) — 2.5D bbox with temporal consistency

### Input / output
- Input: 3 consecutive slices (z‑1, z, z+1) × 3 modalities → **9 channels**
- Output: `objectness` + bbox `(x1,y1,x2,y2)`

### Default model (stable + fast)
- **ResNet18** (conv1 modified to 9 channels) + simple heads

### Loss
- Objectness: **BCE weighted** or **Focal (γ=2)**
- Bbox: **SmoothL1 + (1 − GIoU)** (only when `has_tumor=1`)
- Temporal: **Huber / SmoothL1** between bbox_z and bbox_{z+1}  
  Apply only when both slices have tumor.

### Sampling
- Oversample tumor slices (ratio 1:2 or 1:3), keep enough negatives.

### Inference smoothing
- EMA along z: `alpha=0.8` (or median filter k=5)

### PG evaluation (must report)
- Slice detection: Precision / Recall / F1 on `has_tumor`
- Bbox IoU: mean IoU with GT bbox (tumor slices only)
- Stability: mean Δcenter and Δarea between z and z+1 (before/after smoothing)

---

## 5) SAM ViT‑B + LoRA (main model)

### Main choice
- **SAM ViT‑B** as primary.
- ViT‑H is optional “bonus” if you finish early.

### LoRA configuration (default)
- Attach LoRA to:
  - Image encoder attention (qkv + proj)
  - Mask decoder
- Rank: **16** (if underfit → 32)
- Alpha: **32**
- Dropout: **0.05–0.1**

### Optimization
- AdamW lr=1e‑4 (LoRA params), wd=0.01
- Cosine schedule + warmup 10% steps
- Loss: Dice + BCE (optional focal for ET imbalance)

### Prompt training (critical)
- **GT bbox jitter** (scale/shift ±10–20%) to match predicted bbox at inference.
- Optional curriculum: mix predicted boxes late in training.

---

## 6) WT/TC/ET strategy: Cascade + Hierarchical consistency (recommended)

### Training (3 LoRA adapters)
- Adapter WT: bbox from WT (jittered)
- Adapter TC: prompt bbox derived from WT mask bbox (train from GT‑WT)
- Adapter ET: prompt bbox derived from TC mask bbox (train from GT‑TC)

### Inference (per slice)
1) PG bbox (smoothed) → SAM_WT → WT mask  
2) refine bbox from WT mask → SAM_TC → TC mask  
3) refine bbox from TC mask → SAM_ET → ET mask

### Enforce hierarchy
- `ET = ET ∩ TC`
- `TC = (TC ∪ ET) ∩ WT`
- `WT = WT ∪ TC`

### 3D postprocess + fail‑safes
- Remove tiny CC; closing light if needed
- Propagate bbox from nearest positive slice if PG misses
- Fallback to smoothed PG bbox if refine collapses

---

## 7) Baselines (must include)

### Baseline ladder
1) Vanilla SAM + **GT bbox**  
2) SAM + LoRA + **GT bbox**  
3) SAM + LoRA + **heuristic bbox**  
4) Full method: **PG bbox + smoothing + cascade SAM+LoRA**

### External baseline
- **nnU‑Net 3D fullres** trained on same split

---

## 8) Ablations (minimum 6)

1) PG 2D vs PG 2.5D  
2) Temporal loss on/off  
3) Bbox smoothing on/off  
4) LoRA placement: encoder‑only vs decoder‑only vs both  
5) Bbox jitter: 0 vs 10 vs 20%  
6) Cascade refine on/off  

---

## 9) Robustness study

Inject bbox errors:
- Shift ±10px, ±20px  
- Scale ±10%, ±20%

Plot Dice(WT/TC/ET) vs bbox error.

---

## 10) 4‑month timeline (16 weeks)

- Weeks 1–2: preprocess + label probe + splits + eval
- Weeks 3–4: vanilla SAM baseline + end‑to‑end inference
- Weeks 5–6: PG 2.5D + smoothing + PG metrics table
- Weeks 7–8: SAM+LoRA WT stable
- Weeks 9–10: TC/ET + cascade + hierarchy enforcement
- Weeks 11–12: nnU‑Net baseline trained + evaluated
- Weeks 13–14: 6 ablations + robustness plots
- Weeks 15–16: final runs + qualitative + thesis write‑up

---

## 11) Reproducibility checklist
- [ ] fixed seeds
- [ ] save config per run + commit hash
- [ ] consistent splits.json across all runs
- [ ] logs (TensorBoard/W&B)
- [ ] scripts for one‑command reproduce

---

## 12) Output naming (recommended)
- `results/main_table.csv`
- `results/pg_metrics.csv`
- `results/ablation_table.csv`
- `results/robustness_shift.csv`, `results/robustness_scale.csv`
- `figures/qualitative_cases.png`
- `figures/robustness_plot.png`

---

## 13) Final principle
Keep it correct, fair, reproducible. Do not overengineer.
