# datn — BraTS-GLI 2023 SAM+LoRA thesis pipeline
"""
Submodules:
  config          Central paths + hyperparams (from yaml/env)
  io              NIfTI loading
  norm            Z-score normalisation
  prompts         Bbox utilities
  sam_preprocess  SAM resize/pad/coord transforms
  datasets        PGDataset + SAMDataset
  samplers        Tumor oversampling
  pg_model        Prompt Generator (ResNet18)
  lora            LoRA injection for SAM
  losses          All loss functions
  metrics         Dice, HD95, PG metrics
  postprocess     Hierarchy enforcement, CC cleanup
  inference       Cascade pipeline
"""
