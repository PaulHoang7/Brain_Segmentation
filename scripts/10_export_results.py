#!/usr/bin/env python3
"""
[10] Export final tables and figures for the thesis.

Aggregates results from all experiments into paper-ready outputs.

Usage:
    python scripts/10_export_results.py

Output:
    OUTPUT_ROOT/results/main_table.csv          — baseline ladder + full method
    OUTPUT_ROOT/results/ablation_table.csv      — 6 ablations
    OUTPUT_ROOT/results/robustness_shift.csv    — Dice vs bbox shift
    OUTPUT_ROOT/results/robustness_scale.csv    — Dice vs bbox scale
    OUTPUT_ROOT/figures/qualitative_cases.png   — overlay examples
    OUTPUT_ROOT/figures/robustness_plot.png      — Dice vs error curve
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datn.config import RESULTS_DIR, FIGURES_DIR, ensure_dirs


def main():
    ensure_dirs()

    # TODO: aggregate all CSVs into final tables
    # TODO: generate qualitative overlay figure (best/worst cases)
    # TODO: generate robustness plot (Dice vs bbox error)
    raise NotImplementedError(
        "Collect results from all experiments, merge into final "
        "CSV tables and generate figures for the thesis."
    )


if __name__ == "__main__":
    main()
