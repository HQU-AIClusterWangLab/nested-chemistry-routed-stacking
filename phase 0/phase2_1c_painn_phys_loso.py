# -*- coding: utf-8 -*-
"""
Phase 2.1c: PaiNN + Static Phys-Features LOSO.

This creates the static-phys PaiNN counterpart needed before comparing dynamic
physical features.
"""
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from phase2_phys_loso_common import (  # noqa: E402
    BATCH_SIZE_PAINN,
    PaiNNPhys,
    run_loso_experiment,
)


run_loso_experiment(
    model_name="PaiNNStaticPhys",
    model_class=PaiNNPhys,
    dataset_dir=r"D:\lunwen\2.1sci\phase 0\dataset\processed",
    output_dir=r"D:\lunwen\2.1sci\phase 2\loso_painn_phys_output",
    result_csv_name="loso_painn_phys_results.csv",
    bar_name="loso_painn_phys_bar.png",
    model_prefix="model_painn_phys_loso",
    batch_size=BATCH_SIZE_PAINN,
    baseline_csv=r"D:\lunwen\2.1sci\phase 1\loso_painn_output\loso_painn_results.csv",
    static_csv=None,
)
