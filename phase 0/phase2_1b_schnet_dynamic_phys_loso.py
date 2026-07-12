# -*- coding: utf-8 -*-
"""
Phase 2.1b: SchNet + Dynamic Phys-Features LOSO.

Uses phase 0/dataset/processed_dynamic by default and compares against:
  - Phase 1.1 SchNet without physical features
  - Phase 2.1 SchNet with static physical features
"""
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from phase2_phys_loso_common import (  # noqa: E402
    BATCH_SIZE_SCHNET,
    SchNetPhys,
    run_loso_experiment,
)


run_loso_experiment(
    model_name="SchNetDynamicPhys",
    model_class=SchNetPhys,
    dataset_dir=r"D:\lunwen\2.1sci\phase 0\dataset\processed_dynamic",
    output_dir=r"D:\lunwen\2.1sci\phase 2\loso_schnet_dynamic_phys_output",
    result_csv_name="loso_schnet_dynamic_phys_results.csv",
    bar_name="loso_schnet_dynamic_phys_bar.png",
    model_prefix="model_schnet_dynamic_phys_loso",
    batch_size=BATCH_SIZE_SCHNET,
    baseline_csv=r"D:\lunwen\2.1sci\phase 1\loso_schnet_output\loso_results.csv",
    static_csv=r"D:\lunwen\2.1sci\phase 2\loso_schnet_phys_output\loso_schnet_phys_results.csv",
)
