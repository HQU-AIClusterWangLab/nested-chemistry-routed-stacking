# -*- coding: utf-8 -*-
"""Phase 4.1: final-branch no-UQ stacking from Phase3.2 predictions."""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from phase_final_branch_common import ROOT  # noqa: E402
from phase4_final_branch_stacking_common import (  # noqa: E402
    available_systems,
    load_phase3_fold,
    fold_metrics,
    write_results_csv,
)


OUTPUT_DIR = ROOT / "phase 4" / "phase4_1_final_branch_stacking_output"


def plot_comparison(results, out_dir):
    systems = [r["system"] for r in results]
    x = np.arange(len(systems))
    width = 0.20
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    for ax, suffix, title in [
        (axes[0], "mae", "MAE Energy"),
        (axes[1], "mae95", "MAE_95 Energy"),
    ]:
        ax.bar(x - 1.5 * width, [r[f"best_single_{suffix}"] for r in results], width, label="best single")
        ax.bar(x - 0.5 * width, [r[f"simple_mean_{suffix}"] for r in results], width, label="simple mean")
        ax.bar(x + 0.5 * width, [r[f"branch_mean_{suffix}"] for r in results], width, label="branch mean")
        ax.bar(x + 1.5 * width, [r[f"gated_{suffix}"] for r in results], width, label="gated stack")
        ax.set_xticks(x)
        ax.set_xticklabels(systems, rotation=45)
        ax.set_title(title)
        ax.set_ylabel("eV")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
    plt.tight_layout()
    path = Path(out_dir) / "phase4_1_final_branch_stacking_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    systems = available_systems()
    if not systems:
        raise FileNotFoundError(
            "No Phase3.2 folds found. Run first:\n"
            '  python "phase 0\\phase3_2_final_branch_uq.py"'
        )
    results = []
    for system in systems:
        print(f"Phase4.1 stacking fold: {system}")
        oof_rows, test_rows = load_phase3_fold(system)
        results.append(fold_metrics(system, oof_rows, test_rows, include_uq=False))

    csv_path = OUTPUT_DIR / "phase4_1_final_branch_stacking_results.csv"
    write_results_csv(csv_path, results)
    plot_path = plot_comparison(results, OUTPUT_DIR)

    print("\nPHASE 4.1 FINAL-BRANCH STACKING")
    for key in ["best_single_mae", "simple_mean_mae", "branch_mean_mae", "gated_mae"]:
        print(f"  {key}: {np.mean([r[key] for r in results]):.6f}")
    print(f"Results: {csv_path}")
    print(f"Plot   : {plot_path}")


if __name__ == "__main__":
    main()
