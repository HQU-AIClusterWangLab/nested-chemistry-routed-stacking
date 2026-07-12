# -*- coding: utf-8 -*-
"""Phase 4.2: final-branch UQ-gated stacking from Phase3.2 predictions."""
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

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


OUTPUT_DIR = ROOT / "phase 4" / "phase4_2_final_branch_uq_gated_stacking_output"


def safe_spearman(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def top_enrichment(uq, err, frac=0.05):
    n = max(1, int(len(uq) * frac))
    idx = np.argsort(-np.asarray(uq))[:n]
    base = np.mean(err)
    return float(np.mean(np.asarray(err)[idx]) / base) if base else float("nan")


def plot_uq_vs_nouq(no_uq, uq, out_dir):
    systems = [r["system"] for r in uq]
    x = np.arange(len(systems))
    width = 0.28
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    for ax, key, title in [
        (axes[0], "gated_mae", "Gated Stack MAE"),
        (axes[1], "gated_mae95", "Gated Stack MAE_95"),
    ]:
        ax.bar(x - width / 2, [r[key] for r in no_uq], width, label="no UQ", color="#ff7f0e")
        ax.bar(x + width / 2, [r[key] for r in uq], width, label="+ calibrated UQ", color="#2ca02c")
        ax.set_xticks(x)
        ax.set_xticklabels(systems, rotation=45)
        ax.set_title(title)
        ax.set_ylabel("eV")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
    plt.tight_layout()
    path = Path(out_dir) / "phase4_2_final_branch_uq_vs_nouq.png"
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_uq_diagnostics(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "System", "RawVariance_Spearman_vs_GateError",
                "CalibratedUQ_Spearman_vs_GateError",
                "CalibratedUQ_Top5_Error_Enrichment",
                "RawVariance_Mean", "CalibratedUQ_Mean",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    systems = available_systems()
    if not systems:
        raise FileNotFoundError(
            "No Phase3.2 folds found. Run first:\n"
            '  python "phase 0\\phase3_2_final_branch_uq.py"'
        )

    no_uq_results = []
    uq_results = []
    diagnostics = []
    for system in systems:
        print(f"Phase4.2 UQ-gated fold: {system}")
        oof_rows, test_rows = load_phase3_fold(system)
        no_uq = fold_metrics(system, oof_rows, test_rows, include_uq=False)
        uq = fold_metrics(system, oof_rows, test_rows, include_uq=True)
        no_uq_results.append(no_uq)
        uq_results.append(uq)

        gate_err = np.abs(uq["gate_pred"] - uq["y_true"])
        diagnostics.append({
            "System": system,
            "RawVariance_Spearman_vs_GateError": f"{safe_spearman(uq['raw_var'], gate_err):.6f}",
            "CalibratedUQ_Spearman_vs_GateError": f"{safe_spearman(uq['uq'], gate_err):.6f}",
            "CalibratedUQ_Top5_Error_Enrichment": f"{top_enrichment(uq['uq'], gate_err):.6f}",
            "RawVariance_Mean": f"{float(np.mean(uq['raw_var'])):.10f}",
            "CalibratedUQ_Mean": f"{float(np.mean(uq['uq'])):.10f}",
        })

    no_uq_csv = OUTPUT_DIR / "phase4_2_final_branch_gate_no_uq_results.csv"
    uq_csv = OUTPUT_DIR / "phase4_2_final_branch_gate_plus_uq_results.csv"
    write_results_csv(no_uq_csv, no_uq_results)
    write_results_csv(uq_csv, uq_results)

    combined_csv = OUTPUT_DIR / "phase4_2_final_branch_uq_gated_results.csv"
    with open(combined_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "System", "BestSingle_MAE", "SimpleMean_MAE", "BranchMean_MAE",
                "GateNoUQ_MAE", "GateUQ_MAE", "Delta_MAE_UQ_minus_NoUQ",
                "BestSingle_MAE95", "SimpleMean_MAE95", "BranchMean_MAE95",
                "GateNoUQ_MAE95", "GateUQ_MAE95", "Delta_MAE95_UQ_minus_NoUQ",
            ],
        )
        writer.writeheader()
        for a, b in zip(no_uq_results, uq_results):
            writer.writerow({
                "System": b["system"],
                "BestSingle_MAE": f"{b['best_single_mae']:.6f}",
                "SimpleMean_MAE": f"{b['simple_mean_mae']:.6f}",
                "BranchMean_MAE": f"{b['branch_mean_mae']:.6f}",
                "GateNoUQ_MAE": f"{a['gated_mae']:.6f}",
                "GateUQ_MAE": f"{b['gated_mae']:.6f}",
                "Delta_MAE_UQ_minus_NoUQ": f"{b['gated_mae'] - a['gated_mae']:.6f}",
                "BestSingle_MAE95": f"{b['best_single_mae95']:.6f}",
                "SimpleMean_MAE95": f"{b['simple_mean_mae95']:.6f}",
                "BranchMean_MAE95": f"{b['branch_mean_mae95']:.6f}",
                "GateNoUQ_MAE95": f"{a['gated_mae95']:.6f}",
                "GateUQ_MAE95": f"{b['gated_mae95']:.6f}",
                "Delta_MAE95_UQ_minus_NoUQ": f"{b['gated_mae95'] - a['gated_mae95']:.6f}",
            })

    diag_csv = OUTPUT_DIR / "phase4_2_final_branch_uq_diagnostics.csv"
    write_uq_diagnostics(diag_csv, diagnostics)
    plot_path = plot_uq_vs_nouq(no_uq_results, uq_results, OUTPUT_DIR)

    print("\nPHASE 4.2 FINAL-BRANCH UQ-GATED STACKING")
    print(f"  Gate no-UQ MAE: {np.mean([r['gated_mae'] for r in no_uq_results]):.6f}")
    print(f"  Gate +UQ  MAE: {np.mean([r['gated_mae'] for r in uq_results]):.6f}")
    print(f"  Gate no-UQ MAE95: {np.mean([r['gated_mae95'] for r in no_uq_results]):.6f}")
    print(f"  Gate +UQ  MAE95: {np.mean([r['gated_mae95'] for r in uq_results]):.6f}")
    print(f"Combined results: {combined_csv}")
    print(f"Diagnostics     : {diag_csv}")
    print(f"Plot            : {plot_path}")


if __name__ == "__main__":
    main()
