# -*- coding: utf-8 -*-
"""
Phase 4.2: chemistry/context-aware reliability policy stacking.

This is the Phase 4 promotion of the best Phase 3.5 policy:

  only_high_ood_fallback

It does not retrain base models. It consumes Phase 3.2 prediction tables plus
Phase 3.4 context diagnostics and applies a fixed, label-free policy:

  - near-ID systems: reliability-weighted branch ensemble
  - La extreme-OOD systems: OOF-best-branch fallback
  - otherwise: simple ensemble mean

The policy uses inference-time visible chemistry/context information only.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(r"D:\lunwen\2.1sci")


PHASE3_4_DIR = ROOT / "phase 3" / "phase3_4_context_adaptive_uq_output"
PHASE3_5_DIR = ROOT / "phase 3" / "phase3_5_chem_context_policy_uq_output"
PHASE4_1_DIR = ROOT / "phase 4" / "phase4_1_final_branch_stacking_output"
OUTPUT_DIR = ROOT / "phase 4" / "phase4_2_chem_context_policy_stacking_output"

POLICY_NAME = "only_high_ood_fallback"


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def contains_la(system):
    return system.startswith("La")


def choose_policy(system, context_row):
    context = context_row["context"]
    ood_score = float(context_row["ood_score"])
    if contains_la(system) and ood_score >= 10.0:
        return "oof_best_branch_fallback", "La extreme-OOD -> OOF-best-branch fallback"
    if context == "near_id":
        return "reliability_weighted", "near-ID -> reliability-weighted branch ensemble"
    return "simple_ensemble_mean", "otherwise -> simple ensemble mean"


def metric_pair(mae, mae95):
    return float(mae), float(mae95)


def build_results():
    strategy_csv = PHASE3_4_DIR / "phase3_4_context_adaptive_strategy_metrics.csv"
    context_csv = PHASE3_4_DIR / "phase3_4_system_context_metrics.csv"
    phase4_1_csv = PHASE4_1_DIR / "phase4_1_final_branch_stacking_results.csv"
    if not strategy_csv.exists() or not context_csv.exists():
        raise FileNotFoundError(
            "Missing Phase3.4 outputs. Run first:\n"
            '  python "phase 0\\phase3_4_context_adaptive_uq.py"'
        )
    if not phase4_1_csv.exists():
        raise FileNotFoundError(
            "Missing Phase4.1 final-branch results. Run first:\n"
            '  python "phase 0\\phase4_1_final_branch_stacking.py"'
        )

    strategy_rows = read_csv(strategy_csv)
    context_rows = {row["system"]: row for row in read_csv(context_csv)}
    phase4_1_rows = {row["System"]: row for row in read_csv(phase4_1_csv)}

    by_system_strategy = {}
    for row in strategy_rows:
        by_system_strategy.setdefault(row["system"], {})[row["strategy"]] = row

    results = []
    for system in sorted(by_system_strategy):
        chosen, reason = choose_policy(system, context_rows[system])
        chosen_row = by_system_strategy[system][chosen]
        oracle_row = by_system_strategy[system]["oracle_best_branch"]
        simple_row = by_system_strategy[system]["simple_ensemble_mean"]
        fallback_row = by_system_strategy[system]["oof_best_branch_fallback"]
        reliability_row = by_system_strategy[system]["reliability_weighted"]
        phase4_1_row = phase4_1_rows.get(system)
        results.append({
            "System": system,
            "Policy": POLICY_NAME,
            "Chosen_Strategy": chosen,
            "Reason": reason,
            "Contains_La": str(contains_la(system)),
            "Context": context_rows[system]["context"],
            "OOD_Score": context_rows[system]["ood_score"],
            "SimpleMean_MAE": float(simple_row["mae"]),
            "SimpleMean_MAE95": float(simple_row["mae95"]),
            "OOFBestFallback_MAE": float(fallback_row["mae"]),
            "OOFBestFallback_MAE95": float(fallback_row["mae95"]),
            "ReliabilityWeighted_MAE": float(reliability_row["mae"]),
            "ReliabilityWeighted_MAE95": float(reliability_row["mae95"]),
            "Policy_MAE": float(chosen_row["mae"]),
            "Policy_MAE95": float(chosen_row["mae95"]),
            "OracleBranch_MAE": float(oracle_row["mae"]),
            "OracleBranch_MAE95": float(oracle_row["mae95"]),
            "OracleGap_MAE": float(chosen_row["mae"]) - float(oracle_row["mae"]),
            "Phase4_1_Gated_MAE": float(phase4_1_row["Gated_MAE"]) if phase4_1_row else np.nan,
            "Phase4_1_Gated_MAE95": float(phase4_1_row["Gated_MAE95"]) if phase4_1_row else np.nan,
        })
    return results


def write_results(results):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result_csv = OUTPUT_DIR / "phase4_2_chem_context_policy_stacking_results.csv"
    fields = [
        "System", "Policy", "Chosen_Strategy", "Reason", "Contains_La", "Context", "OOD_Score",
        "SimpleMean_MAE", "SimpleMean_MAE95",
        "OOFBestFallback_MAE", "OOFBestFallback_MAE95",
        "ReliabilityWeighted_MAE", "ReliabilityWeighted_MAE95",
        "Policy_MAE", "Policy_MAE95",
        "OracleBranch_MAE", "OracleBranch_MAE95", "OracleGap_MAE",
        "Phase4_1_Gated_MAE", "Phase4_1_Gated_MAE95",
    ]
    with open(result_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    summary = {
        "policy": POLICY_NAME,
        "mean_simple_mae": float(np.mean([r["SimpleMean_MAE"] for r in results])),
        "mean_simple_mae95": float(np.mean([r["SimpleMean_MAE95"] for r in results])),
        "mean_oof_fallback_mae": float(np.mean([r["OOFBestFallback_MAE"] for r in results])),
        "mean_oof_fallback_mae95": float(np.mean([r["OOFBestFallback_MAE95"] for r in results])),
        "mean_reliability_weighted_mae": float(np.mean([r["ReliabilityWeighted_MAE"] for r in results])),
        "mean_reliability_weighted_mae95": float(np.mean([r["ReliabilityWeighted_MAE95"] for r in results])),
        "mean_policy_mae": float(np.mean([r["Policy_MAE"] for r in results])),
        "mean_policy_mae95": float(np.mean([r["Policy_MAE95"] for r in results])),
        "mean_oracle_branch_mae": float(np.mean([r["OracleBranch_MAE"] for r in results])),
        "mean_oracle_branch_mae95": float(np.mean([r["OracleBranch_MAE95"] for r in results])),
        "mean_oracle_gap_mae": float(np.mean([r["OracleGap_MAE"] for r in results])),
        "mean_phase4_1_gated_mae": float(np.nanmean([r["Phase4_1_Gated_MAE"] for r in results])),
        "mean_phase4_1_gated_mae95": float(np.nanmean([r["Phase4_1_Gated_MAE95"] for r in results])),
        "la_mean_policy_mae": float(np.mean([r["Policy_MAE"] for r in results if r["Contains_La"] == "True"])),
        "nonla_mean_policy_mae": float(np.mean([r["Policy_MAE"] for r in results if r["Contains_La"] == "False"])),
    }
    summary_csv = OUTPUT_DIR / "phase4_2_chem_context_policy_stacking_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    with open(OUTPUT_DIR / "phase4_2_chem_context_policy_stacking_metrics.json", "w") as f:
        json.dump({"summary": summary, "per_system": results}, f, indent=2)
    return result_csv, summary_csv, summary


def plot_results(results):
    systems = [r["System"] for r in results]
    x = np.arange(len(systems))
    width = 0.16
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    series = [
        ("Simple", "SimpleMean_MAE", "SimpleMean_MAE95"),
        ("OOF fallback", "OOFBestFallback_MAE", "OOFBestFallback_MAE95"),
        ("Reliability", "ReliabilityWeighted_MAE", "ReliabilityWeighted_MAE95"),
        ("Policy", "Policy_MAE", "Policy_MAE95"),
        ("Phase4.1 gate", "Phase4_1_Gated_MAE", "Phase4_1_Gated_MAE95"),
    ]
    for idx, (label, mae_key, mae95_key) in enumerate(series):
        offset = (idx - 2) * width
        axes[0].bar(x + offset, [r[mae_key] for r in results], width, label=label)
        axes[1].bar(x + offset, [r[mae95_key] for r in results], width, label=label)
    for ax, title in [(axes[0], "MAE"), (axes[1], "MAE_95")]:
        ax.set_xticks(x)
        ax.set_xticklabels(systems, rotation=45)
        ax.set_title(f"Phase4.2 chemistry/context policy: {title}")
        ax.set_ylabel("eV")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
    plt.tight_layout()
    path = OUTPUT_DIR / "phase4_2_chem_context_policy_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#d62728" if r["Contains_La"] == "True" else "#1f77b4" for r in results]
    ax.bar(systems, [r["OOD_Score"] for r in results], color=colors)
    ax.set_yscale("log")
    ax.set_ylabel("OOD score (log scale)")
    ax.set_title("System context used by Phase4.2 policy")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    context_path = OUTPUT_DIR / "phase4_2_chem_context_ood_scores.png"
    plt.savefig(context_path, dpi=150)
    plt.close(fig)
    return path, context_path


def main():
    results = build_results()
    result_csv, summary_csv, summary = write_results(results)
    plot_path, context_path = plot_results(results)
    print("=" * 80)
    print("Phase 4.2 Chemistry/Context Policy Stacking")
    print(f"Policy: {POLICY_NAME}")
    print(f"Mean Policy MAE    : {summary['mean_policy_mae']:.6f}")
    print(f"Mean Policy MAE_95 : {summary['mean_policy_mae95']:.6f}")
    print(f"Mean Oracle Gap MAE: {summary['mean_oracle_gap_mae']:.6f}")
    print(f"Mean Phase4.1 Gate MAE: {summary['mean_phase4_1_gated_mae']:.6f}")
    print(f"Results CSV : {result_csv}")
    print(f"Summary CSV : {summary_csv}")
    print(f"Plot        : {plot_path}")
    print(f"Context plot: {context_path}")


if __name__ == "__main__":
    main()
