# -*- coding: utf-8 -*-
"""
Phase 3.5: chemistry/context policy test for UQ-guided branch selection.

This script does not retrain base models and does not use labels to construct
the policy. It reads Phase 3.4 outputs and evaluates a fixed policy based on
inference-time visible context:

  - whether the held-out system contains La
  - OOD/context type from unlabeled prediction/descriptor shift
  - OOD score magnitude

The goal is to test whether a simple chemistry-aware policy is more stable than
sample-level branch reliability when Phase 4 stacking is unstable.
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
OUTPUT_DIR = ROOT / "phase 3" / "phase3_5_chem_context_policy_uq_output"


BASELINE_STRATEGIES = [
    "simple_ensemble_mean",
    "oof_best_branch_fallback",
    "reliability_weighted",
    "conservative_fallback",
    "context_adaptive",
]


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def contains_la(system):
    return system.startswith("La")


def choose_policy(system, context_row):
    """Fixed policy from visible chemistry/context only."""
    has_la = contains_la(system)
    context = context_row["context"]
    ood_score = float(context_row["ood_score"])
    variance_ratio = float(context_row["variance_median_ratio"])
    branch_ratio = float(context_row["branch_range_median_ratio"])

    if has_la:
        if context == "severe_ood" or ood_score >= 3.0 or variance_ratio >= 3.0:
            return "oof_best_branch_fallback", "La severe-OOD -> robust OOF branch"
        return "reliability_weighted", "La near/moderate -> reliability weighted"

    if context == "near_id":
        return "reliability_weighted", "non-La near-ID -> reliability weighted"
    if context == "moderate_shift" and branch_ratio < 1.0:
        return "simple_ensemble_mean", "non-La moderate with low branch shift -> simple ensemble"
    return "simple_ensemble_mean", "non-La shifted/OOD -> avoid branch fallback"


def choose_policy_variant(system, context_row, variant):
    has_la = contains_la(system)
    context = context_row["context"]
    ood_score = float(context_row["ood_score"])
    variance_ratio = float(context_row["variance_median_ratio"])

    if variant == "chem_policy_v1":
        return choose_policy(system, context_row)

    if variant == "la_fallback_nonla_simple":
        if has_la and (context == "severe_ood" or ood_score >= 3.0 or variance_ratio >= 3.0):
            return "oof_best_branch_fallback", "La high-risk -> OOF fallback"
        if has_la:
            return "reliability_weighted", "La low-risk -> reliability weighted"
        return "simple_ensemble_mean", "non-La -> simple ensemble"

    if variant == "only_high_ood_fallback":
        if has_la and ood_score >= 10.0:
            return "oof_best_branch_fallback", "La extreme-OOD -> OOF fallback"
        if context == "near_id":
            return "reliability_weighted", "near-ID -> reliability weighted"
        return "simple_ensemble_mean", "otherwise -> simple ensemble"

    raise ValueError(f"Unknown policy variant: {variant}")


def build_policy_rows(strategy_rows, context_rows, variant):
    strategy_by_system = {}
    for row in strategy_rows:
        strategy_by_system.setdefault(row["system"], {})[row["strategy"]] = row
    context_by_system = {row["system"]: row for row in context_rows}

    policy_rows = []
    for system in sorted(strategy_by_system):
        chosen, reason = choose_policy_variant(system, context_by_system[system], variant)
        chosen_row = strategy_by_system[system][chosen]
        oracle = strategy_by_system[system]["oracle_best_branch"]
        policy_rows.append({
            "system": system,
            "policy_variant": variant,
            "chosen_strategy": chosen,
            "reason": reason,
            "contains_la": str(contains_la(system)),
            "context": context_by_system[system]["context"],
            "ood_score": context_by_system[system]["ood_score"],
            "mae": float(chosen_row["mae"]),
            "mae95": float(chosen_row["mae95"]),
            "oracle_gap_mae": float(chosen_row["mae"]) - float(oracle["mae"]),
            "oracle_mae": float(oracle["mae"]),
            "oracle_mae95": float(oracle["mae95"]),
        })
    return policy_rows


def summarize_policy(policy_rows):
    return {
        "policy_variant": policy_rows[0]["policy_variant"],
        "mean_mae": float(np.mean([r["mae"] for r in policy_rows])),
        "mean_mae95": float(np.mean([r["mae95"] for r in policy_rows])),
        "mean_oracle_gap_mae": float(np.mean([r["oracle_gap_mae"] for r in policy_rows])),
        "la_mean_mae": float(np.mean([r["mae"] for r in policy_rows if r["contains_la"] == "True"])),
        "nonla_mean_mae": float(np.mean([r["mae"] for r in policy_rows if r["contains_la"] == "False"])),
    }


def summarize_strategy(strategy_rows, strategy):
    rows = [r for r in strategy_rows if r["strategy"] == strategy]
    return {
        "policy_variant": strategy,
        "mean_mae": float(np.mean([float(r["mae"]) for r in rows])),
        "mean_mae95": float(np.mean([float(r["mae95"]) for r in rows])),
        "mean_oracle_gap_mae": float(np.mean([float(r["oracle_gap_mae"]) for r in rows])),
        "la_mean_mae": float(np.mean([float(r["mae"]) for r in rows if contains_la(r["system"])])),
        "nonla_mean_mae": float(np.mean([float(r["mae"]) for r in rows if not contains_la(r["system"])])),
    }


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_summary(summary_rows, out_dir):
    labels = [r["policy_variant"] for r in summary_rows]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    for ax, key, title in [
        (axes[0], "mean_mae", "Mean MAE"),
        (axes[1], "mean_mae95", "Mean MAE_95"),
        (axes[2], "mean_oracle_gap_mae", "Mean Oracle Gap"),
    ]:
        ax.bar(x, [r[key] for r in summary_rows], color="#1f77b4")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title(title)
        ax.set_ylabel("eV")
        ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    path = Path(out_dir) / "phase3_5_chem_context_policy_summary.png"
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_system_choices(policy_rows, strategy_rows, out_dir, variant):
    systems = [r["system"] for r in policy_rows]
    x = np.arange(len(systems))
    by_strategy = {}
    for row in strategy_rows:
        by_strategy.setdefault(row["strategy"], {})[row["system"]] = float(row["mae"])
    policy_mae = [r["mae"] for r in policy_rows]
    simple = [by_strategy["simple_ensemble_mean"][s] for s in systems]
    fallback = [by_strategy["oof_best_branch_fallback"][s] for s in systems]
    reliability = [by_strategy["reliability_weighted"][s] for s in systems]

    fig, ax = plt.subplots(figsize=(14, 5))
    width = 0.20
    ax.bar(x - 1.5 * width, simple, width, label="simple mean")
    ax.bar(x - 0.5 * width, fallback, width, label="OOF fallback")
    ax.bar(x + 0.5 * width, reliability, width, label="reliability weighted")
    ax.bar(x + 1.5 * width, policy_mae, width, label=variant)
    ax.set_xticks(x)
    ax.set_xticklabels(systems, rotation=45)
    ax.set_ylabel("MAE (eV)")
    ax.set_title(f"Phase3.5 policy choices: {variant}")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    path = Path(out_dir) / f"phase3_5_{variant}_system_mae.png"
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    strategy_csv = PHASE3_4_DIR / "phase3_4_context_adaptive_strategy_metrics.csv"
    context_csv = PHASE3_4_DIR / "phase3_4_system_context_metrics.csv"
    if not strategy_csv.exists() or not context_csv.exists():
        raise FileNotFoundError("Run phase3_4_context_adaptive_uq.py first.")

    strategy_rows = read_csv(strategy_csv)
    context_rows = read_csv(context_csv)
    variants = [
        "chem_policy_v1",
        "la_fallback_nonla_simple",
        "only_high_ood_fallback",
    ]
    all_policy_rows = []
    summary_rows = []
    for variant in variants:
        rows = build_policy_rows(strategy_rows, context_rows, variant)
        all_policy_rows.extend(rows)
        summary_rows.append(summarize_policy(rows))
        plot_system_choices(rows, strategy_rows, OUTPUT_DIR, variant)
        print(
            f"{variant}: MAE={summary_rows[-1]['mean_mae']:.4f}, "
            f"MAE95={summary_rows[-1]['mean_mae95']:.4f}, "
            f"gap={summary_rows[-1]['mean_oracle_gap_mae']:.4f}"
        )

    for strategy in BASELINE_STRATEGIES + ["oracle_best_branch"]:
        summary_rows.append(summarize_strategy(strategy_rows, strategy))

    policy_csv = OUTPUT_DIR / "phase3_5_chem_context_policy_per_system.csv"
    summary_csv = OUTPUT_DIR / "phase3_5_chem_context_policy_summary.csv"
    write_csv(policy_csv, all_policy_rows, [
        "system", "policy_variant", "chosen_strategy", "reason", "contains_la",
        "context", "ood_score", "mae", "mae95", "oracle_gap_mae",
        "oracle_mae", "oracle_mae95",
    ])
    write_csv(summary_csv, summary_rows, [
        "policy_variant", "mean_mae", "mean_mae95", "mean_oracle_gap_mae",
        "la_mean_mae", "nonla_mean_mae",
    ])
    with open(OUTPUT_DIR / "phase3_5_chem_context_policy_metrics.json", "w") as f:
        json.dump({
            "per_system": all_policy_rows,
            "summary": summary_rows,
        }, f, indent=2)
    plot_path = plot_summary(summary_rows, OUTPUT_DIR)
    print("\nComplete.")
    print(f"Per-system policy: {policy_csv}")
    print(f"Summary          : {summary_csv}")
    print(f"Summary plot     : {plot_path}")


if __name__ == "__main__":
    main()
