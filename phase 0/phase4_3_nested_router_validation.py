# -*- coding: utf-8 -*-
"""
Phase 4.3: nested LOSO validation for chemistry/context routing.

Goal:
  Validate whether a final hybrid router can choose between Phase4.1 gate and
  Phase4.2 chemistry/context policy without using the outer held-out labels.

For each outer held-out system:
  1. Treat the other systems as inner validation systems.
  2. Compare pre-defined routing rules on inner systems.
  3. Select the rule with lowest inner objective.
  4. Apply that fixed rule to the outer system.

This avoids selecting "La -> gate, non-La -> policy" directly from the outer
test result.
"""
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(r"D:\lunwen\2.1sci")
PHASE4_1_CSV = ROOT / "phase 4" / "phase4_1_final_branch_stacking_output" / "phase4_1_final_branch_stacking_results.csv"
PHASE4_2_CSV = ROOT / "phase 4" / "phase4_2_chem_context_policy_stacking_output" / "phase4_2_chem_context_policy_stacking_results.csv"
CONTEXT_CSV = ROOT / "phase 3" / "phase3_4_context_adaptive_uq_output" / "phase3_4_system_context_metrics.csv"
OUTPUT_DIR = ROOT / "phase 4" / "phase4_3_nested_router_validation_output"


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def contains_la(system):
    return system.startswith("La")


def to_float(row, key):
    return float(row[key])


def load_tables():
    if not PHASE4_1_CSV.exists():
        raise FileNotFoundError(f"Missing Phase4.1 results: {PHASE4_1_CSV}")
    if not PHASE4_2_CSV.exists():
        raise FileNotFoundError(f"Missing Phase4.2 policy results: {PHASE4_2_CSV}")
    if not CONTEXT_CSV.exists():
        raise FileNotFoundError(f"Missing context results: {CONTEXT_CSV}")
    p41 = {r["System"]: r for r in read_csv(PHASE4_1_CSV)}
    p42 = {r["System"]: r for r in read_csv(PHASE4_2_CSV)}
    ctx = {r["system"]: r for r in read_csv(CONTEXT_CSV)}
    systems = sorted(set(p41) & set(p42) & set(ctx))
    return systems, p41, p42, ctx


def candidate_rules():
    return {
        "all_gate": lambda system, ctx: "gate",
        "all_policy": lambda system, ctx: "policy",
        "la_gate_nonla_policy": lambda system, ctx: "gate" if contains_la(system) else "policy",
        "la_extreme_gate_else_policy": (
            lambda system, ctx: "gate"
            if contains_la(system) and float(ctx["ood_score"]) >= 10.0
            else "policy"
        ),
        "nearid_gate_else_policy": (
            lambda system, ctx: "gate" if ctx["context"] == "near_id" else "policy"
        ),
        "severe_policy_else_gate": (
            lambda system, ctx: "policy" if ctx["context"] == "severe_ood" else "gate"
        ),
        "nonla_policy_la_gate_if_ood": (
            lambda system, ctx: "gate"
            if contains_la(system) and float(ctx["ood_score"]) >= 3.0
            else "policy"
        ),
    }


def system_metrics(system, choice, p41, p42):
    if choice == "gate":
        return {
            "mae": to_float(p41[system], "Gated_MAE"),
            "mae95": to_float(p41[system], "Gated_MAE95"),
            "source": "phase4_1_gate",
        }
    if choice == "policy":
        return {
            "mae": to_float(p42[system], "Policy_MAE"),
            "mae95": to_float(p42[system], "Policy_MAE95"),
            "source": "phase4_2_policy",
        }
    raise ValueError(choice)


def objective(rows, alpha=0.25):
    return float(np.mean([r["mae"] + alpha * r["mae95"] for r in rows]))


def validate_router(alpha=0.25):
    systems, p41, p42, ctx = load_tables()
    rules = candidate_rules()
    outer_rows = []
    inner_rows = []
    for outer in systems:
        inner_systems = [s for s in systems if s != outer]
        scores = []
        for rule_name, rule in rules.items():
            per_inner = []
            for system in inner_systems:
                choice = rule(system, ctx[system])
                m = system_metrics(system, choice, p41, p42)
                per_inner.append({
                    "outer_system": outer,
                    "inner_system": system,
                    "rule": rule_name,
                    "choice": choice,
                    **m,
                })
            scores.append({
                "rule": rule_name,
                "inner_objective": objective(per_inner, alpha),
                "inner_mean_mae": float(np.mean([r["mae"] for r in per_inner])),
                "inner_mean_mae95": float(np.mean([r["mae95"] for r in per_inner])),
                "rows": per_inner,
            })
        scores = sorted(scores, key=lambda x: (x["inner_objective"], x["inner_mean_mae"]))
        selected = scores[0]
        inner_rows.extend(selected["rows"])
        choice = rules[selected["rule"]](outer, ctx[outer])
        m = system_metrics(outer, choice, p41, p42)
        outer_rows.append({
            "System": outer,
            "Selected_Rule": selected["rule"],
            "Selected_Choice": choice,
            "Source": m["source"],
            "MAE": m["mae"],
            "MAE95": m["mae95"],
            "Inner_Objective": selected["inner_objective"],
            "Inner_Mean_MAE": selected["inner_mean_mae"],
            "Inner_Mean_MAE95": selected["inner_mean_mae95"],
            "Contains_La": str(contains_la(outer)),
            "Context": ctx[outer]["context"],
            "OOD_Score": ctx[outer]["ood_score"],
            "Gate_MAE": to_float(p41[outer], "Gated_MAE"),
            "Gate_MAE95": to_float(p41[outer], "Gated_MAE95"),
            "Policy_MAE": to_float(p42[outer], "Policy_MAE"),
            "Policy_MAE95": to_float(p42[outer], "Policy_MAE95"),
        })
    return outer_rows, inner_rows


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_outputs(outer_rows, inner_rows):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outer_csv = OUTPUT_DIR / "phase4_3_nested_router_outer_results.csv"
    inner_csv = OUTPUT_DIR / "phase4_3_nested_router_selected_inner_results.csv"
    summary_csv = OUTPUT_DIR / "phase4_3_nested_router_summary.csv"
    write_csv(outer_csv, outer_rows, [
        "System", "Selected_Rule", "Selected_Choice", "Source",
        "MAE", "MAE95", "Inner_Objective", "Inner_Mean_MAE", "Inner_Mean_MAE95",
        "Contains_La", "Context", "OOD_Score",
        "Gate_MAE", "Gate_MAE95", "Policy_MAE", "Policy_MAE95",
    ])
    write_csv(inner_csv, inner_rows, [
        "outer_system", "inner_system", "rule", "choice", "mae", "mae95", "source",
    ])
    summary = {
        "mean_nested_router_mae": float(np.mean([r["MAE"] for r in outer_rows])),
        "mean_nested_router_mae95": float(np.mean([r["MAE95"] for r in outer_rows])),
        "mean_gate_mae": float(np.mean([r["Gate_MAE"] for r in outer_rows])),
        "mean_gate_mae95": float(np.mean([r["Gate_MAE95"] for r in outer_rows])),
        "mean_policy_mae": float(np.mean([r["Policy_MAE"] for r in outer_rows])),
        "mean_policy_mae95": float(np.mean([r["Policy_MAE95"] for r in outer_rows])),
        "n_gate_choices": int(sum(1 for r in outer_rows if r["Selected_Choice"] == "gate")),
        "n_policy_choices": int(sum(1 for r in outer_rows if r["Selected_Choice"] == "policy")),
    }
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    with open(OUTPUT_DIR / "phase4_3_nested_router_metrics.json", "w") as f:
        json.dump({"summary": summary, "outer": outer_rows, "inner_selected": inner_rows}, f, indent=2)
    return outer_csv, inner_csv, summary_csv, summary


def plot_outputs(outer_rows):
    systems = [r["System"] for r in outer_rows]
    x = np.arange(len(systems))
    width = 0.24
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    series = [
        ("Gate", "Gate_MAE", "Gate_MAE95"),
        ("Policy", "Policy_MAE", "Policy_MAE95"),
        ("Nested router", "MAE", "MAE95"),
    ]
    for idx, (label, mae_key, m95_key) in enumerate(series):
        offset = (idx - 1) * width
        axes[0].bar(x + offset, [r[mae_key] for r in outer_rows], width, label=label)
        axes[1].bar(x + offset, [r[m95_key] for r in outer_rows], width, label=label)
    for ax, title in [(axes[0], "MAE"), (axes[1], "MAE_95")]:
        ax.set_xticks(x)
        ax.set_xticklabels(systems, rotation=45)
        ax.set_title(f"Phase4.3 nested router validation: {title}")
        ax.set_ylabel("eV")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    path = OUTPUT_DIR / "phase4_3_nested_router_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4))
    labels = [r["Selected_Rule"] for r in outer_rows]
    ax.bar(systems, [1 if r["Selected_Choice"] == "gate" else 0 for r in outer_rows], color="#1f77b4")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["policy", "gate"])
    ax.set_title("Nested-selected routing choice")
    ax.tick_params(axis="x", rotation=45)
    for i, label in enumerate(labels):
        ax.text(i, 0.5, label, rotation=90, ha="center", va="center", fontsize=7)
    plt.tight_layout()
    route_path = OUTPUT_DIR / "phase4_3_nested_router_choices.png"
    plt.savefig(route_path, dpi=150)
    plt.close(fig)
    return path, route_path


def main():
    outer_rows, inner_rows = validate_router(alpha=0.25)
    outer_csv, inner_csv, summary_csv, summary = write_outputs(outer_rows, inner_rows)
    plot_path, route_path = plot_outputs(outer_rows)
    print("=" * 80)
    print("Phase 4.3 Nested Router Validation")
    print(f"Mean nested router MAE   : {summary['mean_nested_router_mae']:.6f}")
    print(f"Mean nested router MAE95 : {summary['mean_nested_router_mae95']:.6f}")
    print(f"Mean gate MAE            : {summary['mean_gate_mae']:.6f}")
    print(f"Mean policy MAE          : {summary['mean_policy_mae']:.6f}")
    print(f"Outer results: {outer_csv}")
    print(f"Inner selected results: {inner_csv}")
    print(f"Summary: {summary_csv}")
    print(f"Plot: {plot_path}")
    print(f"Route plot: {route_path}")


if __name__ == "__main__":
    main()
