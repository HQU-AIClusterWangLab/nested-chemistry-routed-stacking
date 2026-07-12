# -*- coding: utf-8 -*-
"""Phase 6.2 select DFT candidates from final nested-router predictions."""
import argparse
import csv
from pathlib import Path

import numpy as np


ROOT = Path(r"D:\lunwen\2.1sci")
DEFAULT_PRED_DIR = ROOT / "phase 6" / "final_nested_router" / "00_final_predictions"
DEFAULT_OUT_DIR = ROOT / "phase 6" / "final_nested_router" / "02_dft_candidate_selection"


def read_rows(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key, value in list(row.items()):
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                pass
    return rows


def add_unique(selected, candidates, target):
    seen = set(selected)
    for idx in candidates:
        if int(idx) not in seen:
            selected.append(int(idx))
            seen.add(int(idx))
        if len(selected) >= target:
            break
    return selected


def select(rows, budget):
    pred = np.asarray([r["pred_nested_router"] for r in rows], dtype=float)
    variance = np.asarray([r.get("ensemble_variance", 0.0) for r in rows], dtype=float)
    branch_spread = np.asarray([
        max(r["pred_schnet_static_phys"], r["pred_paa_schnet_coord"], r["pred_painn_coord_bond"])
        - min(r["pred_schnet_static_phys"], r["pred_paa_schnet_coord"], r["pred_painn_coord_bond"])
        for r in rows
    ], dtype=float)
    risk = variance + branch_spread
    order = np.argsort(pred)
    n_low = max(1, int(round(budget * 0.70)))
    n_risk = max(1, int(round(budget * 0.20)))
    selected = []
    selected = add_unique(selected, order, n_low)
    low_pool = order[:max(n_low, int(0.30 * len(order)))]
    high_risk_low = low_pool[np.argsort(-risk[low_pool])]
    selected = add_unique(selected, high_risk_low, n_low + n_risk)
    spread_positions = np.linspace(0, len(order) - 1, max(1, budget - len(selected)) * 4, dtype=int)
    selected = add_unique(selected, order[spread_positions], budget)
    selected = add_unique(selected, order, budget)
    return selected[:budget], order, risk


def write_selection(system, rows, selected, order, risk, out_path):
    y = np.asarray([r.get("y_true", np.nan) for r in rows], dtype=float)
    rel = y - np.nanmin(y)
    rank_map = {idx: rank + 1 for rank, idx in enumerate(order)}
    with open(out_path, "w", newline="") as f:
        fields = [
            "selection_rank", "sample_id", "system_id", "selection_reason",
            "predicted_rank", "pred_nested_router", "risk_score", "ensemble_variance",
            "nested_source", "router_rule", "router_choice", "policy_choice", "robust_branch",
            "y_true_replay", "replay_relative_energy",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        n_low = max(1, int(round(len(selected) * 0.70)))
        n_risk = max(1, int(round(len(selected) * 0.20)))
        for sel_rank, idx in enumerate(selected, start=1):
            if sel_rank <= n_low:
                reason = "low_predicted_energy"
            elif sel_rank <= n_low + n_risk:
                reason = "low_energy_high_risk"
            else:
                reason = "diversity_fill"
            row = rows[idx]
            writer.writerow({
                "selection_rank": sel_rank,
                "sample_id": row["sample_id"],
                "system_id": system,
                "selection_reason": reason,
                "predicted_rank": rank_map[idx],
                "pred_nested_router": row["pred_nested_router"],
                "risk_score": risk[idx],
                "ensemble_variance": row.get("ensemble_variance", ""),
                "nested_source": row.get("nested_source", ""),
                "router_rule": row.get("router_rule", ""),
                "router_choice": row.get("router_choice", ""),
                "policy_choice": row.get("policy_choice", ""),
                "robust_branch": row.get("robust_branch", ""),
                "y_true_replay": row.get("y_true", ""),
                "replay_relative_energy": rel[idx] if np.isfinite(rel[idx]) else "",
            })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", default=str(DEFAULT_PRED_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--systems", default="LaCu12,LaSi9,BSe9")
    parser.add_argument("--budget", type=int, default=30)
    args = parser.parse_args()
    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for system in [s.strip() for s in args.systems.split(",") if s.strip()]:
        path = pred_dir / f"phase6_final_predictions_{system}.csv"
        if not path.exists():
            print(f"Missing {path}; skip")
            continue
        rows = read_rows(path)
        selected, order, risk = select(rows, args.budget)
        out_path = out_dir / f"phase6_final_dft_selection_{system}.csv"
        write_selection(system, rows, selected, order, risk, out_path)
        print(f"Wrote {len(selected)} candidates: {out_path}")


if __name__ == "__main__":
    main()
