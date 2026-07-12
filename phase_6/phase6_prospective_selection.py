# -*- coding: utf-8 -*-
"""
Phase 6.3: make finite-budget prospective DFT selection lists.

This script does not run DFT. It produces a candidate list using the planned
70/20/10 policy:
  70% lowest predicted energy
  20% low-energy but high-UQ points
  10% diversity fillers by coarse true-energy/rank spread when replay labels exist
"""
import argparse
import csv
from pathlib import Path

import numpy as np


DEFAULT_ROOT = Path(r"D:\lunwen\2.1sci")
DEFAULT_PRED_DIR = DEFAULT_ROOT / "phase 4" / "uq_gated_stacking_output"
DEFAULT_OUT_DIR = DEFAULT_ROOT / "phase 6" / "prospective_selection"


def read_rows(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in ["y_true", "pred_gate_uq", "uq_raw_variance", "pred_ensemble_mean"]:
            if key in row:
                row[key] = float(row[key])
    return rows


def add_unique(selected, candidate_indices, target_count):
    seen = {idx for idx in selected}
    for idx in candidate_indices:
        if idx not in seen:
            selected.append(idx)
            seen.add(idx)
        if len(selected) >= target_count:
            break
    return selected


def select_candidates(rows, budget):
    pred = np.asarray([r["pred_gate_uq"] for r in rows], dtype=float)
    uq = np.asarray([r.get("uq_raw_variance", 0.0) for r in rows], dtype=float)
    low_energy_order = np.argsort(pred)

    n_low = max(1, int(round(budget * 0.70)))
    n_high_uq = max(1, int(round(budget * 0.20)))
    n_diverse = max(0, budget - n_low - n_high_uq)

    selected = []
    selected = add_unique(selected, low_energy_order, n_low)

    low_pool = low_energy_order[:max(n_low, int(0.30 * len(rows)))]
    high_uq_in_low_pool = low_pool[np.argsort(-uq[low_pool])]
    selected = add_unique(selected, high_uq_in_low_pool, n_low + n_high_uq)

    if n_diverse > 0:
        rank_positions = np.linspace(0, len(low_energy_order) - 1, n_diverse * 4, dtype=int)
        diverse_candidates = low_energy_order[rank_positions]
        selected = add_unique(selected, diverse_candidates, budget)

    selected = add_unique(selected, low_energy_order, budget)
    selected = selected[:budget]

    labels = []
    for idx in selected:
        rank = int(np.where(low_energy_order == idx)[0][0]) + 1
        if rank <= n_low:
            reason = "low_predicted_energy"
        elif idx in set(high_uq_in_low_pool.tolist()):
            reason = "low_energy_high_uq"
        else:
            reason = "diversity_fill"
        labels.append((idx, reason, rank))
    return labels


def write_selection(system, rows, selected, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        fieldnames = [
            "selection_rank", "sample_id", "system_id", "group_id", "selection_reason",
            "predicted_rank", "pred_gate_uq", "uq_raw_variance", "y_true_replay",
            "replay_relative_energy"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        y = np.asarray([r.get("y_true", np.nan) for r in rows], dtype=float)
        rel = y - np.nanmin(y)
        for selection_rank, (idx, reason, predicted_rank) in enumerate(selected, start=1):
            row = rows[idx]
            writer.writerow({
                "selection_rank": selection_rank,
                "sample_id": row.get("sample_id", ""),
                "system_id": row.get("system_id", system),
                "group_id": row.get("group_id", ""),
                "selection_reason": reason,
                "predicted_rank": predicted_rank,
                "pred_gate_uq": row.get("pred_gate_uq", ""),
                "uq_raw_variance": row.get("uq_raw_variance", ""),
                "y_true_replay": row.get("y_true", ""),
                "replay_relative_energy": rel[idx] if np.isfinite(rel[idx]) else "",
            })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", default=str(DEFAULT_PRED_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--systems", default="LaCu12,LaSi9")
    parser.add_argument("--budget", type=int, default=30)
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.out_dir)
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]

    for system in systems:
        path = pred_dir / f"phase4_2_fold_predictions_{system}.csv"
        if not path.exists():
            print(f"Missing {path}; skip {system}. Run phase4_2 first.")
            continue
        rows = read_rows(path)
        selected = select_candidates(rows, args.budget)
        out_path = out_dir / f"phase6_dft_selection_{system}.csv"
        write_selection(system, rows, selected, out_path)
        print(f"Wrote {len(selected)} selected candidates for {system}: {out_path}")


if __name__ == "__main__":
    main()
