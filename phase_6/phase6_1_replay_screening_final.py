# -*- coding: utf-8 -*-
"""Phase 6.1 replay screening for final nested-router predictions."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(r"D:\lunwen\2.1sci")
DEFAULT_PRED_DIR = ROOT / "phase 6" / "final_nested_router" / "00_final_predictions"
DEFAULT_OUT_DIR = ROOT / "phase 6" / "final_nested_router" / "01_replay_screening"

MODEL_COLUMNS = {
    "nested_router": "pred_nested_router",
    "phase4_1_gate": "pred_phase4_1_gate",
    "phase4_2_policy": "pred_phase4_2_policy",
    "simple_ensemble": "pred_simple_ensemble",
    "oof_best_fallback": "pred_oof_best_fallback",
    "reliability_weighted": "pred_reliability_weighted",
    "schnet_static_phys": "pred_schnet_static_phys",
    "paa_schnet_coord": "pred_paa_schnet_coord",
    "painn_coord_bond": "pred_painn_coord_bond",
}


def read_rows(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key, val in list(row.items()):
            try:
                row[key] = float(val)
            except (TypeError, ValueError):
                pass
    return rows


def rank_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        return float("nan")
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def metrics_for(rows, model, pred_col, ks, deltas):
    y = np.asarray([r["y_true"] for r in rows], dtype=float)
    pred = np.asarray([r[pred_col] for r in rows], dtype=float)
    rel_y = y - np.min(y)
    order = np.argsort(pred)
    abs_err = np.abs(pred - y)
    out = []
    for k in ks:
        kk = min(k, len(rows))
        top = order[:kk]
        for delta in deltas:
            ref = np.where(rel_y <= delta)[0]
            hit = np.intersect1d(top, ref)
            hit_mask = rel_y[top] <= delta
            out.append({
                "system": rows[0]["system_id"],
                "model": model,
                "K": kk,
                "delta_e_threshold": delta,
                "n_reference_low_energy": int(len(ref)),
                "recall_at_k": float(len(hit) / max(1, len(ref))),
                "best_of_k_gap": float(np.min(rel_y[top])),
                "budget_to_hit": int(np.where(hit_mask)[0][0] + 1) if np.any(hit_mask) else -1,
                "top_k_contamination": float(np.mean(rel_y[top] > delta)),
                "spearman_pred_vs_true": rank_corr(pred, y),
                "mae": float(abs_err.mean()),
                "mae95": float(np.mean(np.sort(abs_err)[-max(1, int(len(abs_err) * 0.05)):])),
            })
    return out


def plot_system(rows, out_dir):
    system = rows[0]["system_id"]
    y = np.asarray([r["y_true"] for r in rows], dtype=float)
    rel_y = y - np.min(y)
    pred = np.asarray([r["pred_nested_router"] for r in rows], dtype=float)
    order = np.argsort(pred)
    uq = np.asarray([r.get("ensemble_variance", np.nan) for r in rows], dtype=float)
    abs_err = np.abs(pred - y)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sc = axes[0].scatter(np.arange(len(order)), rel_y[order], c=uq[order], s=8, cmap="viridis", alpha=0.75)
    axes[0].set_title(f"{system}: nested-router ranking")
    axes[0].set_xlabel("Predicted rank")
    axes[0].set_ylabel("DFT relative energy (eV)")
    fig.colorbar(sc, ax=axes[0], label="ensemble variance")

    axes[1].scatter(pred, y, c=abs_err, s=8, cmap="magma", alpha=0.75)
    axes[1].set_title(f"{system}: prediction vs DFT")
    axes[1].set_xlabel("Predicted energy")
    axes[1].set_ylabel("DFT energy")
    plt.tight_layout()
    path = out_dir / f"phase6_replay_{system}.png"
    plt.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", default=str(DEFAULT_PRED_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--k", default="10,25,50,100")
    parser.add_argument("--delta", default="0.05,0.10,0.20")
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(pred_dir.glob("phase6_final_predictions_*.csv"))
    if not files:
        raise FileNotFoundError(f"No phase6_final_predictions_*.csv found in {pred_dir}. Run phase6_0 first.")
    ks = [int(x) for x in args.k.split(",") if x.strip()]
    deltas = [float(x) for x in args.delta.split(",") if x.strip()]

    all_metrics = []
    for path in files:
        rows = read_rows(path)
        for model, col in MODEL_COLUMNS.items():
            if col in rows[0]:
                all_metrics.extend(metrics_for(rows, model, col, ks, deltas))
        plot_system(rows, out_dir)
    write_csv(out_dir / "phase6_final_replay_metrics.csv", all_metrics)
    print(f"Wrote replay metrics to {out_dir / 'phase6_final_replay_metrics.csv'}")


if __name__ == "__main__":
    main()
