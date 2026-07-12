# -*- coding: utf-8 -*-
"""
Phase 6.4: physical explanation analysis.

Requires the same Python environment used for training because it reads .pt
files with torch. It links replay predictions to dynamic geometry features and
Mulliken charges saved by phase5_dynamic_phys_features.py.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch


DEFAULT_ROOT = Path(r"D:\lunwen\2.1sci")
DEFAULT_DYNAMIC = DEFAULT_ROOT / "phase 0" / "dataset" / "processed_dynamic"
DEFAULT_PRED_DIR = DEFAULT_ROOT / "phase 4" / "uq_gated_stacking_output"
DEFAULT_OUT_DIR = DEFAULT_ROOT / "phase 6" / "physical_explanation"


def sample_id_from_pt(path):
    name = Path(path).stem
    return name[:-7] if name.endswith("_sample") else name


def spearman_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def summarize_pt(path):
    data = torch.load(path, weights_only=False)
    row = {"sample_id": sample_id_from_pt(path), "system_id": path.parent.name}
    if hasattr(data, "dynamic_phys"):
        dyn = data.dynamic_phys.float().cpu().numpy()
    else:
        dyn = data.x[:, 5:10].float().cpu().numpy() if data.x.shape[1] >= 10 else np.full((data.x.shape[0], 5), np.nan)
    names = ["coordination", "bond_mean", "bond_std", "bond_min", "bond_max"]
    for idx, name in enumerate(names):
        vals = dyn[:, idx]
        row[f"{name}_mean"] = float(np.nanmean(vals))
        row[f"{name}_std"] = float(np.nanstd(vals))
        row[f"{name}_max"] = float(np.nanmax(vals))
    if hasattr(data, "mulliken_charge"):
        q = data.mulliken_charge.float().cpu().numpy()
        row["mulliken_mean"] = float(np.nanmean(q))
        row["mulliken_std"] = float(np.nanstd(q))
        row["mulliken_absmax"] = float(np.nanmax(np.abs(q)))
        row["mulliken_range"] = float(np.nanmax(q) - np.nanmin(q))
    return row


def read_predictions(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for row in rows:
        sid = row["sample_id"]
        out[sid] = {
            "sample_id": sid,
            "system_id": row["system_id"],
            "abs_error_gate_uq": float(row["abs_error_gate_uq"]),
            "abs_error_ensemble": float(row["abs_error_ensemble"]),
            "uq_raw_variance": float(row["uq_raw_variance"]),
            "y_true": float(row["y_true"]),
            "pred_gate_uq": float(row["pred_gate_uq"]),
        }
    return out


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dynamic-dir", default=str(DEFAULT_DYNAMIC))
    parser.add_argument("--pred-dir", default=str(DEFAULT_PRED_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    dynamic_dir = Path(args.dynamic_dir)
    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_by_id = {}
    for pred_csv in pred_dir.glob("phase4_2_fold_predictions_*.csv"):
        pred_by_id.update(read_predictions(pred_csv))
    if not pred_by_id:
        raise FileNotFoundError("No phase4_2_fold_predictions_*.csv files found. Run phase4_2 first.")

    rows = []
    for pt_path in sorted(dynamic_dir.glob("*/*.pt")):
        sid = sample_id_from_pt(pt_path)
        if sid not in pred_by_id:
            continue
        row = summarize_pt(pt_path)
        row.update(pred_by_id[sid])
        rows.append(row)

    write_csv(out_dir / "phase6_physical_sample_summary.csv", rows)

    corr_rows = []
    feature_cols = [
        "coordination_mean", "coordination_std", "coordination_max",
        "bond_mean_mean", "bond_std_mean", "bond_max_mean",
        "mulliken_std", "mulliken_absmax", "mulliken_range",
    ]
    target_cols = ["abs_error_gate_uq", "abs_error_ensemble", "uq_raw_variance"]
    systems = sorted({r["system_id"] for r in rows})
    for system in systems + ["ALL"]:
        subset = rows if system == "ALL" else [r for r in rows if r["system_id"] == system]
        for feature in feature_cols:
            if feature not in subset[0]:
                continue
            for target in target_cols:
                corr_rows.append({
                    "system": system,
                    "feature": feature,
                    "target": target,
                    "spearman": spearman_corr(
                        [r.get(feature, np.nan) for r in subset],
                        [r.get(target, np.nan) for r in subset],
                    ),
                    "n": len(subset),
                })
    write_csv(out_dir / "phase6_physical_correlations.csv", corr_rows)
    print(f"Wrote {len(rows)} sample summaries to {out_dir / 'phase6_physical_sample_summary.csv'}")
    print(f"Wrote {len(corr_rows)} correlations to {out_dir / 'phase6_physical_correlations.csv'}")


if __name__ == "__main__":
    main()
