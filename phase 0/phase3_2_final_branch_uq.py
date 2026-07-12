# -*- coding: utf-8 -*-
"""
Phase 3.2: final-branch ensemble UQ test.

Branches:
  - SchNet-static-phys
  - PAA-SchNet-coord
  - PaiNN-coord_bond

The script writes OOF predictions for the training pool and final ensemble
predictions for the held-out system. UQ calibration uses only OOF variance and
OOF absolute error: winsorization followed by isotonic regression.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.isotonic import IsotonicRegression

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from phase_final_branch_common import (  # noqa: E402
    BRANCH_CONFIGS,
    ROOT,
    load_branch_samples,
    prediction_columns,
    generate_oof_predictions,
    train_final_and_predict,
    write_prediction_table,
)


OUTPUT_DIR = ROOT / "phase 3" / "phase3_2_final_branch_uq_output"


def safe_corr(x, y, method="spearman"):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), float("nan")
    if method == "pearson":
        r, p = pearsonr(x, y)
    else:
        r, p = spearmanr(x, y)
    return float(r), float(p)


def risk_coverage_auc(uq, abs_error, points=50):
    uq = np.asarray(uq)
    abs_error = np.asarray(abs_error)
    order = np.argsort(uq)
    coverages = np.linspace(0.05, 1.0, points)
    risks = []
    for c in coverages:
        n = max(1, int(round(len(order) * c)))
        risks.append(float(abs_error[order[:n]].mean()))
    auc = float(np.trapz(risks, coverages) / (coverages[-1] - coverages[0]))
    return coverages, np.array(risks), auc


def top_uq_enrichment(uq, abs_error, frac=0.05):
    n = max(1, int(round(len(uq) * frac)))
    idx = np.argsort(-np.asarray(uq))[:n]
    base = float(np.mean(abs_error))
    if base == 0:
        return float("nan")
    return float(np.mean(np.asarray(abs_error)[idx]) / base)


def build_oof_rows(reference_samples, oof_by_branch, pred_cols):
    rows = []
    sample_map = {sample["key"]: sample for sample in reference_samples}
    for key in sorted(sample_map):
        sample = sample_map[key]
        row = {
            "key": key,
            "system_id": sample["system_id"],
            "sample_id": sample["sample_id"],
            "y_true": f"{sample['y']:.10f}",
        }
        for branch_key, preds_by_key in oof_by_branch.items():
            values = preds_by_key[key]
            for col, value in values.items():
                row[col] = f"{value:.10f}"
        missing = [c for c in pred_cols if c not in row]
        if missing:
            raise RuntimeError(f"Missing OOF predictions for {key}: {missing[:3]}")
        rows.append(row)
    return rows


def build_test_rows(meta, preds_by_col, pred_cols):
    rows = []
    for idx, key in enumerate(meta["keys"]):
        row = {
            "key": key,
            "system_id": meta["systems"][idx],
            "sample_id": meta["sample_ids"][idx],
            "y_true": f"{float(meta['true'][idx]):.10f}",
        }
        for col in pred_cols:
            row[col] = f"{float(preds_by_col[col][idx]):.10f}"
        rows.append(row)
    return rows


def attach_uq(rows, pred_cols, iso=None, winsor_p95=None):
    y = np.array([float(r["y_true"]) for r in rows], dtype=np.float64)
    preds = np.array([[float(r[c]) for c in pred_cols] for r in rows], dtype=np.float64)
    mean = preds.mean(axis=1)
    var = preds.var(axis=1)
    if winsor_p95 is None:
        winsor_p95 = float(np.percentile(var, 95))
    var_w = np.clip(var, 0.0, winsor_p95)
    abs_error = np.abs(mean - y)
    if iso is None:
        iso = IsotonicRegression(out_of_bounds="clip")
        uq_cal = iso.fit_transform(var_w, abs_error)
    else:
        uq_cal = iso.transform(var_w)
    for row, m, v, vw, uq, ae in zip(rows, mean, var, var_w, uq_cal, abs_error):
        row["ensemble_mean"] = f"{float(m):.10f}"
        row["ensemble_variance"] = f"{float(v):.10f}"
        row["abs_error_ensemble"] = f"{float(ae):.10f}"
        row["uq_winsor"] = f"{float(vw):.10f}"
        row["uq_calibrated"] = f"{float(uq):.10f}"
    return iso, winsor_p95, {
        "y": y,
        "preds": preds,
        "mean": mean,
        "var": var,
        "winsor": var_w,
        "uq": uq_cal,
        "abs_error": abs_error,
    }


def metrics_for(system, split, arrays):
    raw_s, _ = safe_corr(arrays["var"], arrays["abs_error"], "spearman")
    cal_s, _ = safe_corr(arrays["uq"], arrays["abs_error"], "spearman")
    raw_p, _ = safe_corr(arrays["var"], arrays["abs_error"], "pearson")
    cal_p, _ = safe_corr(arrays["uq"], arrays["abs_error"], "pearson")
    _, _, raw_auc = risk_coverage_auc(arrays["var"], arrays["abs_error"])
    cov, risk, cal_auc = risk_coverage_auc(arrays["uq"], arrays["abs_error"])
    return {
        "system": system,
        "split": split,
        "n": int(len(arrays["y"])),
        "ensemble_mae": float(np.mean(arrays["abs_error"])),
        "ensemble_mae95": float(np.mean(np.sort(arrays["abs_error"])[-max(1, int(len(arrays["abs_error"]) * 0.05)):])),
        "raw_spearman": raw_s,
        "calibrated_spearman": cal_s,
        "raw_pearson": raw_p,
        "calibrated_pearson": cal_p,
        "raw_risk_auc": raw_auc,
        "calibrated_risk_auc": cal_auc,
        "raw_top5_enrichment": top_uq_enrichment(arrays["var"], arrays["abs_error"]),
        "calibrated_top5_enrichment": top_uq_enrichment(arrays["uq"], arrays["abs_error"]),
        "variance_mean": float(np.mean(arrays["var"])),
        "variance_p95": float(np.percentile(arrays["var"], 95)),
        "coverage": cov.tolist(),
        "calibrated_risk": risk.tolist(),
    }


def plot_fold(system, arrays, out_dir):
    out_dir = Path(out_dir)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sample = np.arange(len(arrays["abs_error"]))
    if len(sample) > 3000:
        rng = np.random.default_rng(42)
        sample = rng.choice(sample, 3000, replace=False)

    axes[0].scatter(arrays["var"][sample], arrays["abs_error"][sample], s=8, alpha=0.35)
    r, _ = safe_corr(arrays["var"], arrays["abs_error"], "spearman")
    axes[0].set_title(f"{system}: raw variance vs error, rho={r:.3f}")
    axes[0].set_xlabel("raw ensemble variance")
    axes[0].set_ylabel("|ensemble error|")

    axes[1].scatter(arrays["uq"][sample], arrays["abs_error"][sample], s=8, alpha=0.35, color="#2ca02c")
    r, _ = safe_corr(arrays["uq"], arrays["abs_error"], "spearman")
    axes[1].set_title(f"{system}: calibrated UQ vs error, rho={r:.3f}")
    axes[1].set_xlabel("winsorized + isotonic UQ")

    cov_raw, risk_raw, _ = risk_coverage_auc(arrays["var"], arrays["abs_error"])
    cov_cal, risk_cal, _ = risk_coverage_auc(arrays["uq"], arrays["abs_error"])
    axes[2].plot(cov_raw, risk_raw, label="raw variance", color="#ff7f0e")
    axes[2].plot(cov_cal, risk_cal, label="calibrated UQ", color="#2ca02c")
    axes[2].set_title(f"{system}: risk-coverage")
    axes[2].set_xlabel("coverage retained from low UQ")
    axes[2].set_ylabel("MAE")
    axes[2].legend()
    axes[2].grid(alpha=0.25)
    plt.tight_layout()
    path = out_dir / f"uq_diagnostics_{system}.png"
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_summary(metrics_rows, out_dir):
    systems = [r["system"] for r in metrics_rows if r["split"] == "test"]
    raw = [r["raw_spearman"] for r in metrics_rows if r["split"] == "test"]
    cal = [r["calibrated_spearman"] for r in metrics_rows if r["split"] == "test"]
    raw_en = [r["raw_top5_enrichment"] for r in metrics_rows if r["split"] == "test"]
    cal_en = [r["calibrated_top5_enrichment"] for r in metrics_rows if r["split"] == "test"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    x = np.arange(len(systems))
    width = 0.36
    axes[0].bar(x - width / 2, raw, width, label="raw variance", color="#ff7f0e")
    axes[0].bar(x + width / 2, cal, width, label="calibrated UQ", color="#2ca02c")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(systems, rotation=45)
    axes[0].set_ylabel("Spearman rho vs |error|")
    axes[0].set_title("UQ ranking quality by held-out system")
    axes[0].legend()

    axes[1].bar(x - width / 2, raw_en, width, label="raw variance", color="#ff7f0e")
    axes[1].bar(x + width / 2, cal_en, width, label="calibrated UQ", color="#2ca02c")
    axes[1].axhline(1, color="black", linewidth=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(systems, rotation=45)
    axes[1].set_ylabel("Top 5% high-UQ error enrichment")
    axes[1].set_title("High-UQ error enrichment")
    axes[1].legend()
    plt.tight_layout()
    path = Path(out_dir) / "phase3_2_final_branch_uq_summary.png"
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_metrics(metrics_rows, out_dir):
    csv_path = Path(out_dir) / "phase3_2_final_branch_uq_metrics.csv"
    fieldnames = [
        "system", "split", "n", "ensemble_mae", "ensemble_mae95",
        "raw_spearman", "calibrated_spearman", "raw_pearson", "calibrated_pearson",
        "raw_risk_auc", "calibrated_risk_auc",
        "raw_top5_enrichment", "calibrated_top5_enrichment",
        "variance_mean", "variance_p95",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics_rows:
            writer.writerow({k: row[k] for k in fieldnames})

    json_path = Path(out_dir) / "phase3_2_final_branch_uq_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics_rows, f, indent=2)
    return csv_path, json_path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("Phase 3.2: final-branch UQ")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Device: {device}")

    pred_cols = prediction_columns()
    branch_samples = load_branch_samples()
    systems = sorted({s["system_id"] for s in next(iter(branch_samples.values()))})
    metrics_rows = []

    for test_system in systems:
        print("\n" + "-" * 80)
        print(f"LOSO system: {test_system}")
        fold_dir = OUTPUT_DIR / f"fold_{test_system}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        oof_by_branch = {}
        test_preds_by_col = {}
        test_meta = None
        reference_train_pool = None

        for branch in BRANCH_CONFIGS:
            samples = branch_samples[branch["key"]]
            train_pool = [s for s in samples if s["system_id"] != test_system]
            test_samples = [s for s in samples if s["system_id"] == test_system]
            if reference_train_pool is None:
                reference_train_pool = train_pool
            print(f"  Branch {branch['label']}: train_pool={len(train_pool)}, test={len(test_samples)}")
            oof_by_branch[branch["key"]] = generate_oof_predictions(
                branch, train_pool, device, test_system
            )
            meta, preds = train_final_and_predict(branch, train_pool, test_samples, device, fold_dir, test_system)
            if test_meta is None:
                test_meta = meta
            elif test_meta["keys"] != meta["keys"]:
                raise RuntimeError(f"Test sample order mismatch for {branch['label']} / {test_system}")
            test_preds_by_col.update(preds)

        oof_rows = build_oof_rows(reference_train_pool, oof_by_branch, pred_cols)
        iso, winsor_p95, oof_arrays = attach_uq(oof_rows, pred_cols)
        test_rows = build_test_rows(test_meta, test_preds_by_col, pred_cols)
        _, _, test_arrays = attach_uq(test_rows, pred_cols, iso=iso, winsor_p95=winsor_p95)

        oof_csv = fold_dir / f"oof_predictions_{test_system}.csv"
        test_csv = fold_dir / f"test_predictions_{test_system}.csv"
        write_prediction_table(oof_csv, oof_rows, pred_cols)
        write_prediction_table(test_csv, test_rows, pred_cols)

        oof_m = metrics_for(test_system, "oof_train", oof_arrays)
        test_m = metrics_for(test_system, "test", test_arrays)
        oof_m["winsor_p95_from_oof"] = float(winsor_p95)
        test_m["winsor_p95_from_oof"] = float(winsor_p95)
        metrics_rows.extend([oof_m, test_m])
        plot_fold(test_system, test_arrays, fold_dir)
        print(
            f"  Test ensemble MAE={test_m['ensemble_mae']:.4f}, "
            f"raw rho={test_m['raw_spearman']:.4f}, calibrated rho={test_m['calibrated_spearman']:.4f}, "
            f"top5 enrich={test_m['calibrated_top5_enrichment']:.3f}"
        )

    csv_path, json_path = write_metrics(metrics_rows, OUTPUT_DIR)
    plot_path = plot_summary(metrics_rows, OUTPUT_DIR)
    print("\nComplete.")
    print(f"Metrics CSV : {csv_path}")
    print(f"Metrics JSON: {json_path}")
    print(f"Summary plot: {plot_path}")


if __name__ == "__main__":
    main()
