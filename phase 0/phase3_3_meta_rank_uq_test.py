# -*- coding: utf-8 -*-
"""
Phase 3.3: meta/rank UQ signal test from Phase 3.2 predictions.

This script does not retrain base models. It reads Phase 3.2 OOF/test
prediction tables and evaluates alternative UQ scores:

  - raw ensemble variance
  - old winsorized isotonic UQ
  - unsupervised rank-UQ
  - meta quantile UQ (LightGBM if installed, sklearn fallback otherwise)
  - combined rank-UQ = local rank + OOD rank

All supervised calibration uses OOF rows only. Test labels are used only for
evaluation.
"""
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, rankdata
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from phase_final_branch_common import ROOT, prediction_columns, read_prediction_table, rows_to_arrays  # noqa: E402


PHASE3_2_DIR = ROOT / "phase 3" / "phase3_2_final_branch_uq_output"
OUTPUT_DIR = ROOT / "phase 3" / "phase3_3_meta_rank_uq_test_output"
PRED_COLS = prediction_columns()

BRANCH_PREFIXES = [
    "schnet_static_phys",
    "paa_schnet_coord",
    "painn_coord_bond",
]


def rank01(x):
    x = np.asarray(x, dtype=np.float64)
    if len(x) <= 1 or np.nanstd(x) == 0:
        return np.full_like(x, 0.5, dtype=np.float64)
    return (rankdata(x, method="average") - 1.0) / (len(x) - 1.0)


def safe_spearman(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 3 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def risk_auc(uq, abs_error, points=50):
    uq = np.asarray(uq, dtype=np.float64)
    err = np.asarray(abs_error, dtype=np.float64)
    order = np.argsort(uq)
    coverages = np.linspace(0.05, 1.0, points)
    risks = []
    for c in coverages:
        n = max(1, int(round(len(order) * c)))
        risks.append(float(err[order[:n]].mean()))
    return float(np.trapz(risks, coverages) / (coverages[-1] - coverages[0])), coverages, np.array(risks)


def top_enrichment(uq, abs_error, frac=0.05):
    uq = np.asarray(uq, dtype=np.float64)
    err = np.asarray(abs_error, dtype=np.float64)
    n = max(1, int(round(len(uq) * frac)))
    idx = np.argsort(-uq)[:n]
    base = err.mean()
    return float(err[idx].mean() / base) if base else float("nan")


def unique_count(x):
    return len(set(np.round(np.asarray(x, dtype=np.float64), 10)))


def branch_indices():
    result = []
    for prefix in BRANCH_PREFIXES:
        result.append([i for i, col in enumerate(PRED_COLS) if col.startswith(prefix)])
    return result


def build_meta_features(rows):
    y, preds, raw_var, old_uq = rows_to_arrays(rows, PRED_COLS)
    branch_idxs = branch_indices()
    branch_means = np.stack([preds[:, idxs].mean(axis=1) for idxs in branch_idxs], axis=1)
    branch_vars = np.stack([preds[:, idxs].var(axis=1) for idxs in branch_idxs], axis=1)
    pred_std = preds.std(axis=1)
    pred_range = preds.max(axis=1) - preds.min(axis=1)
    branch_range = branch_means.max(axis=1) - branch_means.min(axis=1)
    pairwise = []
    for i in range(branch_means.shape[1]):
        for j in range(i + 1, branch_means.shape[1]):
            pairwise.append(np.abs(branch_means[:, i] - branch_means[:, j]))
    pairwise = np.stack(pairwise, axis=1)
    features = np.concatenate([
        np.log1p(raw_var).reshape(-1, 1),
        np.log1p(pred_std).reshape(-1, 1),
        np.log1p(pred_range).reshape(-1, 1),
        np.log1p(branch_range).reshape(-1, 1),
        branch_means,
        np.log1p(branch_vars),
        np.log1p(pairwise),
    ], axis=1)
    abs_error = np.abs(preds.mean(axis=1) - y)
    local_disagreement = (
        0.35 * rank01(raw_var)
        + 0.25 * rank01(pred_range)
        + 0.25 * rank01(branch_range)
        + 0.15 * rank01(branch_vars.mean(axis=1))
    )
    return {
        "y": y,
        "preds": preds,
        "raw_var": raw_var,
        "old_uq": old_uq,
        "abs_error": abs_error,
        "features": features,
        "pred_range": pred_range,
        "branch_range": branch_range,
        "branch_var_mean": branch_vars.mean(axis=1),
        "local_disagreement": local_disagreement,
    }


def fit_quantile_model(X, y, alpha=0.90):
    try:
        from lightgbm import LGBMRegressor  # type: ignore
        model = LGBMRegressor(
            objective="quantile",
            alpha=alpha,
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=15,
            min_child_samples=80,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            verbose=-1,
        )
        model.fit(X, y)
        return model, "lightgbm_quantile_q90"
    except Exception:
        model = GradientBoostingRegressor(
            loss="quantile",
            alpha=alpha,
            n_estimators=250,
            learning_rate=0.035,
            max_depth=3,
            min_samples_leaf=60,
            random_state=42,
        )
        model.fit(X, y)
        return model, "sklearn_gbr_quantile_q90"


def compute_ood_rank(oof_features, test_features):
    scaler = StandardScaler()
    X_oof = scaler.fit_transform(oof_features)
    X_test = scaler.transform(test_features)
    k = min(25, len(X_oof))
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(X_oof)
    dist, _ = nn.kneighbors(X_test)
    knn_dist = dist.mean(axis=1)
    return rank01(knn_dist), knn_dist


def test_fold(system, oof_rows, test_rows):
    oof = build_meta_features(oof_rows)
    test = build_meta_features(test_rows)
    scaler = StandardScaler()
    X_oof = scaler.fit_transform(oof["features"])
    X_test = scaler.transform(test["features"])
    quantile_model, model_name = fit_quantile_model(X_oof, oof["abs_error"])
    meta_q90 = np.asarray(quantile_model.predict(X_test), dtype=np.float64)
    meta_q90 = np.maximum(meta_q90, 0.0)

    ood_rank, knn_dist = compute_ood_rank(oof["features"], test["features"])
    unsup_rank_uq = (
        0.40 * rank01(test["raw_var"])
        + 0.25 * rank01(test["pred_range"])
        + 0.25 * rank01(test["branch_range"])
        + 0.10 * ood_rank
    )
    combined_rank_uq = (
        0.45 * rank01(meta_q90)
        + 0.25 * test["local_disagreement"]
        + 0.30 * ood_rank
    )

    methods = {
        "raw_variance": test["raw_var"],
        "old_isotonic_uq": test["old_uq"],
        "unsupervised_rank_uq": unsup_rank_uq,
        "meta_quantile_q90": meta_q90,
        "combined_rank_uq": combined_rank_uq,
        "ood_knn_distance": knn_dist,
    }
    rows = []
    curves = {}
    for name, uq in methods.items():
        auc, cov, risk = risk_auc(uq, test["abs_error"])
        rows.append({
            "system": system,
            "method": name,
            "n": len(test["abs_error"]),
            "uq_unique": unique_count(uq),
            "spearman": safe_spearman(uq, test["abs_error"]),
            "risk_auc": auc,
            "top5_enrichment": top_enrichment(uq, test["abs_error"]),
            "uq_min": float(np.min(uq)),
            "uq_max": float(np.max(uq)),
            "uq_mean": float(np.mean(uq)),
            "ensemble_mae": float(np.mean(test["abs_error"])),
            "ensemble_mae95": float(np.mean(np.sort(test["abs_error"])[-max(1, int(len(test["abs_error"]) * 0.05)):])),
            "quantile_backend": model_name if name == "meta_quantile_q90" else "",
        })
        curves[name] = (cov, risk)
    return rows, curves, test


def plot_fold(system, rows, curves, out_dir):
    out_dir = Path(out_dir)
    method_order = [r["method"] for r in rows]
    spearman = [r["spearman"] for r in rows]
    enrich = [r["top5_enrichment"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    x = np.arange(len(method_order))
    axes[0].bar(x, spearman, color="#1f77b4")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(method_order, rotation=45, ha="right")
    axes[0].set_title(f"{system}: UQ Spearman vs error")
    axes[0].set_ylabel("rho")

    axes[1].bar(x, enrich, color="#2ca02c")
    axes[1].axhline(1, color="black", linewidth=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(method_order, rotation=45, ha="right")
    axes[1].set_title(f"{system}: top 5% high-UQ enrichment")

    for name, (cov, risk) in curves.items():
        if name in ("raw_variance", "old_isotonic_uq", "unsupervised_rank_uq", "meta_quantile_q90", "combined_rank_uq"):
            axes[2].plot(cov, risk, label=name)
    axes[2].set_title(f"{system}: risk-coverage")
    axes[2].set_xlabel("coverage retained from low UQ")
    axes[2].set_ylabel("MAE")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.25)
    plt.tight_layout()
    path = out_dir / f"phase3_3_uq_methods_{system}.png"
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_summary(all_rows, out_dir):
    methods = ["raw_variance", "old_isotonic_uq", "unsupervised_rank_uq", "meta_quantile_q90", "combined_rank_uq"]
    means = {}
    for metric in ["spearman", "top5_enrichment", "risk_auc"]:
        means[metric] = []
        for method in methods:
            vals = [r[metric] for r in all_rows if r["method"] == method and not math.isnan(r[metric])]
            means[metric].append(float(np.mean(vals)) if vals else float("nan"))

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    x = np.arange(len(methods))
    for ax, metric, title, baseline in [
        (axes[0], "spearman", "Mean Spearman vs error", 0),
        (axes[1], "top5_enrichment", "Mean top 5% enrichment", 1),
        (axes[2], "risk_auc", "Mean risk-coverage AUC", None),
    ]:
        ax.bar(x, means[metric], color="#1f77b4")
        if baseline is not None:
            ax.axhline(baseline, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha="right")
        ax.set_title(title)
    plt.tight_layout()
    path = Path(out_dir) / "phase3_3_meta_rank_uq_summary.png"
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_csv(path, rows):
    fields = [
        "system", "method", "n", "uq_unique", "spearman", "risk_auc",
        "top5_enrichment", "uq_min", "uq_max", "uq_mean",
        "ensemble_mae", "ensemble_mae95", "quantile_backend",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fields})


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not PHASE3_2_DIR.exists():
        raise FileNotFoundError(
            "Phase3.2 output not found. Run phase3_2_final_branch_uq.py first."
        )
    systems = sorted(p.name.replace("fold_", "", 1) for p in PHASE3_2_DIR.glob("fold_*"))
    all_rows = []
    for system in systems:
        fold_dir = PHASE3_2_DIR / f"fold_{system}"
        oof_csv = fold_dir / f"oof_predictions_{system}.csv"
        test_csv = fold_dir / f"test_predictions_{system}.csv"
        if not oof_csv.exists() or not test_csv.exists():
            raise FileNotFoundError(f"Missing prediction CSVs for {system}")
        oof_rows = read_prediction_table(oof_csv)
        test_rows = read_prediction_table(test_csv)
        rows, curves, _ = test_fold(system, oof_rows, test_rows)
        all_rows.extend(rows)
        plot_fold(system, rows, curves, OUTPUT_DIR)
        best = max(rows, key=lambda r: (-999 if math.isnan(r["spearman"]) else r["spearman"]))
        print(
            f"{system}: best Spearman method={best['method']} "
            f"rho={best['spearman']:.4f}, top5={best['top5_enrichment']:.3f}"
        )

    csv_path = OUTPUT_DIR / "phase3_3_meta_rank_uq_metrics.csv"
    json_path = OUTPUT_DIR / "phase3_3_meta_rank_uq_metrics.json"
    write_csv(csv_path, all_rows)
    with open(json_path, "w") as f:
        json.dump(all_rows, f, indent=2)
    plot_path = plot_summary(all_rows, OUTPUT_DIR)
    print("\nComplete.")
    print(f"Metrics CSV : {csv_path}")
    print(f"Metrics JSON: {json_path}")
    print(f"Summary plot: {plot_path}")


if __name__ == "__main__":
    main()
