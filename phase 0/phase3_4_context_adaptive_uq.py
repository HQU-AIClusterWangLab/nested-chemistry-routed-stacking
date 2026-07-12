# -*- coding: utf-8 -*-
"""
Phase 3.4: context-adaptive per-branch reliability UQ.

This is a Phase 3 diagnostic/selection test, not a Phase 4 gate run.
It reads Phase 3.2 prediction tables and evaluates whether UQ can estimate
which final branch should be trusted for each sample/system.

No held-out test labels are used to fit reliability or choose the adaptive
strategy. Test labels are used only for evaluation metrics.
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
from scipy.special import softmax
from scipy.stats import rankdata
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from phase_final_branch_common import ROOT, prediction_columns, read_prediction_table, rows_to_arrays  # noqa: E402


PHASE3_2_DIR = ROOT / "phase 3" / "phase3_2_final_branch_uq_output"
OUTPUT_DIR = ROOT / "phase 3" / "phase3_4_context_adaptive_uq_output"
PRED_COLS = prediction_columns()

BRANCHES = [
    ("schnet_static_phys", "SchNet-static-phys"),
    ("paa_schnet_coord", "PAA-SchNet-coord"),
    ("painn_coord_bond", "PaiNN-coord_bond"),
]


def rank01(x):
    x = np.asarray(x, dtype=np.float64)
    if len(x) <= 1 or np.nanstd(x) == 0:
        return np.full_like(x, 0.5)
    return (rankdata(x, method="average") - 1.0) / (len(x) - 1.0)


def mae95(err):
    err = np.asarray(err, dtype=np.float64)
    k = max(1, int(len(err) * 0.05))
    return float(np.mean(np.sort(err)[-k:]))


def metric_pair(y, pred):
    err = np.abs(np.asarray(pred, dtype=np.float64) - np.asarray(y, dtype=np.float64))
    return float(err.mean()), mae95(err)


def branch_cols():
    cols = {}
    for key, _ in BRANCHES:
        cols[key] = [i for i, col in enumerate(PRED_COLS) if col.startswith(key + "_seed")]
    return cols


def build_arrays(rows):
    y, seed_preds, raw_var, old_uq = rows_to_arrays(rows, PRED_COLS)
    cols = branch_cols()
    branch_mean = np.stack([seed_preds[:, cols[key]].mean(axis=1) for key, _ in BRANCHES], axis=1)
    branch_var = np.stack([seed_preds[:, cols[key]].var(axis=1) for key, _ in BRANCHES], axis=1)
    pred_range = seed_preds.max(axis=1) - seed_preds.min(axis=1)
    branch_range = branch_mean.max(axis=1) - branch_mean.min(axis=1)
    pairwise = []
    for i in range(branch_mean.shape[1]):
        for j in range(i + 1, branch_mean.shape[1]):
            pairwise.append(np.abs(branch_mean[:, i] - branch_mean[:, j]))
    pairwise = np.stack(pairwise, axis=1)
    features = np.concatenate([
        np.log1p(raw_var).reshape(-1, 1),
        np.log1p(pred_range).reshape(-1, 1),
        np.log1p(branch_range).reshape(-1, 1),
        branch_mean,
        np.log1p(branch_var),
        np.log1p(pairwise),
    ], axis=1)
    branch_abs_error = np.abs(branch_mean - y.reshape(-1, 1))
    best_branch_idx = np.argmin(branch_abs_error, axis=1)
    return {
        "y": y,
        "seed_preds": seed_preds,
        "raw_var": raw_var,
        "old_uq": old_uq,
        "branch_mean": branch_mean,
        "branch_var": branch_var,
        "pred_range": pred_range,
        "branch_range": branch_range,
        "features": features,
        "branch_abs_error": branch_abs_error,
        "best_branch_idx": best_branch_idx,
        "ensemble_mean": seed_preds.mean(axis=1),
    }


def fit_quantile_regressor(X, y, alpha=0.90):
    try:
        from lightgbm import LGBMRegressor  # type: ignore
        model = LGBMRegressor(
            objective="quantile",
            alpha=alpha,
            n_estimators=350,
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
            n_estimators=280,
            learning_rate=0.035,
            max_depth=3,
            min_samples_leaf=70,
            random_state=42,
        )
        model.fit(X, y)
        return model, "sklearn_gbr_quantile_q90"


def fit_branch_reliability(oof):
    scaler = StandardScaler()
    X = scaler.fit_transform(oof["features"])
    models = []
    backend = None
    for b in range(len(BRANCHES)):
        model, name = fit_quantile_regressor(X, oof["branch_abs_error"][:, b])
        models.append(model)
        backend = name
    return scaler, models, backend


def predict_branch_reliability(scaler, models, test):
    X = scaler.transform(test["features"])
    pred_err = np.stack([np.maximum(model.predict(X), 0.0) for model in models], axis=1)
    return pred_err


def ood_context(oof, test):
    scaler = StandardScaler()
    X_oof = scaler.fit_transform(oof["features"])
    X_test = scaler.transform(test["features"])
    k = min(25, len(X_oof) - 1)
    nn_oof = NearestNeighbors(n_neighbors=k + 1)
    nn_oof.fit(X_oof)
    oof_dist = nn_oof.kneighbors(X_oof, return_distance=True)[0][:, 1:].mean(axis=1)
    test_dist = nn_oof.kneighbors(X_test, n_neighbors=k, return_distance=True)[0].mean(axis=1)

    def ratio(a, b):
        denom = float(b)
        return float(a / denom) if denom > 1e-12 else float("inf")

    stats = {
        "knn_median_ratio": ratio(np.median(test_dist), np.percentile(oof_dist, 95)),
        "variance_median_ratio": ratio(np.median(test["raw_var"]), np.percentile(oof["raw_var"], 95)),
        "branch_range_median_ratio": ratio(np.median(test["branch_range"]), np.percentile(oof["branch_range"], 95)),
        "test_knn_dist_mean": float(np.mean(test_dist)),
        "ood_score": None,
        "context": None,
    }
    score = max(
        stats["knn_median_ratio"],
        stats["variance_median_ratio"],
        stats["branch_range_median_ratio"],
    )
    stats["ood_score"] = float(score)
    if score >= 3.0:
        stats["context"] = "severe_ood"
    elif score >= 1.25:
        stats["context"] = "moderate_shift"
    else:
        stats["context"] = "near_id"
    return stats, test_dist


def reliability_weights(pred_err, temperature=None):
    pred_err = np.asarray(pred_err, dtype=np.float64)
    if temperature is None:
        temperature = float(np.median(pred_err))
    temperature = max(temperature, 1e-6)
    return softmax(-pred_err / temperature, axis=1)


def evaluate_strategies(system, oof_rows, test_rows):
    oof = build_arrays(oof_rows)
    test = build_arrays(test_rows)
    scaler, models, backend = fit_branch_reliability(oof)
    pred_branch_err = predict_branch_reliability(scaler, models, test)
    oof_pred_branch_err = predict_branch_reliability(scaler, models, oof)
    context, test_knn_dist = ood_context(oof, test)

    oof_branch_mae = oof["branch_abs_error"].mean(axis=0)
    robust_branch_idx = int(np.argmin(oof_branch_mae))
    pred_best_idx = np.argmin(pred_branch_err, axis=1)
    true_best_idx = test["best_branch_idx"]
    oracle_pred = test["branch_mean"][np.arange(len(test["y"])), true_best_idx]
    reliability_pick_pred = test["branch_mean"][np.arange(len(test["y"])), pred_best_idx]

    temp = float(np.median(oof_pred_branch_err))
    rel_w = reliability_weights(pred_branch_err, temp)
    rel_weighted_pred = np.sum(rel_w * test["branch_mean"], axis=1)

    fallback_pred = test["branch_mean"][:, robust_branch_idx]
    branch_disagreement_rank = rank01(test["branch_range"])
    ood_rank = rank01(test_knn_dist)
    conservative_score = 0.65 * ood_rank + 0.35 * branch_disagreement_rank
    conservative_w = np.clip(conservative_score, 0.0, 1.0)
    conservative_pred = (
        conservative_w * fallback_pred
        + (1.0 - conservative_w) * rel_weighted_pred
    )

    if context["context"] == "severe_ood":
        adaptive_pred = conservative_pred
        adaptive_name = "conservative_fallback"
    elif context["context"] == "moderate_shift":
        adaptive_pred = 0.5 * conservative_pred + 0.5 * rel_weighted_pred
        adaptive_name = "mixed_reliability_conservative"
    else:
        adaptive_pred = rel_weighted_pred
        adaptive_name = "reliability_weighted"

    y = test["y"]
    branch_metrics = []
    for i, (_, label) in enumerate(BRANCHES):
        mae, m95 = metric_pair(y, test["branch_mean"][:, i])
        branch_metrics.append((label, mae, m95))

    strategies = {
        "oracle_best_branch": oracle_pred,
        "simple_ensemble_mean": test["ensemble_mean"],
        "oof_best_branch_fallback": fallback_pred,
        "reliability_pick_branch": reliability_pick_pred,
        "reliability_weighted": rel_weighted_pred,
        "conservative_fallback": conservative_pred,
        "context_adaptive": adaptive_pred,
    }
    rows = []
    for name, pred in strategies.items():
        mae, m95 = metric_pair(y, pred)
        rows.append({
            "system": system,
            "strategy": name,
            "n": len(y),
            "mae": mae,
            "mae95": m95,
            "oracle_gap_mae": mae - metric_pair(y, oracle_pred)[0],
            "branch_hit_rate": float(np.mean(pred_best_idx == true_best_idx)) if name in ("reliability_pick_branch", "reliability_weighted", "conservative_fallback", "context_adaptive") else "",
            "context": context["context"],
            "adaptive_policy": adaptive_name if name == "context_adaptive" else "",
            "robust_branch": BRANCHES[robust_branch_idx][1],
            "quantile_backend": backend,
            "ood_score": context["ood_score"],
            "knn_median_ratio": context["knn_median_ratio"],
            "variance_median_ratio": context["variance_median_ratio"],
            "branch_range_median_ratio": context["branch_range_median_ratio"],
        })

    reliability_rows = []
    for i, (_, label) in enumerate(BRANCHES):
        pred_err_i = pred_branch_err[:, i]
        true_err_i = test["branch_abs_error"][:, i]
        if np.std(pred_err_i) == 0 or np.std(true_err_i) == 0:
            rho = float("nan")
        else:
            from scipy.stats import spearmanr
            rho = float(spearmanr(pred_err_i, true_err_i).statistic)
        reliability_rows.append({
            "system": system,
            "branch": label,
            "pred_error_mean": float(np.mean(pred_err_i)),
            "true_error_mean": float(np.mean(true_err_i)),
            "spearman_pred_vs_true_error": rho,
            "pred_error_unique": len(set(np.round(pred_err_i, 10))),
        })
    return rows, reliability_rows, branch_metrics, context


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_system(system, strategy_rows, branch_metrics, out_dir):
    out_dir = Path(out_dir)
    labels = [r["strategy"] for r in strategy_rows if r["strategy"] != "oracle_best_branch"]
    maes = [r["mae"] for r in strategy_rows if r["strategy"] != "oracle_best_branch"]
    mae95 = [r["mae95"] for r in strategy_rows if r["strategy"] != "oracle_best_branch"]
    oracle = next(r for r in strategy_rows if r["strategy"] == "oracle_best_branch")
    fig, axes = plt.subplots(1, 2, figsize=(17, 5))
    x = np.arange(len(labels))
    axes[0].bar(x, maes, color="#1f77b4")
    axes[0].axhline(oracle["mae"], color="black", linestyle="--", label="oracle branch")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45, ha="right")
    axes[0].set_title(f"{system}: branch reliability MAE")
    axes[0].set_ylabel("eV")
    axes[0].legend()

    axes[1].bar(x, mae95, color="#ff7f0e")
    axes[1].axhline(oracle["mae95"], color="black", linestyle="--", label="oracle branch")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha="right")
    axes[1].set_title(f"{system}: branch reliability MAE_95")
    axes[1].set_ylabel("eV")
    axes[1].legend()
    plt.tight_layout()
    path = out_dir / f"phase3_4_context_adaptive_{system}.png"
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_summary(all_strategy_rows, out_dir):
    strategies = [
        "simple_ensemble_mean",
        "oof_best_branch_fallback",
        "reliability_pick_branch",
        "reliability_weighted",
        "conservative_fallback",
        "context_adaptive",
    ]
    means = []
    tails = []
    gaps = []
    for strategy in strategies:
        rows = [r for r in all_strategy_rows if r["strategy"] == strategy]
        means.append(float(np.mean([r["mae"] for r in rows])))
        tails.append(float(np.mean([r["mae95"] for r in rows])))
        gaps.append(float(np.mean([r["oracle_gap_mae"] for r in rows])))
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    x = np.arange(len(strategies))
    for ax, vals, title in [
        (axes[0], means, "Mean MAE"),
        (axes[1], tails, "Mean MAE_95"),
        (axes[2], gaps, "Mean oracle gap"),
    ]:
        ax.bar(x, vals, color="#1f77b4")
        ax.set_xticks(x)
        ax.set_xticklabels(strategies, rotation=45, ha="right")
        ax.set_title(title)
        ax.set_ylabel("eV")
        ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    path = Path(out_dir) / "phase3_4_context_adaptive_summary.png"
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not PHASE3_2_DIR.exists():
        raise FileNotFoundError("Run phase3_2_final_branch_uq.py first.")
    systems = sorted(p.name.replace("fold_", "", 1) for p in PHASE3_2_DIR.glob("fold_*"))
    all_strategy_rows = []
    all_reliability_rows = []
    context_rows = []
    for system in systems:
        fold_dir = PHASE3_2_DIR / f"fold_{system}"
        oof_csv = fold_dir / f"oof_predictions_{system}.csv"
        test_csv = fold_dir / f"test_predictions_{system}.csv"
        oof_rows = read_prediction_table(oof_csv)
        test_rows = read_prediction_table(test_csv)
        strategy_rows, reliability_rows, branch_metrics, context = evaluate_strategies(system, oof_rows, test_rows)
        all_strategy_rows.extend(strategy_rows)
        all_reliability_rows.extend(reliability_rows)
        context_rows.append({"system": system, **context})
        plot_system(system, strategy_rows, branch_metrics, OUTPUT_DIR)
        best = min(
            [r for r in strategy_rows if r["strategy"] != "oracle_best_branch"],
            key=lambda r: r["mae"],
        )
        print(
            f"{system}: context={context['context']} best={best['strategy']} "
            f"MAE={best['mae']:.4f}, robust={best['robust_branch']}"
        )

    strategy_csv = OUTPUT_DIR / "phase3_4_context_adaptive_strategy_metrics.csv"
    reliability_csv = OUTPUT_DIR / "phase3_4_branch_reliability_metrics.csv"
    context_csv = OUTPUT_DIR / "phase3_4_system_context_metrics.csv"
    write_csv(strategy_csv, all_strategy_rows, [
        "system", "strategy", "n", "mae", "mae95", "oracle_gap_mae", "branch_hit_rate",
        "context", "adaptive_policy", "robust_branch", "quantile_backend",
        "ood_score", "knn_median_ratio", "variance_median_ratio", "branch_range_median_ratio",
    ])
    write_csv(reliability_csv, all_reliability_rows, [
        "system", "branch", "pred_error_mean", "true_error_mean",
        "spearman_pred_vs_true_error", "pred_error_unique",
    ])
    write_csv(context_csv, context_rows, [
        "system", "context", "ood_score", "knn_median_ratio",
        "variance_median_ratio", "branch_range_median_ratio", "test_knn_dist_mean",
    ])
    with open(OUTPUT_DIR / "phase3_4_context_adaptive_all_metrics.json", "w") as f:
        json.dump({
            "strategy_metrics": all_strategy_rows,
            "branch_reliability_metrics": all_reliability_rows,
            "context_metrics": context_rows,
        }, f, indent=2)
    summary_plot = plot_summary(all_strategy_rows, OUTPUT_DIR)
    print("\nComplete.")
    print(f"Strategy metrics   : {strategy_csv}")
    print(f"Reliability metrics: {reliability_csv}")
    print(f"Context metrics    : {context_csv}")
    print(f"Summary plot       : {summary_plot}")


if __name__ == "__main__":
    main()
