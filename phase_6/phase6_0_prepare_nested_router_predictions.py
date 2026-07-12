# -*- coding: utf-8 -*-
"""
Phase 6.0: export per-sample predictions for the final nested router model.

This script bridges Phase 4.3 system-level router validation and Phase 6
screening. Phase 4.3 stores which route is selected per held-out system, while
Phase 6 needs per-sample predictions and ranks.

Output:
  phase 6/final_nested_router/00_final_predictions/
    phase6_final_predictions_<SYSTEM>.csv
    phase6_final_prediction_manifest.csv

Requires torch because it re-fits the Phase 4.1-style gate from saved Phase 3.2
OOF/test prediction tables. It does not retrain base models.
"""
import csv
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent.parent / "phase 0"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from phase4_final_branch_stacking_common import (  # noqa: E402
    PRED_COLS,
    build_features,
    load_phase3_fold,
    predict_gate,
    train_gate,
)


ROOT = Path(r"D:\lunwen\2.1sci")
PHASE3_2_DIR = ROOT / "phase 3" / "phase3_2_final_branch_uq_output"
PHASE3_4_DIR = ROOT / "phase 3" / "phase3_4_context_adaptive_uq_output"
PHASE4_2_DIR = ROOT / "phase 4" / "phase4_2_chem_context_policy_stacking_output"
PHASE4_3_DIR = ROOT / "phase 4" / "phase4_3_nested_router_validation_output"
OUT_DIR = ROOT / "phase 6" / "final_nested_router" / "00_final_predictions"

BRANCHES = [
    ("schnet_static_phys", "SchNet-static-phys"),
    ("paa_schnet_coord", "PAA-SchNet-coord"),
    ("painn_coord_bond", "PaiNN-coord_bond"),
]


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def branch_indices():
    out = {}
    for key, _ in BRANCHES:
        out[key] = [i for i, col in enumerate(PRED_COLS) if col.startswith(key + "_seed")]
    return out


def rows_to_arrays(rows):
    y = np.array([float(r["y_true"]) for r in rows], dtype=np.float64)
    preds = np.array([[float(r[c]) for c in PRED_COLS] for r in rows], dtype=np.float64)
    raw_var = np.array([float(r["ensemble_variance"]) for r in rows], dtype=np.float64)
    uq_cal = np.array([float(r["uq_calibrated"]) for r in rows], dtype=np.float64)
    return y, preds, raw_var, uq_cal


def branch_means(preds):
    idxs = branch_indices()
    return {
        key: preds[:, idxs[key]].mean(axis=1)
        for key, _ in BRANCHES
    }


def fit_quantile_regressor(X, y, alpha=0.90):
    from sklearn.ensemble import GradientBoostingRegressor

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
    return model


def build_reliability_features(preds, raw_var):
    bmeans = np.stack(list(branch_means(preds).values()), axis=1)
    idxs = branch_indices()
    bvars = np.stack([preds[:, idxs[key]].var(axis=1) for key, _ in BRANCHES], axis=1)
    pred_range = preds.max(axis=1) - preds.min(axis=1)
    branch_range = bmeans.max(axis=1) - bmeans.min(axis=1)
    pairwise = []
    for i in range(bmeans.shape[1]):
        for j in range(i + 1, bmeans.shape[1]):
            pairwise.append(np.abs(bmeans[:, i] - bmeans[:, j]))
    pairwise = np.stack(pairwise, axis=1)
    return np.concatenate([
        np.log1p(raw_var).reshape(-1, 1),
        np.log1p(pred_range).reshape(-1, 1),
        np.log1p(branch_range).reshape(-1, 1),
        bmeans,
        np.log1p(bvars),
        np.log1p(pairwise),
    ], axis=1)


def reliability_weighted_prediction(oof_rows, test_rows):
    from scipy.special import softmax
    from sklearn.preprocessing import StandardScaler

    y_oof, preds_oof, raw_oof, _ = rows_to_arrays(oof_rows)
    _, preds_test, raw_test, _ = rows_to_arrays(test_rows)
    oof_bmeans = np.stack(list(branch_means(preds_oof).values()), axis=1)
    test_bmeans = np.stack(list(branch_means(preds_test).values()), axis=1)
    oof_branch_errors = np.abs(oof_bmeans - y_oof.reshape(-1, 1))

    scaler = StandardScaler()
    X_oof = scaler.fit_transform(build_reliability_features(preds_oof, raw_oof))
    X_test = scaler.transform(build_reliability_features(preds_test, raw_test))
    pred_err = []
    for i in range(len(BRANCHES)):
        model = fit_quantile_regressor(X_oof, oof_branch_errors[:, i])
        pred_err.append(np.maximum(model.predict(X_test), 0.0))
    pred_err = np.stack(pred_err, axis=1)
    temperature = max(float(np.median(pred_err)), 1e-6)
    weights = softmax(-pred_err / temperature, axis=1)
    return np.sum(weights * test_bmeans, axis=1), pred_err, weights


def robust_branch_prediction(test_preds, robust_branch):
    bmeans = branch_means(test_preds)
    label_to_key = {label: key for key, label in BRANCHES}
    return bmeans[label_to_key[robust_branch]]


def infer_policy_prediction(system, oof_rows, test_rows, policy_row):
    y_test, preds_test, _, _ = rows_to_arrays(test_rows)
    simple = preds_test.mean(axis=1)
    chosen = policy_row["Chosen_Strategy"]
    if chosen == "simple_ensemble_mean":
        return simple, "simple_ensemble_mean"
    if chosen == "reliability_weighted":
        rel_pred, _, _ = reliability_weighted_prediction(oof_rows, test_rows)
        return rel_pred, "reliability_weighted"
    if chosen == "oof_best_branch_fallback":
        return robust_branch_prediction(preds_test, policy_row.get("Reason_Robust_Branch", "")), "oof_best_branch_fallback"
    # Phase4.2 results do not store Reason_Robust_Branch. Recover from Phase3.4 strategy row below.
    return simple, "simple_ensemble_mean_fallback"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)
    router_rows = {r["System"]: r for r in read_csv(PHASE4_3_DIR / "phase4_3_nested_router_outer_results.csv")}
    policy_rows = {r["System"]: r for r in read_csv(PHASE4_2_DIR / "phase4_2_chem_context_policy_stacking_results.csv")}
    strategy_rows = read_csv(PHASE3_4_DIR / "phase3_4_context_adaptive_strategy_metrics.csv")
    strategy_by_system = {}
    for row in strategy_rows:
        strategy_by_system.setdefault(row["system"], {})[row["strategy"]] = row

    manifest = []
    for system, router in router_rows.items():
        oof_rows, test_rows = load_phase3_fold(system)
        y_test, preds_test, raw_var, uq_cal = rows_to_arrays(test_rows)
        y_oof, preds_oof, _, uq_oof = rows_to_arrays(oof_rows)
        features_oof = build_features(preds_oof, uq_oof, include_uq=False)
        features_test = build_features(preds_test, uq_cal, include_uq=False)
        gate_model, gate_scaler, _ = train_gate(features_oof, y_oof, include_uq=False)
        gate_pred, _ = predict_gate(gate_model, gate_scaler, features_test)

        bmeans = branch_means(preds_test)
        simple_pred = preds_test.mean(axis=1)
        rel_pred, rel_err, rel_weights = reliability_weighted_prediction(oof_rows, test_rows)

        robust_branch = strategy_by_system[system]["oof_best_branch_fallback"]["robust_branch"]
        fallback_pred = robust_branch_prediction(preds_test, robust_branch)

        policy_choice = policy_rows[system]["Chosen_Strategy"]
        if policy_choice == "simple_ensemble_mean":
            policy_pred = simple_pred
        elif policy_choice == "reliability_weighted":
            policy_pred = rel_pred
        elif policy_choice == "oof_best_branch_fallback":
            policy_pred = fallback_pred
        else:
            policy_pred = simple_pred

        if router["Selected_Choice"] == "gate":
            nested_pred = gate_pred
            nested_source = "phase4_1_gate"
        else:
            nested_pred = policy_pred
            nested_source = "phase4_2_policy"

        out_path = OUT_DIR / f"phase6_final_predictions_{system}.csv"
        with open(out_path, "w", newline="") as f:
            fieldnames = [
                "key", "system_id", "sample_id", "y_true",
                "pred_nested_router", "pred_phase4_1_gate", "pred_phase4_2_policy",
                "pred_simple_ensemble", "pred_oof_best_fallback", "pred_reliability_weighted",
                "pred_schnet_static_phys", "pred_paa_schnet_coord", "pred_painn_coord_bond",
                "abs_error_nested_router", "abs_error_gate", "abs_error_policy",
                "ensemble_variance", "uq_calibrated_phase3_2",
                "nested_source", "router_rule", "router_choice", "policy_choice", "robust_branch",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for i, row in enumerate(test_rows):
                writer.writerow({
                    "key": row["key"],
                    "system_id": system,
                    "sample_id": row["sample_id"],
                    "y_true": f"{y_test[i]:.10f}",
                    "pred_nested_router": f"{nested_pred[i]:.10f}",
                    "pred_phase4_1_gate": f"{gate_pred[i]:.10f}",
                    "pred_phase4_2_policy": f"{policy_pred[i]:.10f}",
                    "pred_simple_ensemble": f"{simple_pred[i]:.10f}",
                    "pred_oof_best_fallback": f"{fallback_pred[i]:.10f}",
                    "pred_reliability_weighted": f"{rel_pred[i]:.10f}",
                    "pred_schnet_static_phys": f"{bmeans['schnet_static_phys'][i]:.10f}",
                    "pred_paa_schnet_coord": f"{bmeans['paa_schnet_coord'][i]:.10f}",
                    "pred_painn_coord_bond": f"{bmeans['painn_coord_bond'][i]:.10f}",
                    "abs_error_nested_router": f"{abs(nested_pred[i] - y_test[i]):.10f}",
                    "abs_error_gate": f"{abs(gate_pred[i] - y_test[i]):.10f}",
                    "abs_error_policy": f"{abs(policy_pred[i] - y_test[i]):.10f}",
                    "ensemble_variance": f"{raw_var[i]:.10f}",
                    "uq_calibrated_phase3_2": f"{uq_cal[i]:.10f}",
                    "nested_source": nested_source,
                    "router_rule": router["Selected_Rule"],
                    "router_choice": router["Selected_Choice"],
                    "policy_choice": policy_choice,
                    "robust_branch": robust_branch,
                })
        manifest.append({
            "system": system,
            "prediction_csv": str(out_path),
            "n_samples": len(test_rows),
            "router_rule": router["Selected_Rule"],
            "router_choice": router["Selected_Choice"],
            "nested_source": nested_source,
            "policy_choice": policy_choice,
            "robust_branch": robust_branch,
        })
        print(f"Wrote {out_path}")

    with open(OUT_DIR / "phase6_final_prediction_manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        writer.writeheader()
        writer.writerows(manifest)


if __name__ == "__main__":
    main()
