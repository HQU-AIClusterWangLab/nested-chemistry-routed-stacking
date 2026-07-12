# -*- coding: utf-8 -*-
"""Shared stacking utilities for final-branch Phase 4.1 and 4.2."""
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from phase_final_branch_common import ROOT, prediction_columns, read_prediction_table, rows_to_arrays  # noqa: E402


PHASE3_OUTPUT = ROOT / "phase 3" / "phase3_2_final_branch_uq_output"
PRED_COLS = prediction_columns()
BRANCH_SLICES = {
    "schnet_static_phys": [i for i, c in enumerate(PRED_COLS) if c.startswith("schnet_static_phys_")],
    "paa_schnet_coord": [i for i, c in enumerate(PRED_COLS) if c.startswith("paa_schnet_coord_")],
    "painn_coord_bond": [i for i, c in enumerate(PRED_COLS) if c.startswith("painn_coord_bond_")],
}


class GatingNetwork(nn.Module):
    def __init__(self, input_dim, n_models, hidden=96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, n_models),
        )

    def forward(self, x):
        weights = torch.softmax(self.net(x), dim=-1)
        base = x[:, :weights.size(1)]
        return (weights * base).sum(dim=-1), weights


def load_phase3_fold(system):
    fold_dir = PHASE3_OUTPUT / f"fold_{system}"
    oof_csv = fold_dir / f"oof_predictions_{system}.csv"
    test_csv = fold_dir / f"test_predictions_{system}.csv"
    if not oof_csv.exists() or not test_csv.exists():
        raise FileNotFoundError(
            "Missing Phase3.2 prediction tables. Run first:\n"
            '  python "phase 0\\phase3_2_final_branch_uq.py"\n'
            f"Missing: {oof_csv if not oof_csv.exists() else test_csv}"
        )
    oof_rows = read_prediction_table(oof_csv)
    test_rows = read_prediction_table(test_csv)
    return oof_rows, test_rows


def available_systems():
    if not PHASE3_OUTPUT.exists():
        return []
    systems = []
    for fold_dir in sorted(PHASE3_OUTPUT.glob("fold_*")):
        systems.append(fold_dir.name.replace("fold_", "", 1))
    return systems


def build_features(preds, uq=None, include_uq=False):
    branch_means = []
    branch_vars = []
    for idxs in BRANCH_SLICES.values():
        branch = preds[:, idxs]
        branch_means.append(branch.mean(axis=1))
        branch_vars.append(branch.var(axis=1))
    pieces = [preds, np.stack(branch_means, axis=1), np.stack(branch_vars, axis=1)]
    if include_uq:
        if uq is None:
            raise ValueError("include_uq=True requires uq")
        pieces.append(uq.reshape(-1, 1))
    return np.concatenate(pieces, axis=1)


def mae95(abs_error):
    k = max(1, int(len(abs_error) * 0.05))
    return float(np.mean(np.sort(abs_error)[-k:]))


def evaluate_predictions(y, preds):
    ae = np.abs(preds - y)
    return float(np.mean(ae)), mae95(ae)


def train_gate(X_oof, y_oof, include_uq=False, device=None, epochs=400, patience=50):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = len(y_oof)
    idx = np.arange(n)
    rng = np.random.default_rng(42)
    rng.shuffle(idx)
    n_train = int(n * 0.8)
    train_idx, val_idx = idx[:n_train], idx[n_train:]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_oof)
    X = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    y = torch.tensor(y_oof, dtype=torch.float32).to(device)
    model = GatingNetwork(X_oof.shape[1], len(PRED_COLS)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    best_state, best_val, bad = None, float("inf"), 0
    train_t = torch.tensor(train_idx, dtype=torch.long, device=device)
    val_t = torch.tensor(val_idx, dtype=torch.long, device=device)

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        pred, _ = model(X[train_t])
        loss = F.l1_loss(pred, y[train_t])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        model.eval()
        with torch.no_grad():
            val_pred, _ = model(X[val_t])
            val_loss = F.l1_loss(val_pred, y[val_t]).item()
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= patience:
            break
    model.load_state_dict(best_state)
    return model, scaler, float(best_val)


def predict_gate(model, scaler, X, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    X_scaled = scaler.transform(X)
    with torch.no_grad():
        pred, weights = model(torch.tensor(X_scaled, dtype=torch.float32).to(device))
    return pred.cpu().numpy(), weights.cpu().numpy()


def fold_metrics(system, oof_rows, test_rows, include_uq=False):
    y_oof, preds_oof, _, uq_oof = rows_to_arrays(oof_rows, PRED_COLS)
    y_test, preds_test, raw_var_test, uq_test = rows_to_arrays(test_rows, PRED_COLS)
    X_oof = build_features(preds_oof, uq_oof, include_uq=include_uq)
    X_test = build_features(preds_test, uq_test, include_uq=include_uq)
    gate, scaler, val_loss = train_gate(X_oof, y_oof, include_uq=include_uq)
    gate_pred, weights = predict_gate(gate, scaler, X_test)

    seed_maes = [evaluate_predictions(y_test, preds_test[:, i])[0] for i in range(preds_test.shape[1])]
    best_idx = int(np.argmin(seed_maes))
    best_pred = preds_test[:, best_idx]
    simple_mean = preds_test.mean(axis=1)
    branch_mean_preds = []
    for idxs in BRANCH_SLICES.values():
        branch_mean_preds.append(preds_test[:, idxs].mean(axis=1))
    branch_mean = np.stack(branch_mean_preds, axis=1).mean(axis=1)

    best_mae, best_m95 = evaluate_predictions(y_test, best_pred)
    mean_mae, mean_m95 = evaluate_predictions(y_test, simple_mean)
    branch_mae, branch_m95 = evaluate_predictions(y_test, branch_mean)
    gate_mae, gate_m95 = evaluate_predictions(y_test, gate_pred)

    return {
        "system": system,
        "n_test": len(y_test),
        "best_single_col": PRED_COLS[best_idx],
        "best_single_mae": best_mae,
        "best_single_mae95": best_m95,
        "simple_mean_mae": mean_mae,
        "simple_mean_mae95": mean_m95,
        "branch_mean_mae": branch_mae,
        "branch_mean_mae95": branch_m95,
        "gated_mae": gate_mae,
        "gated_mae95": gate_m95,
        "gate_val_loss": val_loss,
        "gate_weight_mean": weights.mean(axis=0).tolist(),
        "y_true": y_test,
        "gate_pred": gate_pred,
        "simple_mean_pred": simple_mean,
        "branch_mean_pred": branch_mean,
        "raw_var": raw_var_test,
        "uq": uq_test,
    }


def write_results_csv(path, results, uq=False):
    fields = [
        "System", "N_Test", "BestSingle_Model",
        "BestSingle_MAE", "BestSingle_MAE95",
        "SimpleMean_MAE", "SimpleMean_MAE95",
        "BranchMean_MAE", "BranchMean_MAE95",
        "Gated_MAE", "Gated_MAE95", "Gate_Val_Loss",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "System": r["system"],
                "N_Test": r["n_test"],
                "BestSingle_Model": r["best_single_col"],
                "BestSingle_MAE": f"{r['best_single_mae']:.6f}",
                "BestSingle_MAE95": f"{r['best_single_mae95']:.6f}",
                "SimpleMean_MAE": f"{r['simple_mean_mae']:.6f}",
                "SimpleMean_MAE95": f"{r['simple_mean_mae95']:.6f}",
                "BranchMean_MAE": f"{r['branch_mean_mae']:.6f}",
                "BranchMean_MAE95": f"{r['branch_mean_mae95']:.6f}",
                "Gated_MAE": f"{r['gated_mae']:.6f}",
                "Gated_MAE95": f"{r['gated_mae95']:.6f}",
                "Gate_Val_Loss": f"{r['gate_val_loss']:.6f}",
            })
