"""Rule-based chemistry path used by NCRS.

The policy is evaluated from OOF labels and visible inference-time prediction
context. Outer held-out labels are used only after the decision for reporting.
"""
from __future__ import annotations

import numpy as np
from scipy.special import softmax
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .metrics import metric_pair
from .policy import choose_chemistry_policy
from .stacking import branch_slices


def _arrays(predictions: np.ndarray, columns: list[str]) -> dict:
    indices = branch_slices(columns)
    branch_mean = np.stack([predictions[:, group].mean(axis=1) for group in indices.values()], axis=1)
    branch_variance = np.stack([predictions[:, group].var(axis=1) for group in indices.values()], axis=1)
    seed_variance = predictions.var(axis=1)
    prediction_range = predictions.max(axis=1) - predictions.min(axis=1)
    branch_range = branch_mean.max(axis=1) - branch_mean.min(axis=1)
    pairwise = np.stack([np.abs(branch_mean[:, i] - branch_mean[:, j]) for i in range(3) for j in range(i + 1, 3)], axis=1)
    features = np.concatenate([
        np.log1p(seed_variance)[:, None], np.log1p(prediction_range)[:, None], np.log1p(branch_range)[:, None],
        branch_mean, np.log1p(branch_variance), np.log1p(pairwise),
    ], axis=1)
    return {
        "branch_mean": branch_mean,
        "seed_variance": seed_variance,
        "branch_range": branch_range,
        "features": features,
    }


def _context(oof: dict, test: dict) -> dict:
    scaler = StandardScaler()
    x_oof = scaler.fit_transform(oof["features"])
    x_test = scaler.transform(test["features"])
    if len(x_oof) < 3:
        raise ValueError("At least three OOF samples are required for context estimation.")
    neighbors = min(25, len(x_oof) - 1)
    model = NearestNeighbors(n_neighbors=neighbors + 1).fit(x_oof)
    train_distances = model.kneighbors(x_oof, return_distance=True)[0][:, 1:].mean(axis=1)
    test_distances = model.kneighbors(x_test, n_neighbors=neighbors, return_distance=True)[0].mean(axis=1)

    def ratio(value: float, reference: float) -> float:
        return value / reference if reference > 1e-12 else float("inf")

    score = max(
        ratio(float(np.median(test_distances)), float(np.percentile(train_distances, 95))),
        ratio(float(np.median(test["seed_variance"])), float(np.percentile(oof["seed_variance"], 95))),
        ratio(float(np.median(test["branch_range"])), float(np.percentile(oof["branch_range"], 95))),
    )
    return {"ood_score": float(score), "context": "severe_ood" if score >= 3.0 else "moderate_shift" if score >= 1.25 else "near_id"}


def fit_policy_path(
    system: str,
    y_oof: np.ndarray,
    predictions_oof: np.ndarray,
    y_test: np.ndarray,
    predictions_test: np.ndarray,
    columns: list[str],
) -> dict:
    """Fit branch-error models on OOF data and apply the predeclared policy."""
    oof, test = _arrays(predictions_oof, columns), _arrays(predictions_test, columns)
    oof_errors = np.abs(oof["branch_mean"] - y_oof[:, None])
    scaler = StandardScaler()
    x_oof, x_test = scaler.fit_transform(oof["features"]), scaler.transform(test["features"])
    models = []
    for branch in range(3):
        model = GradientBoostingRegressor(
            loss="quantile", alpha=0.90, n_estimators=280, learning_rate=0.035,
            max_depth=3, min_samples_leaf=70, random_state=42,
        )
        model.fit(x_oof, oof_errors[:, branch])
        models.append(model)
    oof_estimated_error = np.stack([np.maximum(model.predict(x_oof), 0.0) for model in models], axis=1)
    estimated_error = np.stack([np.maximum(model.predict(x_test), 0.0) for model in models], axis=1)
    temperature = max(float(np.median(oof_estimated_error)), 1e-6)
    reliability_weights = softmax(-estimated_error / temperature, axis=1)
    reliability_prediction = np.sum(reliability_weights * test["branch_mean"], axis=1)
    robust_branch = int(np.argmin(oof_errors.mean(axis=0)))
    fallback_prediction = test["branch_mean"][:, robust_branch]
    simple_prediction = predictions_test.mean(axis=1)
    context = _context(oof, test)
    selected, reason = choose_chemistry_policy(system, context["context"], context["ood_score"])
    mapping = {
        "simple_ensemble_mean": simple_prediction,
        "oof_best_branch_fallback": fallback_prediction,
        "reliability_weighted": reliability_prediction,
    }
    prediction = mapping[selected]
    mae, tail_mae = metric_pair(y_test, prediction)
    return {
        "prediction": prediction,
        "mae": mae,
        "mae95": tail_mae,
        "selected_strategy": selected,
        "reason": reason,
        "context": context["context"],
        "ood_score": context["ood_score"],
        "robust_branch": robust_branch,
    }
