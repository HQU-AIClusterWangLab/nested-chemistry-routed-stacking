"""Metrics used for strict LOSO evaluation and nested rule selection."""
from __future__ import annotations

import numpy as np


def mae95(abs_error: np.ndarray) -> float:
    """Mean absolute error among the largest 5 percent of sample errors."""
    errors = np.asarray(abs_error, dtype=np.float64)
    if errors.size == 0:
        raise ValueError("Cannot evaluate an empty prediction set.")
    k = max(1, int(errors.size * 0.05))
    return float(np.mean(np.sort(errors)[-k:]))


def metric_pair(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    errors = np.abs(np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64))
    return float(np.mean(errors)), mae95(errors)


def router_objective(rows: list[dict], alpha: float = 0.25) -> float:
    """Inner-only rule score: mean(MAE + alpha * MAE95)."""
    if not rows:
        raise ValueError("Router objective needs at least one inner system.")
    return float(np.mean([row["mae"] + alpha * row["mae95"] for row in rows]))
