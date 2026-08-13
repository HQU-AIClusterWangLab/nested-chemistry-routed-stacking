"""Predeclared chemistry/context policy used as one NCRS fusion path."""
from __future__ import annotations


def contains_lanthanum(system: str) -> bool:
    return system.startswith("La")


def choose_chemistry_policy(system: str, context: str, ood_score: float) -> tuple[str, str]:
    """Select a branch-level prediction without access to outer test labels.

    This is the predeclared final policy used in the manuscript workflow.
    """
    if contains_lanthanum(system) and ood_score >= 10.0:
        return "oof_best_branch_fallback", "La extreme-OOD -> OOF-best-branch fallback"
    if context == "near_id":
        return "reliability_weighted", "near-ID -> reliability-weighted branch ensemble"
    return "simple_ensemble_mean", "otherwise -> simple ensemble mean"
