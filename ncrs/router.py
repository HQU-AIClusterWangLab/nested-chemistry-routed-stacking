"""Leakage-safe selection between the two NCRS fusion paths."""
from __future__ import annotations

from .metrics import router_objective
from .policy import contains_lanthanum


def candidate_rules():
    """Fixed rule family; no rule is fitted on outer-fold labels."""
    return {
        "all_gate": lambda system, context, score: "gate",
        "all_policy": lambda system, context, score: "policy",
        "la_gate_nonla_policy": lambda system, context, score: "gate" if contains_lanthanum(system) else "policy",
        "la_extreme_gate_else_policy": lambda system, context, score: "gate" if contains_lanthanum(system) and score >= 10.0 else "policy",
        "nearid_gate_else_policy": lambda system, context, score: "gate" if context == "near_id" else "policy",
        "severe_policy_else_gate": lambda system, context, score: "policy" if context == "severe_ood" else "gate",
        "nonla_policy_la_gate_if_ood": lambda system, context, score: "gate" if contains_lanthanum(system) and score >= 3.0 else "policy",
    }


def select_rule(outer_system: str, per_system: dict[str, dict], alpha: float = 0.25) -> tuple[str, list[dict]]:
    """Choose a routing rule using the other systems only."""
    rules = candidate_rules()
    inner_systems = [system for system in per_system if system != outer_system]
    if not inner_systems:
        raise ValueError("Nested selection needs at least two systems.")
    ranked = []
    for name, rule in rules.items():
        rows = []
        for system in inner_systems:
            data = per_system[system]
            choice = rule(system, data["context"], data["ood_score"])
            rows.append({"system": system, "choice": choice, **data[choice]})
        ranked.append((router_objective(rows, alpha), name, rows))
    ranked.sort(key=lambda item: (item[0], sum(row["mae"] for row in item[2])))
    return ranked[0][1], ranked[0][2]
