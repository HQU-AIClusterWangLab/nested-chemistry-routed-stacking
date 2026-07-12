# -*- coding: utf-8 -*-
"""Build the Phase 6 computational-chemistry validation package.

This script does not run DFT. It turns the final nested-router predictions,
replay labels, selected candidates, Gaussian logs, and PAA edge exports into
paper-ready tables/figures plus concrete calculation input/task lists.
"""
import argparse
import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(r"D:\lunwen\2.1sci")
FINAL_ROOT = ROOT / "phase 6" / "final_nested_router"
PRED_DIR = FINAL_ROOT / "00_final_predictions"
REPLAY_DIR = FINAL_ROOT / "01_replay_screening"
SELECTION_DIR = FINAL_ROOT / "02_dft_candidate_selection"
OUT_DIR = FINAL_ROOT / "04_computational_chemistry_plan"
LOG_DIR = ROOT / "phase 0" / "dataset"
ATTENTION_ROOT = ROOT / "phase 2" / "phase2_2_paa_attention_ablation_output"

MAIN_SYSTEMS = ["LaCu12", "LaSi9", "BSe9"]
DEFAULT_SI_BUDGET = 8
DEFAULT_MAIN_BUDGET = 30
DELTA_FOR_MAIN = 0.10
KS_FOR_TABLE = [10, 25, 50, 100]

ATOM_SYMBOLS = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 10: "Ne",
    11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl", 18: "Ar",
    19: "K", 20: "Ca", 21: "Sc", 22: "Ti", 23: "V", 24: "Cr", 25: "Mn", 26: "Fe",
    27: "Co", 28: "Ni", 29: "Cu", 30: "Zn", 31: "Ga", 32: "Ge", 33: "As", 34: "Se",
    35: "Br", 36: "Kr", 47: "Ag", 57: "La", 79: "Au",
}


def safe_float(value, default=np.nan):
    try:
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def read_csv(path, typed=True):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if typed:
        for row in rows:
            for key, value in list(row.items()):
                val = safe_float(value, None)
                if val is not None and np.isfinite(val):
                    row[key] = val
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_manifest():
    path = PRED_DIR / "phase6_final_prediction_manifest.csv"
    if path.exists():
        return read_csv(path, typed=False)
    rows = []
    for csv_path in sorted(PRED_DIR.glob("phase6_final_predictions_*.csv")):
        system = csv_path.stem.replace("phase6_final_predictions_", "")
        rows.append({"system": system, "prediction_csv": str(csv_path)})
    return rows


def load_predictions(system):
    path = PRED_DIR / f"phase6_final_predictions_{system}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing final prediction CSV: {path}")
    rows = read_csv(path)
    y = np.asarray([safe_float(r["y_true"]) for r in rows], dtype=float)
    pred = np.asarray([safe_float(r["pred_nested_router"]) for r in rows], dtype=float)
    rel = y - np.nanmin(y)
    order = np.argsort(pred)
    true_order = np.argsort(y)
    pred_rank = np.empty(len(rows), dtype=int)
    true_rank = np.empty(len(rows), dtype=int)
    pred_rank[order] = np.arange(1, len(rows) + 1)
    true_rank[true_order] = np.arange(1, len(rows) + 1)
    for idx, row in enumerate(rows):
        row["predicted_rank_all"] = int(pred_rank[idx])
        row["true_rank_all"] = int(true_rank[idx])
        row["relative_energy_replay"] = float(rel[idx])
        row["ranking_residual"] = int(pred_rank[idx] - true_rank[idx])
    return rows


def rank_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def add_unique(selected, candidates, target):
    seen = set(selected)
    for idx in candidates:
        idx = int(idx)
        if idx not in seen:
            selected.append(idx)
            seen.add(idx)
        if len(selected) >= target:
            break
    return selected


def select_candidates_from_predictions(rows, budget):
    pred = np.asarray([safe_float(r["pred_nested_router"]) for r in rows], dtype=float)
    variance = np.asarray([safe_float(r.get("ensemble_variance", 0.0), 0.0) for r in rows], dtype=float)
    branch_spread = np.asarray([
        max(
            safe_float(r.get("pred_schnet_static_phys", np.nan)),
            safe_float(r.get("pred_paa_schnet_coord", np.nan)),
            safe_float(r.get("pred_painn_coord_bond", np.nan)),
        )
        - min(
            safe_float(r.get("pred_schnet_static_phys", np.nan)),
            safe_float(r.get("pred_paa_schnet_coord", np.nan)),
            safe_float(r.get("pred_painn_coord_bond", np.nan)),
        )
        for r in rows
    ], dtype=float)
    risk = np.nan_to_num(variance + branch_spread, nan=0.0)
    order = np.argsort(pred)
    n_low = max(1, int(round(budget * 0.70)))
    n_risk = max(1, int(round(budget * 0.20)))
    selected = []
    selected = add_unique(selected, order, n_low)
    low_pool = order[:max(n_low, int(0.30 * len(order)))]
    selected = add_unique(selected, low_pool[np.argsort(-risk[low_pool])], n_low + n_risk)
    spread_positions = np.linspace(0, len(order) - 1, max(1, budget - len(selected)) * 4, dtype=int)
    selected = add_unique(selected, order[spread_positions], budget)
    selected = add_unique(selected, order, budget)
    return selected[:budget], order, risk


def selection_rows_from_predictions(system, rows, budget):
    selected, order, risk = select_candidates_from_predictions(rows, budget)
    rank_map = {idx: rank + 1 for rank, idx in enumerate(order)}
    y = np.asarray([safe_float(r["y_true"]) for r in rows], dtype=float)
    rel = y - np.nanmin(y)
    n_low = max(1, int(round(budget * 0.70)))
    n_risk = max(1, int(round(budget * 0.20)))
    out = []
    for sel_rank, idx in enumerate(selected, start=1):
        reason = "low_predicted_energy"
        if sel_rank > n_low:
            reason = "low_energy_high_risk" if sel_rank <= n_low + n_risk else "diversity_fill"
        row = rows[idx]
        out.append({
            "selection_rank": sel_rank,
            "sample_id": row["sample_id"],
            "system_id": system,
            "selection_reason": reason,
            "predicted_rank": rank_map[idx],
            "pred_nested_router": row["pred_nested_router"],
            "risk_score": risk[idx],
            "ensemble_variance": row.get("ensemble_variance", ""),
            "nested_source": row.get("nested_source", ""),
            "router_rule": row.get("router_rule", ""),
            "router_choice": row.get("router_choice", ""),
            "policy_choice": row.get("policy_choice", ""),
            "robust_branch": row.get("robust_branch", ""),
            "y_true_replay": row.get("y_true", ""),
            "replay_relative_energy": rel[idx],
        })
    return out


def ensure_selection(system, budget, out_dir):
    existing = SELECTION_DIR / f"phase6_final_dft_selection_{system}.csv"
    if existing.exists() and budget == DEFAULT_MAIN_BUDGET:
        return read_csv(existing)
    rows = load_predictions(system)
    selected = selection_rows_from_predictions(system, rows, budget)
    write_csv(out_dir / f"phase6_si_light_dft_selection_{system}.csv", selected)
    return selected


def candidate_replay_summary(system, selected_rows, delta=DELTA_FOR_MAIN):
    rel = np.asarray([safe_float(r.get("replay_relative_energy", np.nan)) for r in selected_rows], dtype=float)
    pred = np.asarray([safe_float(r.get("pred_nested_router", np.nan)) for r in selected_rows], dtype=float)
    true = np.asarray([safe_float(r.get("y_true_replay", np.nan)) for r in selected_rows], dtype=float)
    risk = np.asarray([safe_float(r.get("risk_score", np.nan)) for r in selected_rows], dtype=float)
    hit_mask = rel <= delta
    reasons = {}
    for r in selected_rows:
        reasons[r["selection_reason"]] = reasons.get(r["selection_reason"], 0) + 1
    return {
        "system": system,
        "n_selected": len(selected_rows),
        "delta_e_threshold": delta,
        "hit_count": int(np.sum(hit_mask)),
        "hit_rate": float(np.mean(hit_mask)) if len(hit_mask) else float("nan"),
        "best_of_selected_gap": float(np.nanmin(rel)) if len(rel) else float("nan"),
        "budget_to_hit": int(np.where(hit_mask)[0][0] + 1) if np.any(hit_mask) else -1,
        "selected_top_k_contamination": float(np.mean(~hit_mask)) if len(hit_mask) else float("nan"),
        "selected_spearman_pred_vs_true": rank_corr(pred, true),
        "mean_risk_score": float(np.nanmean(risk)) if len(risk) else float("nan"),
        "low_predicted_energy_n": reasons.get("low_predicted_energy", 0),
        "low_energy_high_risk_n": reasons.get("low_energy_high_risk", 0),
        "diversity_fill_n": reasons.get("diversity_fill", 0),
    }


def build_replay_tables(main_systems, si_systems):
    replay = read_csv(REPLAY_DIR / "phase6_final_replay_metrics.csv")
    main_rows = [
        r for r in replay
        if r["system"] in main_systems
        and r["model"] == "nested_router"
        and int(float(r["K"])) in KS_FOR_TABLE
        and abs(float(r["delta_e_threshold"]) - DELTA_FOR_MAIN) < 1e-9
    ]
    si_rows = [
        r for r in replay
        if r["model"] == "nested_router"
        and int(float(r["K"])) in KS_FOR_TABLE
    ]
    write_csv(OUT_DIR / "tables" / "manuscript_nested_router_replay_topk.csv", main_rows)
    write_csv(OUT_DIR / "tables" / "si_all_systems_nested_router_replay.csv", si_rows)
    return main_rows, si_rows


def plot_ranking_curves(main_systems):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for system in main_systems:
        rows = load_predictions(system)
        y = np.asarray([safe_float(r["y_true"]) for r in rows], dtype=float)
        pred = np.asarray([safe_float(r["pred_nested_router"]) for r in rows], dtype=float)
        rel = y - np.nanmin(y)
        order = np.argsort(pred)
        max_k = min(150, len(rows))
        ks = np.arange(1, max_k + 1)
        best_gap = np.asarray([np.nanmin(rel[order[:k]]) for k in ks])
        ref = np.where(rel <= DELTA_FOR_MAIN)[0]
        recall = np.asarray([len(np.intersect1d(order[:k], ref)) / max(1, len(ref)) for k in ks])
        axes[0].plot(ks, best_gap, label=system, linewidth=2)
        axes[1].plot(ks, recall, label=system, linewidth=2)
    axes[0].set_xlabel("DFT budget K")
    axes[0].set_ylabel("Best-of-K DFT relative energy gap (eV)")
    axes[0].set_title("Best structure recovered by budget")
    axes[0].grid(alpha=0.25)
    axes[1].set_xlabel("DFT budget K")
    axes[1].set_ylabel(f"Recall of <= {DELTA_FOR_MAIN:.2f} eV structures")
    axes[1].set_title("Low-energy recall")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    plt.tight_layout()
    path = OUT_DIR / "figures" / "manuscript_topk_budget_curves.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_landscape_proxy(main_systems):
    fig, axes = plt.subplots(1, len(main_systems), figsize=(5 * len(main_systems), 4.5), squeeze=False)
    basin_rows = []
    for ax, system in zip(axes[0], main_systems):
        rows = load_predictions(system)
        features = np.asarray([
            [
                safe_float(r.get("pred_schnet_static_phys", np.nan)),
                safe_float(r.get("pred_paa_schnet_coord", np.nan)),
                safe_float(r.get("pred_painn_coord_bond", np.nan)),
                safe_float(r.get("ensemble_variance", np.nan)),
            ]
            for r in rows
        ], dtype=float)
        y = np.asarray([safe_float(r["y_true"]) for r in rows], dtype=float)
        rel = y - np.nanmin(y)
        mask = np.all(np.isfinite(features), axis=1) & np.isfinite(rel)
        X = features[mask]
        X = (X - X.mean(axis=0)) / np.maximum(X.std(axis=0), 1e-8)
        _, _, vh = np.linalg.svd(X, full_matrices=False)
        pc = X @ vh[:2].T
        sc = ax.scatter(pc[:, 0], pc[:, 1], c=rel[mask], s=9, cmap="viridis_r", alpha=0.75)
        ax.set_title(system)
        ax.set_xlabel("Prediction-space PC1")
        ax.set_ylabel("Prediction-space PC2")
        ax.grid(alpha=0.2)
        low = rel[mask] <= DELTA_FOR_MAIN
        basin_rows.append({
            "system": system,
            "n_samples": int(mask.sum()),
            "n_low_energy_delta_0p10": int(low.sum()),
            "low_energy_fraction": float(low.mean()) if len(low) else float("nan"),
            "best_replay_relative_energy": float(np.nanmin(rel)),
            "median_replay_relative_energy": float(np.nanmedian(rel)),
        })
    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.85)
    cbar.set_label("Replay DFT relative energy (eV)")
    fig.suptitle("Low-energy basin proxy from branch-prediction landscape", y=1.02)
    path = OUT_DIR / "figures" / "manuscript_low_energy_landscape_proxy.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    write_csv(OUT_DIR / "tables" / "manuscript_low_energy_basin_summary.csv", basin_rows)
    return path


def parse_charge_multiplicity(lines):
    for line in lines:
        m = re.search(r"Charge\s*=\s*(-?\d+)\s+Multiplicity\s*=\s*(\d+)", line)
        if m:
            return int(m.group(1)), int(m.group(2))
    return -1, 1


def parse_symbolic_coordinates(lines):
    start = None
    for idx, line in enumerate(lines):
        if "Symbolic Z-matrix:" in line:
            start = idx + 1
            break
    if start is None:
        return []
    coords = []
    pattern = re.compile(
        r"^\s*([A-Z][a-z]?)\s+([-+]?\d+(?:\.\d*)?)\s+([-+]?\d+(?:\.\d*)?)\s+([-+]?\d+(?:\.\d*)?)"
    )
    for line in lines[start:]:
        if not line.strip():
            if coords:
                break
            continue
        if "Charge =" in line:
            continue
        m = pattern.match(line)
        if m:
            coords.append((m.group(1), float(m.group(2)), float(m.group(3)), float(m.group(4))))
        elif coords:
            break
    return coords


def parse_last_standard_orientation(lines):
    starts = [idx for idx, line in enumerate(lines) if "Standard orientation:" in line]
    if not starts:
        return []
    start = starts[-1] + 5
    coords = []
    for line in lines[start:]:
        if "-----" in line:
            break
        parts = line.split()
        if len(parts) >= 6 and parts[0].isdigit():
            atomic_number = int(parts[1])
            symbol = ATOM_SYMBOLS.get(atomic_number, f"X{atomic_number}")
            coords.append((symbol, float(parts[3]), float(parts[4]), float(parts[5])))
    return coords


def read_log_geometry(sample_id):
    log_path = LOG_DIR / f"{sample_id.replace('_sample', '')}.log"
    if not log_path.exists():
        return None, None, None, []
    lines = log_path.read_text(errors="ignore").splitlines()
    charge, mult = parse_charge_multiplicity(lines)
    coords = parse_symbolic_coordinates(lines)
    source = "symbolic_z_matrix_input"
    if not coords:
        coords = parse_last_standard_orientation(lines)
        source = "last_standard_orientation"
    return charge, mult, str(log_path), source, coords


def write_gaussian_input(path, title, charge, mult, coords, route, nproc=12, mem="8GB"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"%nprocshared={nproc}\n")
        f.write(f"%mem={mem}\n")
        f.write(f"%chk={path.stem}.chk\n")
        f.write(f"{route}\n\n")
        f.write(f"{title}\n\n")
        f.write(f"{charge} {mult}\n")
        for sym, x, y, z in coords:
            f.write(f"{sym:<3} {x:>14.8f} {y:>14.8f} {z:>14.8f}\n")
        f.write("\n")


def role_rows(system, selected_rows):
    if not selected_rows:
        return []
    rows = list(selected_rows)
    rows_sorted_rel = sorted(rows, key=lambda r: safe_float(r.get("replay_relative_energy", np.inf), np.inf))
    rows_sorted_risk = sorted(rows, key=lambda r: safe_float(r.get("risk_score", -np.inf), -np.inf), reverse=True)
    low_pred_rows = [r for r in rows if r.get("selection_reason") == "low_predicted_energy"] or rows
    rows_sorted_misrank = sorted(
        low_pred_rows,
        key=lambda r: safe_float(r.get("replay_relative_energy", -np.inf), -np.inf),
        reverse=True,
    )
    candidates = [
        ("low_energy_hit", rows_sorted_rel[0], "best replay DFT energy among selected candidates"),
        ("high_risk_low_energy", rows_sorted_risk[0], "largest risk score among selected candidates"),
        ("misranked_low_predicted", rows_sorted_misrank[0], "low-predicted candidate with poor replay DFT rank"),
    ]
    out = []
    seen = set()
    for role, row, reason in candidates:
        sid = row["sample_id"]
        if sid in seen:
            for alt in rows:
                if alt["sample_id"] not in seen:
                    row = alt
                    sid = row["sample_id"]
                    break
        seen.add(sid)
        out.append((role, row, reason))
    return out


def build_dft_inputs(main_systems, si_systems, main_selections, si_selections, route):
    calc_rows = []
    rep_rows = []
    all_sets = []
    for system in main_systems:
        all_sets.append(("main", system, main_selections[system], "main_30_candidate_validation"))
    for system in si_systems:
        all_sets.append(("si_light", system, si_selections[system], "si_light_candidate_validation"))

    for tier, system, selected_rows, purpose in all_sets:
        for row in selected_rows:
            sample_id = row["sample_id"]
            charge, mult, log_path, geom_source, coords = read_log_geometry(sample_id)
            status = "ready" if coords else "missing_coordinates"
            gjf_path = OUT_DIR / "dft_inputs" / tier / system / f"{sample_id}.gjf"
            if coords:
                write_gaussian_input(
                    gjf_path,
                    f"{purpose}: {sample_id}",
                    charge,
                    mult,
                    coords,
                    route,
                )
            calc_rows.append({
                "tier": tier,
                "system": system,
                "sample_id": sample_id,
                "selection_rank": row.get("selection_rank", ""),
                "selection_reason": row.get("selection_reason", ""),
                "purpose": purpose,
                "recommended_job": "geometry_optimization_plus_frequency",
                "charge": charge if charge is not None else "",
                "multiplicity": mult if mult is not None else "",
                "coordinate_source": geom_source or "",
                "source_log": log_path or "",
                "gaussian_input": str(gjf_path) if coords else "",
                "status": status,
                "pred_nested_router": row.get("pred_nested_router", ""),
                "risk_score": row.get("risk_score", ""),
                "replay_relative_energy": row.get("replay_relative_energy", ""),
            })

    for system in main_systems:
        for role, row, reason in role_rows(system, main_selections[system]):
            sample_id = row["sample_id"]
            rep_rows.append({
                "system": system,
                "sample_id": sample_id,
                "role": role,
                "selection_rank": row.get("selection_rank", ""),
                "reason": reason,
                "recommended_followup": "NBO/Mayer/Wiberg bond order; CDD cube; ELF cube",
                "replay_relative_energy": row.get("replay_relative_energy", ""),
                "risk_score": row.get("risk_score", ""),
                "pred_nested_router": row.get("pred_nested_router", ""),
            })
    write_csv(OUT_DIR / "tables" / "dft_calculation_queue.csv", calc_rows)
    write_csv(OUT_DIR / "tables" / "representative_structure_tasks.csv", rep_rows)
    return calc_rows, rep_rows


def load_attention_edges_for_system(system):
    preferred = [
        ATTENTION_ROOT / "paa_schnet_coord" / f"attention_edges_schnet_coord_{system}.csv",
        ATTENTION_ROOT / "paa_painn_coord_bond" / f"attention_edges_painn_coord_bond_{system}.csv",
        ATTENTION_ROOT / "paa_painn_coord" / f"attention_edges_painn_coord_{system}.csv",
        ATTENTION_ROOT / "paa_schnet_static" / f"attention_edges_schnet_static_{system}.csv",
    ]
    for path in preferred:
        if path.exists():
            return path
    return None


def build_bond_order_templates(rep_rows):
    by_system_samples = {}
    for row in rep_rows:
        by_system_samples.setdefault(row["system"], set()).add(row["sample_id"])
    edge_rows = []
    for system, sample_ids in by_system_samples.items():
        edge_path = load_attention_edges_for_system(system)
        if edge_path is None:
            continue
        with open(edge_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["sample_id"] not in sample_ids:
                    continue
                edge_rows.append({
                    "system_id": row["system_id"],
                    "sample_id": row["sample_id"],
                    "atom_i": row["atom_i"],
                    "atom_j": row["atom_j"],
                    "Z_i": row["Z_i"],
                    "Z_j": row["Z_j"],
                    "distance": row.get("distance", ""),
                    "gate_mean": row.get("gate_mean", ""),
                    "scale_mean": row.get("scale_mean", ""),
                    "phys_diff_mean": row.get("phys_diff_mean", ""),
                    "coord_diff": row.get("coord_diff", ""),
                    "bond_mean_diff": row.get("bond_mean_diff", ""),
                    "bond_std_diff": row.get("bond_std_diff", ""),
                    "nbo_bond_order": "",
                    "mayer_bond_order": "",
                    "wiberg_bond_index": "",
                    "notes": "Fill bond-order columns after DFT/NBO/Mayer/Wiberg analysis.",
                })
    write_csv(OUT_DIR / "tables" / "bond_order_attention_edge_template.csv", edge_rows)
    summary = []
    for system, sample_ids in by_system_samples.items():
        summary.append({
            "system": system,
            "representative_samples": ";".join(sorted(sample_ids)),
            "attention_edge_source": str(load_attention_edges_for_system(system) or ""),
            "n_template_edges": sum(1 for r in edge_rows if r["system_id"] == system),
        })
    write_csv(OUT_DIR / "tables" / "bond_order_attention_task_summary.csv", summary)
    return edge_rows


def write_cube_task_templates(rep_rows):
    rows = []
    for rep in rep_rows:
        for analysis in ["CDD", "ELF"]:
            rows.append({
                "system": rep["system"],
                "sample_id": rep["sample_id"],
                "role": rep["role"],
                "analysis": analysis,
                "input_checkpoint": f"{rep['sample_id']}.chk",
                "formatted_checkpoint": f"{rep['sample_id']}.fchk",
                "suggested_output": f"{rep['sample_id']}_{analysis.lower()}.cube",
                "status": "pending_dft_output",
                "notes": "Generate from converged representative-structure DFT result.",
            })
    write_csv(OUT_DIR / "tables" / "cdd_elf_cube_task_list.csv", rows)


def write_readme(main_systems, si_systems, route):
    path = OUT_DIR / "README_computational_chemistry_validation.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    text = f"""# Phase 6 Computational-Chemistry Validation Package

This folder implements the paper-facing chemistry validation layer for the final
model name `Nested Chemistry-Routed Stacking`.

## Main Systems

Main-text validation systems:

```text
{", ".join(main_systems)}
```

These cover the hardest La OOD case, a La routed-gain case, and a non-La severe
OOD control. The remaining systems are treated as SI coverage:

```text
{", ".join(si_systems)}
```

## Generated Artifacts

```text
tables/manuscript_nested_router_replay_topk.csv
tables/manuscript_dft_candidate_replay_validation.csv
tables/manuscript_low_energy_basin_summary.csv
tables/si_all_systems_nested_router_replay.csv
tables/si_light_dft_candidate_replay_validation.csv
tables/dft_calculation_queue.csv
tables/representative_structure_tasks.csv
tables/bond_order_attention_edge_template.csv
tables/cdd_elf_cube_task_list.csv
figures/manuscript_topk_budget_curves.png
figures/manuscript_low_energy_landscape_proxy.png
dft_inputs/main/<SYSTEM>/*.gjf
dft_inputs/si_light/<SYSTEM>/*.gjf
```

## DFT Route Template

```text
{route}
```

Review basis/functionals before final production if your group has a stricter
standard for lanthanide or heavy-element clusters.

## Intended Manuscript Logic

1. Use the replay tables and Top-K curves to show whether the model finds low
   replay-DFT structures within a small budget.
2. Use the landscape proxy and basin summary to show where the low-energy
   regions sit in the branch-prediction landscape.
3. Fill `bond_order_attention_edge_template.csv` after NBO/Mayer/Wiberg analysis
   and correlate bond order with PAA gate/scale/Bij-related edge quantities.
4. Use the CDD/ELF task list only for representative structures, not for every
   candidate.
"""
    path.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-systems", default=",".join(MAIN_SYSTEMS))
    parser.add_argument("--main-budget", type=int, default=DEFAULT_MAIN_BUDGET)
    parser.add_argument("--si-budget", type=int, default=DEFAULT_SI_BUDGET)
    parser.add_argument(
        "--route",
        default="#p opt=(maxstep=40,maxcycle=150) freq pbe1pbe/lanl2dz scf",
        help="Gaussian route section for generated validation inputs.",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    main_systems = [s.strip() for s in args.main_systems.split(",") if s.strip()]
    all_systems = [r["system"] for r in load_manifest()]
    si_systems = [s for s in all_systems if s not in main_systems]

    build_replay_tables(main_systems, si_systems)
    plot_ranking_curves(main_systems)
    plot_landscape_proxy(main_systems)

    si_out_dir = OUT_DIR / "tables"
    main_selections = {s: ensure_selection(s, args.main_budget, si_out_dir) for s in main_systems}
    si_selections = {s: ensure_selection(s, args.si_budget, si_out_dir) for s in si_systems}

    main_summary = [candidate_replay_summary(s, main_selections[s]) for s in main_systems]
    si_summary = [candidate_replay_summary(s, si_selections[s]) for s in si_systems]
    write_csv(OUT_DIR / "tables" / "manuscript_dft_candidate_replay_validation.csv", main_summary)
    write_csv(OUT_DIR / "tables" / "si_light_dft_candidate_replay_validation.csv", si_summary)

    _, rep_rows = build_dft_inputs(main_systems, si_systems, main_selections, si_selections, args.route)
    build_bond_order_templates(rep_rows)
    write_cube_task_templates(rep_rows)
    write_readme(main_systems, si_systems, args.route)

    print(f"Wrote computational-chemistry validation package to {OUT_DIR}")
    print(f"Main systems: {', '.join(main_systems)}")
    print(f"SI light systems: {', '.join(si_systems)}")


if __name__ == "__main__":
    main()
