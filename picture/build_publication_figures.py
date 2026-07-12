import csv
import json
import math
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import lines as mlines
from matplotlib import patches as mpatches
from matplotlib.gridspec import GridSpec


ROOT = Path(__file__).resolve().parents[1]
PICTURE_DIR = ROOT / "picture"
OUT_DIR = PICTURE_DIR / "publication_figures"
SOURCE_DIR = OUT_DIR / "source_data"

PHASE3_DIR = ROOT / "phase 3"
PHASE4_DIR = ROOT / "phase 4"
PHASE6_DIR = ROOT / "phase 6" / "final_nested_router"

PALETTE = {
    "baseline_dark": "#484878",
    "baseline_mid": "#7884B4",
    "baseline_soft": "#B4C0E4",
    "hero_light": "#E4E4F0",
    "hero_mid": "#E4CCD8",
    "hero_dark": "#F0C0CC",
    "neutral_light": "#D8D8D8",
    "neutral_mid": "#A8A8A8",
    "neutral_dark": "#606060",
    "text_dark": "#2B2B2B",
    "teal": "#33B5A5",
    "aqua": "#77D7D1",
    "violet": "#7C6CCF",
    "lilac": "#B9A7E8",
    "callout_red": "#E53935",
    "callout_green": "#2E9E44",
    "band_aqua": "#EAF7F6",
    "band_lilac": "#F4F0FB",
    "band_peach": "#F8EEE8",
}

METHOD_COLORS = {
    "gate": PALETTE["baseline_dark"],
    "policy": PALETTE["teal"],
    "nested": PALETTE["hero_dark"],
}

SYSTEM_COLORS = {
    "AgB8": PALETTE["baseline_mid"],
    "AuB8": PALETTE["baseline_soft"],
    "LaB8": PALETTE["neutral_mid"],
    "LaSe8": PALETTE["neutral_dark"],
    "LaCu12": PALETTE["violet"],
    "LaSi9": PALETTE["teal"],
    "BSe9": PALETTE["callout_red"],
}

SYSTEM_LABELS = {
    "AgB8": r"AgB$_{8}^{-}$",
    "AuB8": r"AuB$_{8}^{-}$",
    "BSe9": r"BSe$_{9}^{-}$",
    "LaB8": r"LaB$_{8}^{-}$",
    "LaCu12": r"LaCu$_{12}^{-}$",
    "LaSe8": r"LaSe$_{8}^{-}$",
    "LaSi9": r"LaSi$_{9}^{-}$",
}

SELECTION_REASON_COLORS = {
    "low_predicted_energy": PALETTE["hero_mid"],
    "low_energy_high_risk": PALETTE["teal"],
    "diversity_fill": PALETTE["neutral_mid"],
}

MAIN_SYSTEMS = ["LaCu12", "LaSi9", "BSe9"]
SI_SYSTEMS = ["AgB8", "AuB8", "LaB8", "LaSe8"]


def apply_publication_style(font_size=7, axes_linewidth=0.9):
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.size"] = font_size
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.linewidth"] = axes_linewidth
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["xtick.color"] = PALETTE["text_dark"]
    plt.rcParams["ytick.color"] = PALETTE["text_dark"]
    plt.rcParams["axes.labelcolor"] = PALETTE["text_dark"]
    plt.rcParams["text.color"] = PALETTE["text_dark"]


def ensure_dirs():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)


def read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def as_float(value, default=np.nan):
    if value is None:
        return default
    if isinstance(value, (float, int)):
        return float(value)
    text = str(value).strip()
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def as_int(value, default=0):
    number = as_float(value, default=np.nan)
    if not np.isfinite(number):
        return default
    return int(round(number))


def mm_to_inch(mm_value):
    return mm_value / 25.4


def save_figure(fig, stem, width_mm=None, height_mm=None, preview_dpi=300):
    if width_mm and height_mm:
        fig.set_size_inches(mm_to_inch(width_mm), mm_to_inch(height_mm))
    base = OUT_DIR / stem
    fig.savefig(f"{base}.svg", bbox_inches="tight")
    fig.savefig(f"{base}.pdf", bbox_inches="tight")
    fig.savefig(f"{base}.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(f"{base}.png", dpi=preview_dpi, bbox_inches="tight")
    plt.close(fig)
    return {
        "svg": f"{base}.svg",
        "pdf": f"{base}.pdf",
        "tiff": f"{base}.tiff",
        "png": f"{base}.png",
    }


def add_panel_label(ax, label, x=-0.06, y=1.02):
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def lighten(color, amount=0.35):
    rgb = np.array(matplotlib.colors.to_rgb(color))
    return matplotlib.colors.to_hex(rgb + (1.0 - rgb) * amount)


def wrap_lines(text, width):
    wrapped = []
    for paragraph in str(text).splitlines():
        if not paragraph:
            wrapped.append("")
        else:
            wrapped.extend(textwrap.wrap(paragraph, width=width, break_long_words=False))
    return "\n".join(wrapped)


def format_system_list(system_ids):
    return ", ".join(SYSTEM_LABELS.get(system_id, system_id) for system_id in system_ids)


def write_figure_contracts():
    contract_text = textwrap.dedent(
        """\
        # Publication Figure Contracts

        Figure 1
        Core conclusion: The manuscript is a leakage-safe workflow that moves from Gaussian-log data to nested routing and then to finite-budget replay and DFT validation.
        Figure archetype: schematic-led composite
        Target journal/output: JCTC-style main-text workflow figure; SVG, PDF, TIFF, PNG
        Backend: Python
        Final size: 183 mm x 132 mm
        Panel map:
          a: Workflow from dataset construction to Phase 6 validation
          b: Evidence cards for main systems, metrics, and claim boundaries
        Evidence hierarchy:
          hero evidence: leakage-safe LOSO and nested-router selection chain
          validation evidence: finite-budget replay and DFT package handoff
          controls/robustness: explicit replay-vs-DFT separation
        Statistics needed: dataset counts, outer-system protocol counts
        Source data needed: _first_draft_extract.json and project phase summaries
        Image-integrity notes: vector schematic only
        Reviewer risk: over-claiming de-duplication or hiding replay-vs-DFT distinction

        Figure 2
        Core conclusion: Nested Chemistry-Routed Stacking is a more robust compromise than either standalone routing branch, while still showing visible regret on two SI systems.
        Figure archetype: asymmetric mixed-modality figure
        Target journal/output: JCTC-style main-text performance figure; SVG, PDF, TIFF, PNG
        Backend: Python
        Final size: 183 mm x 140 mm
        Panel map:
          a: Per-system MAE for gate, policy, and nested router
          b: Mean MAE and MAE95 summary across the three workflows
          c: Router regret versus the lower-error standalone branch
        Evidence hierarchy:
          hero evidence: per-system MAE comparison under outer held-out validation
          validation evidence: mean MAE and MAE95 summary
          controls/robustness: explicit regret panel
        Statistics needed: outer-held-out MAE and MAE95
        Source data needed: phase4_3_nested_router_outer_results.csv and phase4_3_nested_router_summary.csv
        Image-integrity notes: vector chart only
        Reviewer risk: hiding systems where nested selection misses the lower-error standalone branch

        Figure 3
        Core conclusion: Replay screening quality is highly system dependent, and BSe9 only becomes useful after DFT relaxation recovers a low-energy basin from a weak raw ranking.
        Figure archetype: asymmetric mixed-modality figure
        Target journal/output: JCTC-style main-text screening and DFT validation figure; SVG, PDF, TIFF, PNG
        Backend: Python
        Final size: 183 mm x 145 mm
        Panel map:
          a: All-system recall curves versus DFT budget
          b: Main-system best-of-K replay gap curves
          c: BSe9 selected-candidate replay gaps versus completed DFT-relaxed gaps
        Evidence hierarchy:
          hero evidence: all-system budget recall contrast
          validation evidence: main-system best-gap curves and BSe9 DFT audit
          controls/robustness: missing-log markers and explicit replay-vs-DFT separation
        Statistics needed: Recall@K, Best-of-K gap, completed DFT-relaxed energies, hit counts
        Source data needed: figure3 origin CSVs and phase6_final_dft_selection_BSe9.csv
        Image-integrity notes: vector chart only
        Reviewer risk: implying BSe9 raw top-K success or conflating replay ranking with post-optimization ranking

        Figure S1
        Core conclusion: Scalar UQ scores collapse or become negative on most outer held-out systems, so scalar UQ cannot be promoted as a reliable final routing signal.
        Figure archetype: quantitative grid
        Target journal/output: SI diagnostic figure; SVG, PDF, TIFF, PNG
        Backend: Python
        Final size: 183 mm x 78 mm
        Panel map:
          a: Median Spearman by scalar-UQ method
          b: Fraction of held-out systems with negative or undefined Spearman
        Evidence hierarchy:
          hero evidence: median Spearman
          validation evidence: negative-or-NaN burden
          controls/robustness: zero-reference line
        Statistics needed: per-method Spearman summaries
        Source data needed: phase3_uq_spearman_summary_for_origin.csv
        Image-integrity notes: vector chart only
        Reviewer risk: decorating a diagnostic failure into an apparent success

        Figure S2
        Core conclusion: The chemistry/context policy is the best label-free strategy among tested Phase 3.5 variants, but it still trails the later nested router.
        Figure archetype: quantitative grid
        Target journal/output: SI strategy-diagnostic figure; SVG, PDF, TIFF, PNG
        Backend: Python
        Final size: 183 mm x 82 mm
        Panel map:
          a: Mean MAE and MAE95 by strategy
          b: Mean oracle-gap MAE by strategy
        Evidence hierarchy:
          hero evidence: MAE/MAE95 comparison
          validation evidence: oracle-gap distance
          controls/robustness: oracle strategy shown as lower bound only
        Statistics needed: strategy-level MAE, MAE95, oracle gap
        Source data needed: phase3_strategy_summary_for_origin.csv
        Image-integrity notes: vector chart only
        Reviewer risk: overstating the policy as a final solution rather than a diagnostic precursor
        """
    )
    contract_path = OUT_DIR / "figure_contracts.md"
    contract_path.write_text(contract_text, encoding="utf-8")
    return contract_path


def load_dataset_snapshot():
    snapshot = load_json(ROOT / "_first_draft_extract.json")
    totals = snapshot.get("dataset_totals", {})
    return {
        "n_logs": totals.get("logs", ""),
        "n_samples": totals.get("pt_samples", ""),
    }


def load_phase4_outer_results():
    rows = read_csv(PHASE4_DIR / "phase4_3_nested_router_validation_output" / "phase4_3_nested_router_outer_results.csv")
    cooked = []
    for row in rows:
        gate_mae = as_float(row["Gate_MAE"])
        policy_mae = as_float(row["Policy_MAE"])
        nested_mae = as_float(row["MAE"])
        better = min(gate_mae, policy_mae)
        worse = max(gate_mae, policy_mae)
        cooked.append(
            {
                "system": row["System"],
                "selected_rule": row["Selected_Rule"],
                "selected_choice": row["Selected_Choice"],
                "source": row["Source"],
                "context": row["Context"],
                "contains_la": row["Contains_La"] == "True",
                "ood_score": as_float(row["OOD_Score"]),
                "gate_mae": gate_mae,
                "gate_mae95": as_float(row["Gate_MAE95"]),
                "policy_mae": policy_mae,
                "policy_mae95": as_float(row["Policy_MAE95"]),
                "nested_mae": nested_mae,
                "nested_mae95": as_float(row["MAE95"]),
                "regret": nested_mae - better,
                "error_avoided": worse - nested_mae,
            }
        )
    return cooked


def load_phase4_summary():
    rows = read_csv(PHASE4_DIR / "phase4_3_nested_router_validation_output" / "phase4_3_nested_router_summary.csv")
    summary = rows[0]
    return {
        "mean_nested_router_mae": as_float(summary["mean_nested_router_mae"]),
        "mean_nested_router_mae95": as_float(summary["mean_nested_router_mae95"]),
        "mean_gate_mae": as_float(summary["mean_gate_mae"]),
        "mean_gate_mae95": as_float(summary["mean_gate_mae95"]),
        "mean_policy_mae": as_float(summary["mean_policy_mae"]),
        "mean_policy_mae95": as_float(summary["mean_policy_mae95"]),
    }


def load_phase3_uq_summary():
    rows = read_csv(PICTURE_DIR / "phase3_origin_data" / "phase3_uq_spearman_summary_for_origin.csv")
    short_label_map = {
        "raw_variance": "RawVar",
        "old_isotonic_uq": "Iso",
        "meta_quantile_q90": "MetaQ",
        "ood_knn_distance": "OODkNN",
    }
    return [
        {
            "method": row["method"],
            "label": row["method_label"],
            "short_label": short_label_map.get(row["method"], row["method_label"]),
            "median_spearman": as_float(row["median_spearman"]),
            "negative_or_nan_count": as_int(row["negative_or_nan_count"]),
            "n_systems": as_int(row["n_systems"]),
            "negative_or_nan_fraction": as_float(row["negative_or_nan_fraction"]),
        }
        for row in rows
    ]


def load_phase3_strategy_summary():
    rows = read_csv(PICTURE_DIR / "phase3_origin_data" / "phase3_strategy_summary_for_origin.csv")
    short_label_map = {
        "simple_ensemble_mean": "Ens",
        "reliability_weighted": "RelW",
        "context_adaptive": "Ctx",
        "only_high_ood_fallback": "Chem",
        "oracle_best_branch": "Oracle",
    }
    return [
        {
            "policy_variant": row["policy_variant"],
            "policy_label": row["policy_label"],
            "short_label": short_label_map.get(row["policy_variant"], row["policy_label"]),
            "mean_mae": as_float(row["mean_mae"]),
            "mean_mae95": as_float(row["mean_mae95"]),
            "mean_oracle_gap_mae": as_float(row["mean_oracle_gap_mae"]),
        }
        for row in rows
    ]


def load_recall_curves():
    rows = read_csv(PICTURE_DIR / "figure3_origin_data" / "figure3_budget_recall_all_systems_for_origin.csv")
    systems = [key for key in rows[0].keys() if key != "K"]
    ks = np.asarray([as_int(row["K"]) for row in rows], dtype=int)
    curves = {}
    for system in systems:
        curves[system] = np.asarray([as_float(row[system]) for row in rows], dtype=float)
    return ks, curves


def load_best_gap_curves():
    rows = read_csv(PICTURE_DIR / "figure3_origin_data" / "figure3_best_gap_main_systems_for_origin.csv")
    systems = [key for key in rows[0].keys() if key != "K"]
    ks = np.asarray([as_int(row["K"]) for row in rows], dtype=int)
    curves = {}
    for system in systems:
        curves[system] = np.asarray([as_float(row[system]) for row in rows], dtype=float)
    return ks, curves


def load_replay_summary():
    rows = read_csv(PICTURE_DIR / "figure3_origin_data" / "figure3_budget_recall_summary.csv")
    return {row["system"]: row for row in rows}


def load_bse9_selection():
    rows = read_csv(PHASE6_DIR / "02_dft_candidate_selection" / "phase6_final_dft_selection_BSe9.csv")
    cooked = []
    for row in rows:
        cooked.append(
            {
                "selection_rank": as_int(row["selection_rank"]),
                "sample_id": row["sample_id"],
                "selection_reason": row["selection_reason"],
                "replay_relative_energy": as_float(row["replay_relative_energy"]),
                "risk_score": as_float(row["risk_score"]),
                "dft_status": row.get("dft_status", ""),
                "dft_relative_energy_eV": as_float(row.get("dft_relative_energy_eV")),
                "dft_hit_delta_0p10": row.get("dft_hit_delta_0p10", ""),
            }
        )
    cooked.sort(key=lambda item: item["selection_rank"])
    return cooked


def plot_figure1_workflow(dataset_snapshot):
    apply_publication_style(font_size=8, axes_linewidth=0.8)
    fig = plt.figure(figsize=(7.2, 5.2))
    ax = fig.add_axes([0.02, 0.03, 0.96, 0.94])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.02,
        0.97,
        "Nested Chemistry-Routed Stacking workflow",
        fontsize=11,
        fontweight="bold",
        ha="left",
        va="top",
    )
    ax.text(
        0.02,
        0.92,
        "Leakage-safe LOSO training feeds nested routing, replay screening, and finite-budget DFT validation.",
        fontsize=7.0,
        ha="left",
        va="top",
        color=PALETTE["neutral_dark"],
    )
    n_logs = as_int(dataset_snapshot["n_logs"])
    n_samples = as_int(dataset_snapshot["n_samples"])

    hero_y = 0.53
    hero_h = 0.24
    hero_w = 0.168
    hero_x = [0.015, 0.212, 0.409, 0.606, 0.803]
    hero_colors = [
        PALETTE["aqua"],
        PALETTE["baseline_soft"],
        PALETTE["lilac"],
        PALETTE["hero_mid"],
        lighten(PALETTE["callout_red"], 0.72),
    ]
    titles = [
        "1. Dataset",
        "2. Leakage-safe split",
        "3. Branch stack",
        "4. Nested router",
        "5. DFT validation",
    ]
    bodies = [
        f"{n_logs:,} Gaussian logs\n{n_samples:,} graph samples\n7 chemical systems",
        "Outer LOSO by system\nGroup-aware inner split\nNo held-out label tuning",
        "SchNet and PaiNN baselines\nPhysics-aware branches\nSystem-specific error probing",
        "Gate versus policy\nInner-only rule search\nHeld-out system never used",
        "Replay screening\n30-candidate DFT\nReturned-log audit",
    ]
    wrap_widths = [20, 20, 20, 20, 16]

    for idx, x0 in enumerate(hero_x):
        box = mpatches.FancyBboxPatch(
            (x0, hero_y),
            hero_w,
            hero_h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1.0,
            edgecolor=PALETTE["neutral_dark"],
            facecolor=hero_colors[idx],
        )
        ax.add_patch(box)
        ax.text(
            x0 + 0.015,
            hero_y + hero_h - 0.03,
            titles[idx],
            fontsize=7.8 if idx == 4 else 8.2,
            fontweight="bold",
            ha="left",
            va="top",
        )
        ax.text(
            x0 + 0.015,
            hero_y + hero_h - 0.075,
            wrap_lines(bodies[idx], wrap_widths[idx]),
            fontsize=6.25 if idx == 4 else 6.45,
            ha="left",
            va="top",
            linespacing=1.22,
        )

    for idx in range(len(hero_x) - 1):
        start_x = hero_x[idx] + hero_w
        end_x = hero_x[idx + 1]
        arrow = mpatches.FancyArrowPatch(
            (start_x + 0.01, hero_y + hero_h / 2.0),
            (end_x - 0.01, hero_y + hero_h / 2.0),
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.0,
            color=PALETTE["neutral_dark"],
        )
        ax.add_patch(arrow)

    ax.text(0.02, 0.44, "Supporting evidence and claim boundaries", fontsize=8.3, fontweight="bold", ha="left", va="bottom")

    card_specs = [
        (
            0.02,
            0.15,
            0.30,
            0.22,
            PALETTE["band_aqua"],
            "Main-text systems",
            "LaCu$_{12}^{-}$: extreme La OOD\nLaSi$_{9}^{-}$: routed gain\nBSe$_{9}^{-}$: non-La OOD control",
        ),
        (
            0.35,
            0.15,
            0.29,
            0.22,
            PALETTE["band_lilac"],
            "Performance metrics",
            "MAE and MAE95 for cross-system prediction, plus Recall@K, Best-of-K gap, Budget-to-hit, and contamination for finite-budget screening.",
        ),
        (
            0.67,
            0.15,
            0.31,
            0.22,
            PALETTE["band_peach"],
            "Claim boundary",
            "Replay ranking is not identical to post-optimization DFT ranking. BSe$_{9}^{-}$ can only be framed as basin recovery after relaxation, not raw top-K success.",
        ),
    ]

    for x0, y0, width, height, color, title, body in card_specs:
        box = mpatches.FancyBboxPatch(
            (x0, y0),
            width,
            height,
            boxstyle="round,pad=0.015,rounding_size=0.02",
            linewidth=0.8,
            edgecolor=PALETTE["neutral_dark"],
            facecolor=color,
        )
        ax.add_patch(box)
        ax.text(x0 + 0.015, y0 + height - 0.03, title, fontsize=8.0, fontweight="bold", ha="left", va="top")
        body_fontsize = 6.55 if title == "Main-text systems" else 6.9
        body_linespacing = 1.22 if title == "Main-text systems" else 1.35
        ax.text(x0 + 0.015, y0 + height - 0.08, wrap_lines(body, 34), fontsize=body_fontsize, ha="left", va="top", linespacing=body_linespacing)

    ribbon_y = 0.83
    ribbon_text = [
        ("LOSO outer fold", PALETTE["band_lilac"]),
        ("Nested inner rule selection", PALETTE["band_aqua"]),
        ("Replay and DFT split reporting", PALETTE["band_peach"]),
    ]
    ribbon_x = [0.02, 0.365, 0.70]
    ribbon_w = [0.27, 0.28, 0.25]
    for idx, (label, color) in enumerate(ribbon_text):
        box = mpatches.FancyBboxPatch(
            (ribbon_x[idx], ribbon_y),
            ribbon_w[idx],
            0.038,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            linewidth=0.0,
            facecolor=color,
        )
        ax.add_patch(box)
        ax.text(ribbon_x[idx] + ribbon_w[idx] / 2.0, ribbon_y + 0.019, label, fontsize=6.5, ha="center", va="center")

    source_rows = [
        {"key": "n_logs", "value": dataset_snapshot["n_logs"]},
        {"key": "n_samples", "value": dataset_snapshot["n_samples"]},
        {"key": "main_systems", "value": ", ".join(MAIN_SYSTEMS)},
        {"key": "si_systems", "value": ", ".join(SI_SYSTEMS)},
    ]
    write_csv(SOURCE_DIR / "Figure1_workflow_source_data.csv", ["key", "value"], source_rows)

    return save_figure(fig, "Figure1_workflow_publication", width_mm=183, height_mm=132)


def plot_figure2_nested_router(outer_rows, summary):
    apply_publication_style(font_size=7, axes_linewidth=0.9)
    fig = plt.figure(figsize=(7.2, 5.6))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.65, 1.0], height_ratios=[1.0, 0.9], wspace=0.26, hspace=0.34)

    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])

    order = ["AuB8", "AgB8", "BSe9", "LaB8", "LaSe8", "LaSi9", "LaCu12"]
    rows_by_system = {row["system"]: row for row in outer_rows}
    ordered_rows = [rows_by_system[system] for system in order]
    y_positions = np.arange(len(order))[::-1]

    ax_a.axhspan(3.5, 6.5, color=PALETTE["band_aqua"], zorder=0)
    ax_a.axhspan(-0.5, 3.5, color=PALETTE["band_lilac"], zorder=0)

    for y_value, row in zip(y_positions, ordered_rows):
        ax_a.plot(
            [row["policy_mae"], row["gate_mae"]],
            [y_value, y_value],
            color=PALETTE["neutral_mid"],
            linewidth=1.1,
            zorder=1,
        )
        ax_a.scatter(row["gate_mae"], y_value + 0.12, s=32, marker="s", color=METHOD_COLORS["gate"], zorder=3)
        ax_a.scatter(row["policy_mae"], y_value - 0.12, s=34, marker="^", color=METHOD_COLORS["policy"], zorder=3)
        ax_a.scatter(
            row["nested_mae"],
            y_value,
            s=60,
            marker="o",
            color=METHOD_COLORS["nested"],
            edgecolor=PALETTE["text_dark"],
            linewidth=0.5,
            zorder=4,
        )

    ax_a.set_yticks(y_positions)
    ax_a.set_yticklabels([SYSTEM_LABELS.get(system, system) for system in order])
    for tick, system in zip(ax_a.get_yticklabels(), order):
        if system in MAIN_SYSTEMS:
            tick.set_fontweight("bold")
            tick.set_color(SYSTEM_COLORS[system])
    ax_a.set_xlabel("MAE (eV, lower is better)")
    ax_a.set_title(
        "Per-system outer-held-out MAE\n"
        f"Mean MAE: nested {summary['mean_nested_router_mae']:.2f} eV, "
        f"gate {summary['mean_gate_mae']:.2f} eV, policy {summary['mean_policy_mae']:.2f} eV",
        fontsize=7.4,
        loc="left",
        pad=6,
    )
    ax_a.grid(axis="x", alpha=0.22, linewidth=0.6)
    ax_a.set_xlim(0.0, 16.8)
    gate_systems = format_system_list([row["system"] for row in ordered_rows if row["selected_choice"] == "gate"])
    policy_systems = format_system_list([row["system"] for row in ordered_rows if row["selected_choice"] == "policy"])
    ax_a.text(
        0.0,
        -0.16,
        f"Nested gate: {gate_systems};  nested policy: {policy_systems}.",
        transform=ax_a.transAxes,
        fontsize=5.2,
        ha="left",
        va="top",
        color=PALETTE["neutral_dark"],
        clip_on=False,
    )

    legend_handles = [
        mlines.Line2D([], [], color=METHOD_COLORS["gate"], marker="s", linestyle="None", markersize=5, label="Gate"),
        mlines.Line2D([], [], color=METHOD_COLORS["policy"], marker="^", linestyle="None", markersize=5, label="Policy"),
        mlines.Line2D(
            [], [], color=METHOD_COLORS["nested"], marker="o", markeredgecolor=PALETTE["text_dark"], linestyle="None", markersize=6, label="Nested"
        ),
    ]
    ax_a.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.995),
        fontsize=5.4,
        borderaxespad=0.1,
        handletextpad=0.5,
        labelspacing=0.45,
    )
    add_panel_label(ax_a, "a")

    metric_labels = ["Gate", "Policy", "Nested"]
    metric_positions = np.arange(len(metric_labels))
    mae_values = [summary["mean_gate_mae"], summary["mean_policy_mae"], summary["mean_nested_router_mae"]]
    mae95_values = [summary["mean_gate_mae95"], summary["mean_policy_mae95"], summary["mean_nested_router_mae95"]]
    colors = [METHOD_COLORS["gate"], METHOD_COLORS["policy"], METHOD_COLORS["nested"]]
    width = 0.34

    bars_mae = ax_b.bar(metric_positions - width / 2.0, mae_values, width=width, color=colors, edgecolor="black", linewidth=0.5, label="MAE")
    bars_mae95 = ax_b.bar(
        metric_positions + width / 2.0,
        mae95_values,
        width=width,
        color=[lighten(color, 0.35) for color in colors],
        edgecolor="black",
        linewidth=0.5,
        hatch="//",
        label="MAE95",
    )
    for bar in list(bars_mae) + list(bars_mae95):
        ax_b.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.10,
            f"{bar.get_height():.2f}",
            fontsize=6.0,
            ha="center",
            va="bottom",
        )
    ax_b.set_xticks(metric_positions)
    ax_b.set_xticklabels(metric_labels)
    ax_b.set_ylabel("Mean error (eV)")
    ax_b.set_title("Workflow summary")
    ax_b.grid(axis="y", alpha=0.22, linewidth=0.6)
    ax_b.text(
        0.98,
        0.97,
        "solid: MAE   hatched: MAE95",
        transform=ax_b.transAxes,
        fontsize=5.4,
        ha="right",
        va="top",
        color="black",
    )
    add_panel_label(ax_b, "b")

    regrets = [row["regret"] for row in ordered_rows]
    ax_c.axvline(0.0, color=PALETTE["neutral_mid"], linewidth=0.8, linestyle="--")
    for y_value, row in zip(y_positions, ordered_rows):
        if row["regret"] > 1e-8:
            color = PALETTE["callout_red"]
            ax_c.hlines(y_value, 0.0, row["regret"], color=color, linewidth=1.3, alpha=0.95)
            ax_c.scatter(row["regret"], y_value, s=28, color=color, zorder=3)
        else:
            color = lighten(SYSTEM_COLORS[row["system"]], 0.35)
            ax_c.scatter(row["regret"], y_value, s=18, color=color, zorder=3)
        if row["regret"] > 0.05:
            ax_c.text(row["regret"] + 0.06, y_value, f"{row['regret']:.2f}", fontsize=6.0, ha="left", va="center")
    ax_c.set_yticks(y_positions)
    ax_c.set_yticklabels([SYSTEM_LABELS.get(system, system) for system in order])
    for tick, system in zip(ax_c.get_yticklabels(), order):
        if system in MAIN_SYSTEMS:
            tick.set_fontweight("bold")
            tick.set_color(SYSTEM_COLORS[system])
    ax_c.set_xlabel("Regret vs lower-error standalone branch (eV)")
    ax_c.text(
        -0.07,
        1.12,
        "c",
        transform=ax_c.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
    )
    ax_c.text(
        0.0,
        1.12,
        "Router misses appear only in SI systems",
        transform=ax_c.transAxes,
        fontsize=7.5,
        ha="left",
        va="bottom",
        clip_on=False,
    )
    ax_c.grid(axis="x", alpha=0.22, linewidth=0.6)
    ax_c.set_xlim(-0.02, max(regrets) + 0.45)

    source_rows = []
    for row in ordered_rows:
        source_rows.append(
            {
                "system": row["system"],
                "context": row["context"],
                "ood_score": f"{row['ood_score']:.10f}",
                "gate_mae": f"{row['gate_mae']:.10f}",
                "gate_mae95": f"{row['gate_mae95']:.10f}",
                "policy_mae": f"{row['policy_mae']:.10f}",
                "policy_mae95": f"{row['policy_mae95']:.10f}",
                "nested_mae": f"{row['nested_mae']:.10f}",
                "nested_mae95": f"{row['nested_mae95']:.10f}",
                "selected_choice": row["selected_choice"],
                "selected_rule": row["selected_rule"],
                "regret_vs_better_standalone": f"{row['regret']:.10f}",
            }
        )
    write_csv(
        SOURCE_DIR / "Figure2_nested_router_source_data.csv",
        [
            "system",
            "context",
            "ood_score",
            "gate_mae",
            "gate_mae95",
            "policy_mae",
            "policy_mae95",
            "nested_mae",
            "nested_mae95",
            "selected_choice",
            "selected_rule",
            "regret_vs_better_standalone",
        ],
        source_rows,
    )

    return save_figure(fig, "Figure2_nested_router_publication", width_mm=183, height_mm=140)


def plot_figure3_screening_and_dft(recall_ks, recall_curves, gap_ks, gap_curves, replay_summary, bse9_rows):
    apply_publication_style(font_size=7, axes_linewidth=0.9)
    fig = plt.figure(figsize=(7.2, 5.95))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.55, 1.0], height_ratios=[1.0, 0.95], wspace=0.26, hspace=0.36)

    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])

    display_order = ["AgB8", "AuB8", "LaB8", "LaSe8", "LaCu12", "LaSi9", "BSe9"]
    si_styles = {
        "AgB8": {"color": "#8D98C8", "linestyle": "--"},
        "AuB8": {"color": "#C0CAE6", "linestyle": "--"},
        "LaB8": {"color": "#B5B5B5", "linestyle": "--"},
        "LaSe8": {"color": "#7D7D7D", "linestyle": "--"},
    }
    for system in display_order:
        linewidth = 2.0 if system in MAIN_SYSTEMS else 1.2
        linestyle = "-" if system in MAIN_SYSTEMS else si_styles[system]["linestyle"]
        alpha = 0.95 if system in MAIN_SYSTEMS else 0.85
        ax_a.plot(
            recall_ks,
            recall_curves[system],
            color=SYSTEM_COLORS[system] if system in MAIN_SYSTEMS else si_styles[system]["color"],
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=alpha,
            label=system,
        )

    ax_a.set_xlabel("Budget K")
    ax_a.set_ylabel("Recall of replay structures within 0.10 eV")
    ax_a.set_title("Budget recall depends strongly on system chemistry", loc="left", fontsize=7.5, pad=6)
    ax_a.set_xlim(1, 100)
    ax_a.set_ylim(-0.02, 1.05)
    ax_a.grid(alpha=0.22, linewidth=0.6)
    ax_a.legend(loc="upper left", fontsize=6.0, ncol=2)
    add_panel_label(ax_a, "a")

    label_offsets = {"LaCu12": 0.05, "LaSi9": -0.06, "BSe9": 0.16}
    label_x = {"LaCu12": 99.0, "LaSi9": 95.0, "BSe9": 95.0}
    for system in MAIN_SYSTEMS:
        ax_b.plot(gap_ks, gap_curves[system], color=SYSTEM_COLORS[system], linewidth=2.0, label=system)
        end_value = gap_curves[system][-1]
        ax_b.text(
            label_x[system],
            end_value + label_offsets[system],
            f"{system} {end_value:.2f}",
            fontsize=5.6,
            ha="right",
            va="center",
            color=SYSTEM_COLORS[system],
        )

    ax_b.axhline(0.10, color=PALETTE["neutral_mid"], linewidth=0.8, linestyle="--")
    bse9_hit_rank = as_int(replay_summary["BSe9"]["budget_to_first_hit"])
    ax_b.scatter(bse9_hit_rank, 0.0, s=28, color=SYSTEM_COLORS["BSe9"], zorder=3)
    ax_b.annotate(
        "BSe9 hit at K=73",
        xy=(bse9_hit_rank, 0.0),
        xytext=(48, 0.18),
        textcoords="data",
        fontsize=5.2,
        ha="left",
        va="bottom",
        color=SYSTEM_COLORS["BSe9"],
        arrowprops={"arrowstyle": "-", "lw": 0.7, "color": SYSTEM_COLORS["BSe9"]},
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.85},
    )
    ax_b.text(
        40,
        0.115,
        "0.10 eV",
        fontsize=5.2,
        ha="left",
        va="bottom",
        color=PALETTE["neutral_dark"],
        bbox={"boxstyle": "round,pad=0.14", "facecolor": "white", "edgecolor": "none", "alpha": 0.85},
    )
    ax_b.set_xlabel("Budget K")
    ax_b.set_ylabel("Best-of-K replay gap (eV)")
    ax_b.set_title("Main-system replay gap")
    ax_b.set_xlim(1, 100)
    ax_b.set_ylim(-0.02, 1.28)
    ax_b.grid(alpha=0.22, linewidth=0.6)
    add_panel_label(ax_b, "b")

    ranks = np.asarray([row["selection_rank"] for row in bse9_rows], dtype=int)
    replay_gap = np.asarray([row["replay_relative_energy"] for row in bse9_rows], dtype=float)
    dft_gap = np.asarray([row["dft_relative_energy_eV"] for row in bse9_rows], dtype=float)
    success_mask = np.asarray([row["dft_status"] == "dft_success" for row in bse9_rows], dtype=bool)
    missing_mask = np.asarray([row["dft_status"] == "resubmitted_or_missing_log" for row in bse9_rows], dtype=bool)
    hit_mask = success_mask & np.isfinite(dft_gap) & (dft_gap <= 0.10)

    reason_spans = [(1, 21, "low_predicted_energy"), (22, 27, "low_energy_high_risk"), (28, 30, "diversity_fill")]
    for start, end, reason in reason_spans:
        ax_c.axvspan(start - 0.5, end + 0.5, ymin=0.00, ymax=0.06, color=SELECTION_REASON_COLORS[reason], alpha=0.18, linewidth=0.0)

    replay_cummin = np.minimum.accumulate(replay_gap)
    dft_for_cummin = np.where(success_mask & np.isfinite(dft_gap), dft_gap, np.inf)
    dft_cummin = np.minimum.accumulate(dft_for_cummin)
    dft_cummin[~np.isfinite(dft_cummin)] = np.nan

    ax_c.step(ranks, replay_cummin, where="post", color=PALETTE["hero_mid"], linewidth=1.8, zorder=2)
    ax_c.step(ranks, dft_cummin, where="post", color=PALETTE["callout_green"], linewidth=2.0, zorder=3)
    ax_c.scatter(ranks[success_mask], dft_gap[success_mask], s=24, marker="o", facecolor="white", edgecolor=PALETTE["neutral_dark"], linewidth=0.7, zorder=4)
    ax_c.scatter(ranks[hit_mask], dft_gap[hit_mask], s=32, marker="o", facecolor=PALETTE["callout_green"], edgecolor=PALETTE["text_dark"], linewidth=0.6, zorder=5)

    ax_c.axhline(0.10, color=PALETTE["neutral_mid"], linewidth=0.8, linestyle="--")
    ax_c.set_xlabel("BSe9 selection rank in 30-candidate queue")
    ax_c.set_ylabel("Relative energy gap (eV)")
    ax_c.text(
        -0.07,
        1.13,
        "c",
        transform=ax_c.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
    )
    ax_c.text(
        0.0,
        1.13,
        "BSe9: weak raw replay, useful basin after DFT",
        transform=ax_c.transAxes,
        fontsize=7.5,
        ha="left",
        va="bottom",
        clip_on=False,
    )
    ax_c.set_xlim(0.5, 30.5)
    ax_c.set_ylim(-0.03, 1.24)
    ax_c.grid(alpha=0.22, linewidth=0.6)

    completed_n = int(np.sum(success_mask))
    missing_n = int(np.sum(missing_mask))
    hit_n = int(np.sum(hit_mask))
    ax_c.text(
        0.02,
        0.98,
        f"{completed_n}/30 completed\n{missing_n} pending or resubmitted\n{hit_n} hits <= 0.10 eV",
        transform=ax_c.transAxes,
        fontsize=5.2,
        ha="left",
        va="top",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": PALETTE["neutral_light"], "linewidth": 0.6},
    )
    legend_handles = [
        mlines.Line2D([], [], color="none", marker="o", markerfacecolor="white", markeredgecolor=PALETTE["neutral_dark"], markersize=5.2, label="DFT"),
        mlines.Line2D([], [], color="none", marker="o", markerfacecolor=PALETTE["callout_green"], markeredgecolor=PALETTE["text_dark"], markersize=5.2, label="<=0.10 hit"),
        mlines.Line2D([], [], color=PALETTE["hero_mid"], linewidth=1.8, label="Replay best"),
        mlines.Line2D([], [], color=PALETTE["callout_green"], linewidth=2.0, label="DFT best"),
    ]
    ax_c.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        fontsize=5.0,
        borderaxespad=0.15,
        handletextpad=0.45,
        labelspacing=0.30,
    )

    ax_c.text(6.0, 0.045, "pred", fontsize=4.9, ha="center", va="center")
    ax_c.text(23.8, 0.045, "risk", fontsize=4.9, ha="center", va="center")
    ax_c.text(29.0, 0.045, "div", fontsize=4.9, ha="center", va="center")

    source_rows = []
    for system in display_order:
        summary_row = replay_summary[system]
        source_rows.append(
            {
                "panel": "a",
                "system": system,
                "n_samples": summary_row["n_samples"],
                "n_reference_low_energy": summary_row["n_reference_low_energy"],
                "recall_at_30": summary_row["recall_at_30"],
                "recall_at_100": summary_row["recall_at_100"],
                "best_gap_at_30": summary_row["best_gap_at_30"],
                "best_gap_at_100": summary_row["best_gap_at_100"],
                "budget_to_first_hit": summary_row["budget_to_first_hit"],
            }
        )
    write_csv(
        SOURCE_DIR / "Figure3_replay_and_dft_source_data.csv",
        [
            "panel",
            "system",
            "n_samples",
            "n_reference_low_energy",
            "recall_at_30",
            "recall_at_100",
            "best_gap_at_30",
            "best_gap_at_100",
            "budget_to_first_hit",
        ],
        source_rows,
    )

    bse9_source_rows = []
    for row in bse9_rows:
        bse9_source_rows.append(
            {
                "selection_rank": row["selection_rank"],
                "sample_id": row["sample_id"],
                "selection_reason": row["selection_reason"],
                "replay_relative_energy": f"{row['replay_relative_energy']:.10f}",
                "dft_status": row["dft_status"],
                "dft_relative_energy_eV": "" if not np.isfinite(row["dft_relative_energy_eV"]) else f"{row['dft_relative_energy_eV']:.10f}",
                "risk_score": f"{row['risk_score']:.10f}",
            }
        )
    write_csv(
        SOURCE_DIR / "Figure3_BSe9_ranked_validation_source_data.csv",
        ["selection_rank", "sample_id", "selection_reason", "replay_relative_energy", "dft_status", "dft_relative_energy_eV", "risk_score"],
        bse9_source_rows,
    )

    return save_figure(fig, "Figure3_replay_and_dft_publication", width_mm=183, height_mm=145)


def plot_supplementary_uq(uq_rows):
    apply_publication_style(font_size=7, axes_linewidth=0.9)
    fig = plt.figure(figsize=(7.2, 3.1))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.2, 0.9], wspace=0.28)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    labels = [row["short_label"] for row in uq_rows]
    x_positions = np.arange(len(uq_rows))
    medians = [row["median_spearman"] for row in uq_rows]
    fractions = [row["negative_or_nan_fraction"] for row in uq_rows]
    counts = [row["negative_or_nan_count"] for row in uq_rows]
    systems = [row["n_systems"] for row in uq_rows]

    colors = []
    for value in medians:
        colors.append(PALETTE["callout_green"] if value > 0 else PALETTE["neutral_mid"])
    ax_a.bar(x_positions, medians, color=colors, edgecolor="black", linewidth=0.5)
    ax_a.axhline(0.0, color=PALETTE["neutral_mid"], linewidth=0.8, linestyle="--")
    for xpos, value in zip(x_positions, medians):
        ax_a.text(xpos, value + (0.02 if value >= 0 else -0.04), f"{value:.2f}", fontsize=6.0, ha="center", va="bottom" if value >= 0 else "top")
    ax_a.set_xticks(x_positions)
    ax_a.set_xticklabels(labels)
    ax_a.set_ylabel("Median Spearman")
    ax_a.set_title("Scalar-UQ rank correlation")
    ax_a.set_ylim(-0.35, 0.15)
    ax_a.grid(axis="y", alpha=0.22, linewidth=0.6)
    add_panel_label(ax_a, "a")

    ax_b.bar(x_positions, fractions, color=PALETTE["hero_mid"], edgecolor="black", linewidth=0.5)
    for xpos, fraction, count, total in zip(x_positions, fractions, counts, systems):
        ax_b.text(xpos, fraction + 0.03, f"{count}/{total}", fontsize=6.0, ha="center", va="bottom")
    ax_b.set_xticks(x_positions)
    ax_b.set_xticklabels(labels)
    ax_b.set_ylabel("Negative or undefined fraction")
    ax_b.set_title("Failure burden across systems")
    ax_b.set_ylim(0.0, 1.02)
    ax_b.grid(axis="y", alpha=0.22, linewidth=0.6)
    add_panel_label(ax_b, "b")

    source_rows = []
    for row in uq_rows:
        source_rows.append(
            {
                "short_label": row["short_label"],
                "method_label": row["label"],
                "median_spearman": f"{row['median_spearman']:.10f}",
                "negative_or_nan_count": row["negative_or_nan_count"],
                "n_systems": row["n_systems"],
                "negative_or_nan_fraction": f"{row['negative_or_nan_fraction']:.10f}",
            }
        )
    write_csv(
        SOURCE_DIR / "FigureS1_scalar_uq_source_data.csv",
        ["short_label", "method_label", "median_spearman", "negative_or_nan_count", "n_systems", "negative_or_nan_fraction"],
        source_rows,
    )

    return save_figure(fig, "FigureS1_scalar_uq_diagnostic_publication", width_mm=183, height_mm=78)


def plot_supplementary_strategy(strategy_rows):
    apply_publication_style(font_size=7, axes_linewidth=0.9)
    fig = plt.figure(figsize=(7.2, 3.25))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.25, 0.95], wspace=0.30)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    labels = [row["short_label"] for row in strategy_rows]
    x_positions = np.arange(len(strategy_rows))
    mae_values = [row["mean_mae"] for row in strategy_rows]
    mae95_values = [row["mean_mae95"] for row in strategy_rows]
    oracle_gap = [row["mean_oracle_gap_mae"] for row in strategy_rows]

    for xpos, row in zip(x_positions, strategy_rows):
        highlight = row["policy_variant"] == "only_high_ood_fallback"
        color = PALETTE["hero_dark"] if highlight else (PALETTE["neutral_mid"] if row["policy_variant"] != "oracle_best_branch" else PALETTE["baseline_soft"])
        ax_a.plot([xpos, xpos], [row["mean_mae"], row["mean_mae95"]], color=color, linewidth=1.4)
        ax_a.scatter(xpos, row["mean_mae"], s=34, color=color, edgecolor=PALETTE["text_dark"], linewidth=0.35, zorder=3)
        ax_a.scatter(xpos, row["mean_mae95"], s=34, color="white", edgecolor=color, linewidth=1.0, zorder=3)
        ax_a.text(xpos, row["mean_mae95"] + 0.10, f"{row['mean_mae']:.2f}/{row['mean_mae95']:.2f}", fontsize=5.9, ha="center", va="bottom")
    ax_a.set_xticks(x_positions)
    ax_a.set_xticklabels(labels)
    ax_a.set_ylabel("Error (eV)")
    ax_a.set_title("Strategy summary")
    ax_a.grid(axis="y", alpha=0.22, linewidth=0.6)
    add_panel_label(ax_a, "a")

    bar_colors = []
    for row in strategy_rows:
        if row["policy_variant"] == "only_high_ood_fallback":
            bar_colors.append(PALETTE["hero_dark"])
        elif row["policy_variant"] == "oracle_best_branch":
            bar_colors.append(PALETTE["baseline_soft"])
        else:
            bar_colors.append(PALETTE["neutral_mid"])
    ax_b.bar(x_positions, oracle_gap, color=bar_colors, edgecolor="black", linewidth=0.5)
    for xpos, value in zip(x_positions, oracle_gap):
        ax_b.text(xpos, value + 0.02, f"{value:.2f}", fontsize=6.0, ha="center", va="bottom")
    ax_b.set_xticks(x_positions)
    ax_b.set_xticklabels(labels)
    ax_b.set_ylabel("Oracle-gap MAE (eV)")
    ax_b.set_title("Distance from oracle branch")
    ax_b.grid(axis="y", alpha=0.22, linewidth=0.6)
    add_panel_label(ax_b, "b")

    source_rows = []
    for row in strategy_rows:
        source_rows.append(
            {
                "short_label": row["short_label"],
                "policy_label": row["policy_label"],
                "mean_mae": f"{row['mean_mae']:.10f}",
                "mean_mae95": f"{row['mean_mae95']:.10f}",
                "mean_oracle_gap_mae": f"{row['mean_oracle_gap_mae']:.10f}",
            }
        )
    write_csv(
        SOURCE_DIR / "FigureS2_strategy_source_data.csv",
        ["short_label", "policy_label", "mean_mae", "mean_mae95", "mean_oracle_gap_mae"],
        source_rows,
    )

    return save_figure(fig, "FigureS2_strategy_diagnostic_publication", width_mm=183, height_mm=82)


def main():
    ensure_dirs()
    contract_path = write_figure_contracts()

    dataset_snapshot = load_dataset_snapshot()
    phase4_outer = load_phase4_outer_results()
    phase4_summary = load_phase4_summary()
    uq_rows = load_phase3_uq_summary()
    strategy_rows = load_phase3_strategy_summary()
    recall_ks, recall_curves = load_recall_curves()
    gap_ks, gap_curves = load_best_gap_curves()
    replay_summary = load_replay_summary()
    bse9_rows = load_bse9_selection()

    outputs = {
        "Figure1": plot_figure1_workflow(dataset_snapshot),
        "Figure2": plot_figure2_nested_router(phase4_outer, phase4_summary),
        "Figure3": plot_figure3_screening_and_dft(recall_ks, recall_curves, gap_ks, gap_curves, replay_summary, bse9_rows),
        "FigureS1": plot_supplementary_uq(uq_rows),
        "FigureS2": plot_supplementary_strategy(strategy_rows),
    }

    manifest_rows = []
    for figure_name, files in outputs.items():
        for fmt, path in files.items():
            manifest_rows.append({"figure": figure_name, "format": fmt, "path": path})
    write_csv(OUT_DIR / "publication_figure_manifest.csv", ["figure", "format", "path"], manifest_rows)

    print(f"Figure contracts: {contract_path}")
    for figure_name, files in outputs.items():
        print(f"{figure_name}:")
        for fmt, path in files.items():
            print(f"  {fmt}: {path}")


if __name__ == "__main__":
    main()
