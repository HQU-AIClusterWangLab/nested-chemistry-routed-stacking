import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "picture" / "publication_figures"
SOURCE_DIR = FIG_DIR / "source_data"
EDGE_TABLE = (
    ROOT
    / "phase_6"
    / "final_nested_router"
    / "04_computational_chemistry_plan"
    / "tables"
    / "bond_order_attention_edge_template.csv"
)

SYSTEMS = ["BSe9", "LaSi9"]
SYSTEM_TITLES = {
    "BSe9": "BSe$_9$ (policy branch)",
    "LaSi9": "LaSi$_9$ (gate branch)",
}
SYSTEM_COLORS = {
    "BSe9": "#D65F5F",
    "LaSi9": "#4AAAEF",
}
METRICS = [
    ("gate_mean", "Gate mean"),
    ("scale_mean", "Scale mean"),
    ("hist", "Mayer distribution"),
]


def apply_style():
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.size"] = 7
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"


def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_float(value):
    if value is None:
        return math.nan
    text = str(value).strip()
    if not text:
        return math.nan
    return float(text)


def pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return math.nan
    x0 = x - x.mean()
    y0 = y - y.mean()
    denom = np.sqrt((x0**2).sum() * (y0**2).sum())
    if denom == 0:
        return math.nan
    return float((x0 * y0).sum() / denom)


def rankdata(values):
    values = list(values)
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        i = j + 1
    return np.asarray(ranks, dtype=float)


def spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return math.nan
    return pearson(rankdata(x), rankdata(y))


def load_edges():
    rows = []
    for row in read_csv(EDGE_TABLE):
        system = row.get("system_id", "")
        if system not in SYSTEMS:
            continue
        mayer = parse_float(row.get("mayer_bond_order"))
        gate = parse_float(row.get("gate_mean"))
        scale = parse_float(row.get("scale_mean"))
        if not np.isfinite(mayer) or not np.isfinite(gate) or not np.isfinite(scale):
            continue
        rows.append(
            {
                "system": system,
                "mayer_bond_order": mayer,
                "gate_mean": gate,
                "scale_mean": scale,
                "bond_mean_diff": parse_float(row.get("bond_mean_diff")),
            }
        )
    return rows


def deduplicate_symmetric_edges(rows):
    dedup = {}
    for row in rows:
        key = (
            row["system"],
            round(row["mayer_bond_order"], 8),
            round(row["gate_mean"], 8),
            round(row["scale_mean"], 8),
            round(row["bond_mean_diff"], 8) if np.isfinite(row["bond_mean_diff"]) else "",
        )
        dedup.setdefault(key, row)
    return list(dedup.values())


def add_panel_label(ax, label):
    ax.text(
        -0.18,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
    )


def plot_scatter(ax, rows, system, metric):
    color = SYSTEM_COLORS[system]
    x = np.asarray([r[metric] for r in rows], dtype=float)
    y = np.asarray([r["mayer_bond_order"] for r in rows], dtype=float)
    ax.scatter(x, y, s=13, color=color, alpha=0.82, edgecolors="white", linewidths=0.25)
    coeff = np.polyfit(x, y, 1)
    xline = np.linspace(np.nanmin(x), np.nanmax(x), 100)
    ax.plot(xline, coeff[0] * xline + coeff[1], color="#9A9A9A", linewidth=0.8, linestyle="--")
    rho = spearman(x, y)
    r = pearson(x, y)
    ax.text(
        0.97,
        0.97,
        f"$\\rho$={rho:.2f}, r={r:.2f}\nn={len(rows)}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.8,
        color="#2B2B2B",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.2, "alpha": 0.86},
    )
    ax.set_xlabel("Mayer bond order", fontsize=7.2)
    if metric == "gate_mean":
        ax.set_ylabel("Gate mean", fontsize=7.2)
    else:
        ax.set_ylabel("Scale mean", fontsize=7.2)
    ax.grid(alpha=0.14, linewidth=0.45)
    ax.tick_params(labelsize=6.7, width=0.7, length=2.2)
    return {
        "system": system,
        "metric": metric,
        "spearman_rho": f"{rho:.6f}",
        "pearson_r": f"{r:.6f}",
        "n_unique_edges": str(len(rows)),
    }


def plot_hist(ax, rows, system):
    color = SYSTEM_COLORS[system]
    x = np.asarray([r["mayer_bond_order"] for r in rows], dtype=float)
    bins = np.linspace(0.0, 1.15, 18)
    ax.hist(x, bins=bins, color=color, alpha=0.82, edgecolor="white", linewidth=0.45)
    ax.text(
        0.97,
        0.97,
        f"{len(rows)} bonds",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.8,
        color="#2B2B2B",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.2, "alpha": 0.86},
    )
    ax.set_xlabel("Mayer bond order", fontsize=7.2)
    ax.set_ylabel("Count", fontsize=7.2)
    ax.grid(axis="y", alpha=0.14, linewidth=0.45)
    ax.tick_params(labelsize=6.7, width=0.7, length=2.2)


def save_figure(fig):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    base = FIG_DIR / "FigureS3_mayer_vs_attention"
    fig.savefig(f"{base}.svg", bbox_inches="tight")
    fig.savefig(f"{base}.pdf", bbox_inches="tight")
    fig.savefig(f"{base}.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(f"{base}.tif", dpi=600, bbox_inches="tight")
    fig.savefig(f"{base}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    apply_style()
    raw_rows = load_edges()
    rows = deduplicate_symmetric_edges(raw_rows)

    source_rows = []
    for row in rows:
        source_rows.append(
            {
                "system": row["system"],
                "mayer_bond_order": f"{row['mayer_bond_order']:.6f}",
                "gate_mean": f"{row['gate_mean']:.6f}",
                "scale_mean": f"{row['scale_mean']:.6f}",
                "bond_mean_diff": f"{row['bond_mean_diff']:.6f}"
                if np.isfinite(row["bond_mean_diff"])
                else "",
            }
        )
    write_csv(
        SOURCE_DIR / "FigureS3_mayer_vs_attention_source_data.csv",
        ["system", "mayer_bond_order", "gate_mean", "scale_mean", "bond_mean_diff"],
        source_rows,
    )

    fig = plt.figure(figsize=(183 / 25.4, 72 / 25.4))
    gs = GridSpec(2, 3, figure=fig, hspace=0.52, wspace=0.36)
    letters = iter("abcdef")
    summary_rows = []

    for row_idx, system in enumerate(SYSTEMS):
        system_rows = [r for r in rows if r["system"] == system]
        for col_idx, (metric, title) in enumerate(METRICS):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            add_panel_label(ax, next(letters))
            if metric == "hist":
                plot_hist(ax, system_rows, system)
            else:
                summary_rows.append(plot_scatter(ax, system_rows, system, metric))
            if col_idx == 0:
                ax.set_title(SYSTEM_TITLES[system], loc="left", fontsize=7.8, pad=8)
            else:
                ax.set_title(title, fontsize=7.8, pad=8)

    fig.subplots_adjust(left=0.065, right=0.995, top=0.95, bottom=0.105)
    save_figure(fig)
    write_csv(
        SOURCE_DIR / "FigureS3_mayer_vs_attention_correlations.csv",
        ["system", "metric", "spearman_rho", "pearson_r", "n_unique_edges"],
        summary_rows,
    )


if __name__ == "__main__":
    main()
