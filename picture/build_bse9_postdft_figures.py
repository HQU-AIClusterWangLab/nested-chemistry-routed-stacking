import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "picture" / "publication_figures" / "bse9_postdft"
SOURCE_DIR = OUT_DIR / "source_data"
PHASE6_DIR = ROOT / "phase_6" / "final_nested_router"
EDGE_TABLE = (
    PHASE6_DIR
    / "04_computational_chemistry_plan"
    / "tables"
    / "bond_order_attention_edge_template.csv"
)
REP_TABLE = (
    PHASE6_DIR
    / "04_computational_chemistry_plan"
    / "tables"
    / "representative_structure_tasks.csv"
)
ORCA_DIR = (
    PHASE6_DIR
    / "05_post_dft_analysis_templates"
    / "generated_inputs"
    / "orca"
)

REP_SAMPLES = ["BSe9-167_sample", "BSe9-84_sample", "BSe9-256_sample"]
REP_ROLES = {
    "BSe9-167_sample": "low-energy reference",
    "BSe9-84_sample": "high-risk low-energy",
    "BSe9-256_sample": "relaxation hit",
}
PANEL_ORDER = ["BSe9-167_sample", "BSe9-84_sample", "BSe9-256_sample"]

PALETTE = {
    "neutral_dark": "#2B2B2B",
    "neutral_mid": "#7A7A7A",
    "neutral_light": "#D5D5D5",
    "bse": "#B64342",
    "se": "#0F4D92",
    "gate": "#33B5A5",
    "scale": "#7C6CCF",
    "bond_diff": "#E28E2C",
    "panel_bg": "#FAFAFA",
}

METRICS = [
    ("gate_mean", "Gate mean", PALETTE["gate"]),
    ("scale_mean", "Scale mean", PALETTE["scale"]),
    ("distance", "Distance (A)", PALETTE["bond_diff"]),
]

SYSTEM_LABELS = {"BSe9": "BSe$_9$ (policy branch)", "LaSi9": "LaSi$_9$ (gate branch)"}
PAIR_MARKERS = {"B-Se": "o", "Se-Se": "s", "La-Si": "o", "Si-Si": "s"}
PAIR_COLORS = {"B-Se": PALETTE["bse"], "Se-Se": PALETTE["se"], "La-Si": PALETTE["bse"], "Si-Si": PALETTE["se"]}


def apply_publication_style():
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.size"] = 7
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["figure.facecolor"] = "white"


def mm_to_inch(value_mm):
    return value_mm / 25.4


def ensure_dirs():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)


def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def add_panel_label(ax, label, x=-0.08, y=1.03, color="black"):
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=color,
    )


def save_pub(fig, stem):
    base = OUT_DIR / stem
    fig.savefig(f"{base}.svg", bbox_inches="tight")
    fig.savefig(f"{base}.pdf", bbox_inches="tight")
    fig.savefig(f"{base}.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(f"{base}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_float(value):
    if value is None:
        return math.nan
    text = str(value).strip()
    if not text:
        return math.nan
    return float(text)


def classify_edge(row):
    zi = int(float(row["Z_i"]))
    zj = int(float(row["Z_j"]))
    pair = tuple(sorted((zi, zj)))
    system = row.get("system_id", "")
    if system == "BSe9":
        if pair == (5, 34):
            return "B-Se"
        if pair == (34, 34):
            return "Se-Se"
    if system == "LaSi9":
        if pair == (14, 57):
            return "La-Si"
        if pair == (14, 14):
            return "Si-Si"
    return "other"


def load_bse9_edges():
    rows = read_csv(EDGE_TABLE)
    filtered = []
    for row in rows:
        if row["system_id"] != "BSe9":
            continue
        if row["sample_id"] not in REP_SAMPLES:
            continue
        if not row["mayer_bond_order"]:
            continue
        pair = classify_edge(row)
        if pair == "other":
            continue
        filtered.append(
            {
                "sample_id": row["sample_id"],
                "pair_type": pair,
                "distance": parse_float(row["distance"]),
                "gate_mean": parse_float(row["gate_mean"]),
                "scale_mean": parse_float(row["scale_mean"]),
                "bond_mean_diff": parse_float(row["bond_mean_diff"]),
                "mayer_bond_order": parse_float(row["mayer_bond_order"]),
            }
        )
    return filtered


def deduplicate_edges(rows):
    dedup = {}
    for row in rows:
        key = (
            row["sample_id"],
            row["pair_type"],
            tuple(sorted((row["distance"], row["mayer_bond_order"], row["gate_mean"], row["scale_mean"], row["bond_mean_diff"]))),
        )
        if key not in dedup:
            dedup[key] = row
    return list(dedup.values())


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
    order = sorted(range(len(values)), key=lambda i: values[i])
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


def export_scatter_source(rows):
    out_rows = []
    for row in rows:
        out_rows.append(
            {
                "sample_id": row["sample_id"],
                "role": REP_ROLES[row["sample_id"]],
                "pair_type": row["pair_type"],
                "distance_A": f"{row['distance']:.6f}",
                "gate_mean": f"{row['gate_mean']:.6f}",
                "scale_mean": f"{row['scale_mean']:.6f}",
                "mayer_bond_order": f"{row['mayer_bond_order']:.4f}",
                "bonded_contact": "yes" if row["mayer_bond_order"] >= 0.1 else "no",
            }
        )
    write_csv(
        SOURCE_DIR / "FigureS3_mayer_vs_attention_source_data.csv",
        list(out_rows[0].keys()),
        out_rows,
    )


def plot_mayer_attention():
    rows = deduplicate_edges(load_bse9_edges())
    export_scatter_source(rows)

    fig = plt.figure(figsize=(mm_to_inch(183), mm_to_inch(66)))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.0, 0.14], hspace=0.08, wspace=0.16)

    pair_markers = {"B-Se": "o", "Se-Se": "s", "La-Si": "o", "Si-Si": "s"}
    pair_colors = {"B-Se": PALETTE["bse"], "Se-Se": PALETTE["se"], "La-Si": PALETTE["bse"], "Si-Si": PALETTE["se"]}
    summary_rows = []

    for idx, (metric, title, metric_color) in enumerate(METRICS):
        ax = fig.add_subplot(gs[0, idx])
        xs = [row[metric] for row in rows]
        ys = [row["mayer_bond_order"] for row in rows]
        for pair_type in ["B-Se", "Se-Se"]:
            subset = [row for row in rows if row["pair_type"] == pair_type]
            bonded = [row for row in subset if row["mayer_bond_order"] >= 0.1]
            weak = [row for row in subset if row["mayer_bond_order"] < 0.1]
            if weak:
                ax.scatter([row[metric] for row in weak], [row["mayer_bond_order"] for row in weak], s=32, marker=pair_markers[pair_type], facecolors="none", edgecolors=pair_colors[pair_type], alpha=0.62, linewidths=1.0)
            if bonded:
                ax.scatter([row[metric] for row in bonded], [row["mayer_bond_order"] for row in bonded], s=36, marker=pair_markers[pair_type], c=pair_colors[pair_type], alpha=0.84, edgecolors="white", linewidths=0.35)
        coeff = np.polyfit(xs, ys, 1)
        xline = np.linspace(min(xs), max(xs), 200)
        yline = coeff[0] * xline + coeff[1]
        ax.plot(xline, yline, color=metric_color, linewidth=1.5)
        r_val = spearman(xs, ys)
        ax.text(0.97, 0.97, f"r = {r_val:.2f}\nn = {len(xs)}", transform=ax.transAxes, ha="right", va="top", fontsize=8.0, color=PALETTE["neutral_dark"], bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.6, "alpha": 0.86})
        ax.set_title(title, fontsize=8.6, pad=4)
        ax.set_xlabel(title, fontsize=8.1)
        if idx == 0:
            ax.set_ylabel("Mayer bond order", fontsize=8.1)
        ax.grid(alpha=0.14, linewidth=0.45)
        ax.tick_params(labelsize=7.8, width=0.7, length=2.5)
        summary_rows.append({"metric": metric, "spearman_r": f"{r_val:.4f}", "n_edges": str(len(xs))})

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=pair_colors["B-Se"], markeredgecolor="white", markeredgewidth=0.45, markersize=5.0, label="B-Se, Mayer >= 0.1"),
        plt.Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="none", markeredgecolor=pair_colors["B-Se"], markeredgewidth=1.0, markersize=5.0, label="B-Se, Mayer < 0.1"),
        plt.Line2D([0], [0], marker="s", linestyle="None", markerfacecolor=pair_colors["Se-Se"], markeredgecolor="white", markeredgewidth=0.45, markersize=5.0, label="Se-Se, Mayer >= 0.1"),
        plt.Line2D([0], [0], marker="s", linestyle="None", markerfacecolor="none", markeredgecolor=pair_colors["Se-Se"], markeredgewidth=1.0, markersize=5.0, label="Se-Se, Mayer < 0.1"),
    ]
    legend_ax = fig.add_subplot(gs[1, :])
    legend_ax.axis("off")
    legend_ax.legend(handles=handles, loc="center", ncol=4, fontsize=7.2, frameon=False, handletextpad=0.45, columnspacing=1.1)
    fig.subplots_adjust(left=0.055, right=0.995, top=0.92, bottom=0.12)
    save_pub(fig, "FigureS3_mayer_vs_attention")
    write_csv(SOURCE_DIR / "FigureS3_mayer_vs_attention_correlations.csv", ["metric", "spearman_r", "n_edges"], summary_rows)


def read_cube(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        lines = handle.readlines()

    atom_line = lines[2].split()
    natoms = abs(int(float(atom_line[0])))
    origin = np.array([float(atom_line[1]), float(atom_line[2]), float(atom_line[3])], dtype=float)

    grid = []
    axes = []
    for idx in range(3, 6):
        parts = lines[idx].split()
        count = int(float(parts[0]))
        vec = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=float)
        grid.append(count)
        axes.append(vec)

    atom_lines = lines[6 : 6 + natoms]
    atoms = []
    for line in atom_lines:
        parts = line.split()
        atoms.append(
            {
                "Z": int(float(parts[0])),
                "x": float(parts[2]),
                "y": float(parts[3]),
                "z": float(parts[4]),
            }
        )

    values = []
    for line in lines[6 + natoms :]:
        for token in line.split():
            values.append(float(token.replace("D", "E")))
    data = np.array(values, dtype=float).reshape(tuple(grid), order="C")
    return {
        "origin": origin,
        "axes": axes,
        "grid": grid,
        "atoms": atoms,
        "data": data,
    }


def cube_extent(cube_info):
    origin = cube_info["origin"]
    nx, ny = cube_info["grid"][0], cube_info["grid"][1]
    step_x = float(np.linalg.norm(cube_info["axes"][0]))
    step_y = float(np.linalg.norm(cube_info["axes"][1]))
    x0 = origin[0]
    y0 = origin[1]
    return [x0, x0 + step_x * (nx - 1), y0, y0 + step_y * (ny - 1)]


def project_cube_max_abs(cube_info, mode="cdd"):
    data = cube_info["data"]
    if mode == "cdd":
        pos = np.max(np.clip(data, 0.0, None), axis=2)
        neg = np.max(np.clip(-data, 0.0, None), axis=2)
        return pos, neg
    return np.max(data, axis=2), None


def atom_projection(cube_info):
    atoms = cube_info["atoms"]
    coords = np.array([[atom["x"], atom["y"]] for atom in atoms], dtype=float)
    z_vals = [atom["Z"] for atom in atoms]
    return coords, z_vals


def draw_structure_overlay(ax, coords, z_vals):
    radii = {5: 28, 34: 24}
    colors = {5: "#D97706", 34: "#6B7280"}
    bond_threshold = 2.65
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist <= bond_threshold:
                ax.plot(
                    [coords[i, 0], coords[j, 0]],
                    [coords[i, 1], coords[j, 1]],
                    color="white",
                    linewidth=0.55,
                    alpha=0.72,
                    zorder=3,
                )
    for idx, (xy, z_val) in enumerate(zip(coords, z_vals)):
        ax.scatter(
            xy[0],
            xy[1],
            s=radii.get(z_val, 18),
            c=colors.get(z_val, "#999999"),
            edgecolors="black",
            linewidths=0.35,
            zorder=4,
        )


def export_mechanism_source(summary_rows):
    write_csv(
        SOURCE_DIR / "bse9_cdd_elf_summary.csv",
        list(summary_rows[0].keys()),
        summary_rows,
    )


def crop_limits(coords, pad=1.25):
    xmin = float(np.min(coords[:, 0]) - pad)
    xmax = float(np.max(coords[:, 0]) + pad)
    ymin = float(np.min(coords[:, 1]) - pad)
    ymax = float(np.max(coords[:, 1]) + pad)
    return xmin, xmax, ymin, ymax


def plot_cdd_elf():
    fig = plt.figure(figsize=(mm_to_inch(178), mm_to_inch(174)))
    gs = GridSpec(
        4,
        2,
        figure=fig,
        height_ratios=[1.0, 1.0, 1.0, 0.06],
        width_ratios=[1.0, 1.0],
        hspace=0.16,
        wspace=0.12,
    )

    summary_rows = []
    cdd_axes = []
    elf_axes = []
    cdd_last = None
    elf_last = None

    for row_idx, sample_id in enumerate(PANEL_ORDER):
        cdd_cube = read_cube(ORCA_DIR / f"{sample_id}_CDD.cube")
        elf_cube = read_cube(ORCA_DIR / f"{sample_id}_orca_mayer_density_elf_elf.cube")

        cdd_pos, cdd_neg = project_cube_max_abs(cdd_cube, mode="cdd")
        elf_proj, _ = project_cube_max_abs(elf_cube, mode="elf")
        coords, z_vals = atom_projection(elf_cube)
        xmin, xmax, ymin, ymax = crop_limits(coords, pad=1.35)

        cdd_ax = fig.add_subplot(gs[row_idx, 0])
        cdd_last = cdd_ax.imshow(
            cdd_pos - cdd_neg,
            cmap="coolwarm",
            origin="lower",
            vmin=-0.36,
            vmax=0.36,
            interpolation="bilinear",
            extent=cube_extent(cdd_cube),
        )
        draw_structure_overlay(cdd_ax, coords, z_vals)
        cdd_ax.set_xlim(xmin, xmax)
        cdd_ax.set_ylim(ymin, ymax)
        cdd_ax.set_aspect("equal")
        cdd_ax.set_xticks([])
        cdd_ax.set_yticks([])
        cdd_ax.set_title(REP_ROLES[sample_id], fontsize=8.5, pad=5)
        cdd_ax.text(
            0.02,
            0.03,
            sample_id.replace("_sample", ""),
            transform=cdd_ax.transAxes,
            fontsize=7.0,
            ha="left",
            va="bottom",
            color="white",
            bbox={"facecolor": (0, 0, 0, 0.42), "edgecolor": "none", "pad": 1.2},
        )
        cdd_axes.append(cdd_ax)

        elf_ax = fig.add_subplot(gs[row_idx, 1])
        elf_last = elf_ax.imshow(
            elf_proj,
            cmap="magma",
            origin="lower",
            vmin=0.0,
            vmax=0.9,
            interpolation="bilinear",
            extent=cube_extent(elf_cube),
        )
        draw_structure_overlay(elf_ax, coords, z_vals)
        elf_ax.set_xlim(xmin, xmax)
        elf_ax.set_ylim(ymin, ymax)
        elf_ax.set_aspect("equal")
        elf_ax.set_xticks([])
        elf_ax.set_yticks([])
        elf_ax.set_title(REP_ROLES[sample_id], fontsize=8.5, pad=5)
        elf_ax.text(
            0.02,
            0.03,
            sample_id.replace("_sample", ""),
            transform=elf_ax.transAxes,
            fontsize=7.0,
            ha="left",
            va="bottom",
            color="white",
            bbox={"facecolor": (0, 0, 0, 0.42), "edgecolor": "none", "pad": 1.2},
        )
        elf_axes.append(elf_ax)

        cdd_vals = cdd_cube["data"].ravel()
        elf_vals = elf_cube["data"].ravel()
        summary_rows.append(
            {
                "sample_id": sample_id,
                "role": REP_ROLES[sample_id],
                "cdd_min_au": f"{float(np.min(cdd_vals)):.6f}",
                "cdd_max_au": f"{float(np.max(cdd_vals)):.6f}",
                "elf_max_projection": f"{float(np.max(elf_proj)):.6f}",
            }
        )

    cdd_cax = fig.add_subplot(gs[3, 0])
    elf_cax = fig.add_subplot(gs[3, 1])
    cbar1 = fig.colorbar(cdd_last, cax=cdd_cax, orientation="horizontal")
    cbar1.set_label("CDD projection (a.u.)", fontsize=7.5, labelpad=2)
    cbar1.ax.tick_params(labelsize=7, width=0.6, length=2)
    cbar2 = fig.colorbar(elf_last, cax=elf_cax, orientation="horizontal")
    cbar2.set_label("ELF projection", fontsize=7.5, labelpad=2)
    cbar2.ax.tick_params(labelsize=7, width=0.6, length=2)

    cdd_axes[0].text(
        0.02,
        0.97,
        "CDD",
        transform=cdd_axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=8.6,
        fontweight="bold",
        color="white",
        bbox={"facecolor": (0, 0, 0, 0.42), "edgecolor": "none", "pad": 1.2},
    )
    elf_axes[0].text(
        0.02,
        0.97,
        "ELF",
        transform=elf_axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=8.6,
        fontweight="bold",
        color="white",
        bbox={"facecolor": (0, 0, 0, 0.42), "edgecolor": "none", "pad": 1.2},
    )

    fig.subplots_adjust(left=0.05, right=0.985, top=0.985, bottom=0.065)
    save_pub(fig, "Figure_BSe9_cdd_elf")
    export_mechanism_source(summary_rows)


def render_png_montage():
    panels = [
        OUT_DIR / "Figure_BSe9_mayer_attention.png",
        OUT_DIR / "Figure_BSe9_cdd_elf.png",
    ]
    images = [Image.open(path).convert("RGB") for path in panels]
    width = max(image.width for image in images)
    pad = 36
    total_height = sum(image.height for image in images) + pad * (len(images) - 1)
    canvas = Image.new("RGB", (width, total_height), "white")
    y = 0
    for image in images:
        x = (width - image.width) // 2
        canvas.paste(image, (x, y))
        y += image.height + pad
    canvas.save(OUT_DIR / "Figure_BSe9_overview.png", dpi=(300, 300))


def main():
    apply_publication_style()
    ensure_dirs()
    plot_mayer_attention()
    plot_cdd_elf()
    render_png_montage()


if __name__ == "__main__":
    main()
