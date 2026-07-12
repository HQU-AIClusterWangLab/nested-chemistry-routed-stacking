import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "picture" / "publication_figures"
MAP_DIR = OUT_DIR / "fig4_2d_maps"
SOURCE_DIR = OUT_DIR / "source_data"
ORCA_DIR = (
    ROOT
    / "phase_6"
    / "final_nested_router"
    / "05_post_dft_analysis_templates"
    / "generated_inputs"
    / "orca"
)
TASK_TABLE = (
    ROOT
    / "phase_6"
    / "final_nested_router"
    / "04_computational_chemistry_plan"
    / "tables"
    / "representative_structure_tasks.csv"
)

PANELS = [
    {
        "sample_id": "BSe9-167_sample",
        "label": "BSe9-167",
        "title": r"BSe$_{9}^{-}$-167",
        "display_role": "low-energy reference",
        "image": MAP_DIR / "Fig4_BSe9_167_cdd_elf.png",
        "cdd_cube": ORCA_DIR / "BSe9-167_sample_CDD.cube",
        "elf_cube": ORCA_DIR / "BSe9-167_sample_orca_mayer_density_elf_elf.cube",
    },
    {
        "sample_id": "LaSi9-1472_sample",
        "label": "LaSi9-1472",
        "title": r"LaSi$_{9}^{-}$-1472",
        "display_role": "low-energy hit",
        "image": MAP_DIR / "Fig4_LaSi9_1472_cdd_elf.png",
        "cdd_cube": ORCA_DIR / "LaSi9" / "LaSi9-1472_sample_CDD.cube",
        "elf_cube": ORCA_DIR / "LaSi9" / "LaSi9-1472_sample_orca_mayer_elf.cube",
    },
    {
        "sample_id": "LaCu12-33_sample",
        "label": "LaCu12-33",
        "title": r"LaCu$_{12}^{-}$-33",
        "display_role": "high-risk low-energy",
        "image": MAP_DIR / "Fig4_LaCu12_33_cdd_elf.png",
        "cdd_cube": ORCA_DIR / "LaCu12" / "LaCu12-33_sample_CDD.cube",
        "elf_cube": ORCA_DIR / "LaCu12" / "LaCu12-33_sample_orca_mayer_elf.cube",
    },
]

SYSTEM_COLORS = {
    "BSe9-167_sample": "#D45D5D",
    "LaSi9-1472_sample": "#2FA6A0",
    "LaCu12-33_sample": "#8A74D6",
}

CDD_VMIN = -0.36
CDD_VMAX = 0.36
ELF_VMIN = 0.0
ELF_VMAX = 0.9


def mm_to_inch(mm_value):
    return mm_value / 25.4


def ensure_dirs():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)


def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, fieldnames, rows):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_role_map():
    role_map = {}
    for row in read_csv(TASK_TABLE):
        sample_id = row["sample_id"]
        role_map[sample_id] = row["role"]
    return role_map


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
        grid.append(int(float(parts[0])))
        axes.append(np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=float))

    values = []
    for line in lines[6 + natoms :]:
        for token in line.split():
            values.append(float(token.replace("D", "E")))
    data = np.array(values, dtype=float).reshape(tuple(grid), order="C")
    return {"origin": origin, "grid": grid, "axes": axes, "data": data}


def elf_projection_max(cube_info):
    return float(np.max(cube_info["data"], axis=2).max())


def summarize_panel(panel):
    cdd = read_cube(panel["cdd_cube"])
    elf = read_cube(panel["elf_cube"])
    return {
        "sample_id": panel["sample_id"],
        "label": panel["label"],
        "title": panel["title"],
        "role": panel["display_role"],
        "cdd_min_au": float(np.min(cdd["data"])),
        "cdd_max_au": float(np.max(cdd["data"])),
        "elf_max_projection": elf_projection_max(elf),
    }


def load_image(path):
    return np.asarray(Image.open(path).convert("RGB"))


def add_panel_label(ax, label, x=-0.02, y=1.02):
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=8.0,
        fontweight="bold",
        ha="left",
        va="bottom",
        color="black",
    )


def save_pub(fig, stem):
    base = OUT_DIR / stem
    fig.savefig(f"{base}.svg", bbox_inches="tight")
    fig.savefig(f"{base}.pdf", bbox_inches="tight")
    fig.savefig(f"{base}.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(f"{base}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def build():
    ensure_dirs()
    summary_rows = [summarize_panel(panel) for panel in PANELS]

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.size"] = 7

    fig = plt.figure(figsize=(mm_to_inch(183), mm_to_inch(188)))
    gs = GridSpec(
        4,
        1,
        figure=fig,
        height_ratios=[1.0, 1.0, 1.0, 0.14],
        hspace=0.18,
    )

    for row_idx, panel in enumerate(PANELS):
        ax = fig.add_subplot(gs[row_idx, 0])
        ax.imshow(load_image(panel["image"]))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
        ax.set_title(panel["title"], fontsize=8.8, pad=6)
        add_panel_label(ax, chr(ord("a") + row_idx), x=-0.025, y=1.01)
        row = summary_rows[row_idx]
        ax.text(
            0.0,
            1.01,
            row["role"],
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=6.9,
            color="#444444",
        )
        ax.text(
            0.995,
            1.01,
            f"CDD min {row['cdd_min_au']:.3f} a.u.   ELF max {row['elf_max_projection']:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=6.6,
            color="#444444",
        )

    cbar_host = fig.add_subplot(gs[3, 0])
    cbar_host.axis("off")
    cdd_cax = cbar_host.inset_axes([0.10, 0.55, 0.34, 0.20])
    elf_cax = cbar_host.inset_axes([0.56, 0.55, 0.34, 0.20])
    cdd_map = ScalarMappable(norm=Normalize(vmin=CDD_VMIN, vmax=CDD_VMAX), cmap="coolwarm")
    elf_map = ScalarMappable(norm=Normalize(vmin=ELF_VMIN, vmax=ELF_VMAX), cmap="magma")
    cdd_cbar = fig.colorbar(cdd_map, cax=cdd_cax, orientation="horizontal")
    elf_cbar = fig.colorbar(elf_map, cax=elf_cax, orientation="horizontal")
    cdd_cbar.set_label("CDD projection (a.u.)", fontsize=6.8, labelpad=2)
    elf_cbar.set_label("ELF projection", fontsize=6.8, labelpad=2)
    cdd_cbar.set_ticks([CDD_VMIN, 0.0, CDD_VMAX])
    elf_cbar.set_ticks([ELF_VMIN, 0.45, ELF_VMAX])
    cdd_cbar.ax.tick_params(labelsize=6.2, length=2.0, pad=1.5)
    elf_cbar.ax.tick_params(labelsize=6.2, length=2.0, pad=1.5)

    fig.subplots_adjust(left=0.04, right=0.995, top=0.988, bottom=0.055)
    save_pub(fig, "Figure4_postdft_chemistry_publication")

    write_csv(
        SOURCE_DIR / "Figure4_postdft_chemistry_source_data.csv",
        ["sample_id", "label", "role", "cdd_min_au", "cdd_max_au", "elf_max_projection", "cdd_colorbar_min_au", "cdd_colorbar_max_au", "elf_colorbar_min", "elf_colorbar_max"],
        [
            {
                "sample_id": row["sample_id"],
                "label": row["label"],
                "role": row["role"],
                "cdd_min_au": f"{row['cdd_min_au']:.6f}",
                "cdd_max_au": f"{row['cdd_max_au']:.6f}",
                "elf_max_projection": f"{row['elf_max_projection']:.6f}",
                "cdd_colorbar_min_au": f"{CDD_VMIN:.6f}",
                "cdd_colorbar_max_au": f"{CDD_VMAX:.6f}",
                "elf_colorbar_min": f"{ELF_VMIN:.6f}",
                "elf_colorbar_max": f"{ELF_VMAX:.6f}",
            }
            for row in summary_rows
        ],
    )


if __name__ == "__main__":
    build()
