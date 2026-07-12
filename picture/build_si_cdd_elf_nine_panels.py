import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "picture" / "publication_figures"
SOURCE_DIR = FIG_DIR / "source_data"
PANEL_IMAGE_DIR = FIG_DIR / "fig4_2d_maps"
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

SYSTEM_ROWS = [
    (
        "BSe9",
        [
            ("BSe9-167_sample", ORCA_DIR / "BSe9-167_sample_CDD.cube", ORCA_DIR / "BSe9-167_sample_orca_mayer_density_elf_elf.cube"),
            ("BSe9-256_sample", ORCA_DIR / "BSe9-256_sample_CDD.cube", ORCA_DIR / "BSe9-256_sample_orca_mayer_density_elf_elf.cube"),
            ("BSe9-84_sample", ORCA_DIR / "BSe9-84_sample_CDD.cube", ORCA_DIR / "BSe9-84_sample_orca_mayer_density_elf_elf.cube"),
        ],
    ),
    (
        "LaSi9",
        [
            ("LaSi9-1472_sample", ORCA_DIR / "LaSi9" / "LaSi9-1472_sample_CDD.cube", ORCA_DIR / "LaSi9" / "LaSi9-1472_sample_orca_mayer_elf.cube"),
            ("LaSi9-2546_sample", ORCA_DIR / "LaSi9" / "LaSi9-2546_sample_CDD.cube", ORCA_DIR / "LaSi9" / "LaSi9-2546_sample_orca_mayer_elf.cube"),
            ("LaSi9-1792_sample", ORCA_DIR / "LaSi9" / "LaSi9-1792_sample_CDD.cube", ORCA_DIR / "LaSi9" / "LaSi9-1792_sample_orca_mayer_elf.cube"),
        ],
    ),
    (
        "LaCu12",
        [
            ("LaCu12-15_sample", ORCA_DIR / "LaCu12" / "LaCu12-15_sample_CDD.cube", ORCA_DIR / "LaCu12" / "LaCu12-15_sample_orca_mayer_elf.cube"),
            ("LaCu12-33_sample", ORCA_DIR / "LaCu12" / "LaCu12-33_sample_CDD.cube", ORCA_DIR / "LaCu12" / "LaCu12-33_sample_orca_mayer_elf.cube"),
            ("LaCu12-95_sample", ORCA_DIR / "LaCu12" / "LaCu12-95_sample_CDD.cube", ORCA_DIR / "LaCu12" / "LaCu12-95_sample_orca_mayer_elf.cube"),
        ],
    ),
]

ATOM_STYLE = {
    5: {"size": 28, "color": "#D97706"},
    14: {"size": 24, "color": "#6D8FB5"},
    29: {"size": 24, "color": "#1F8A9E"},
    34: {"size": 24, "color": "#6B7280"},
    57: {"size": 30, "color": "#D4A017"},
}
SYSTEM_BOND_THRESHOLD = {"BSe9": 2.65, "LaSi9": 3.15, "LaCu12": 3.25}
PANEL_IMAGE_BY_SAMPLE = {
    "BSe9-167_sample": PANEL_IMAGE_DIR / "Fig4_BSe9_167_cdd_elf.png",
    "BSe9-256_sample": PANEL_IMAGE_DIR / "Fig4_BSe9_256_cdd_elf.png",
    "BSe9-84_sample": PANEL_IMAGE_DIR / "Fig4_BSe9_84_cdd_elf.png",
    "LaSi9-1472_sample": PANEL_IMAGE_DIR / "Fig4_LaSi9_1472_cdd_elf.png",
    "LaSi9-2546_sample": PANEL_IMAGE_DIR / "Fig4_LaSi9_2546_cdd_elf.png",
    "LaSi9-1792_sample": PANEL_IMAGE_DIR / "Fig4_LaSi9_1792_cdd_elf.png",
    "LaCu12-15_sample": PANEL_IMAGE_DIR / "Fig4_LaCu12_15_cdd_elf.png",
    "LaCu12-33_sample": PANEL_IMAGE_DIR / "Fig4_LaCu12_33_cdd_elf.png",
    "LaCu12-95_sample": PANEL_IMAGE_DIR / "Fig4_LaCu12_95_cdd_elf.png",
}
CDD_VMIN = -0.36
CDD_VMAX = 0.36
ELF_VMIN = 0.0
ELF_VMAX = 0.9
ELF_CONTOUR_LEVELS = [0.75]


def apply_style():
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.size"] = 7
    plt.rcParams["axes.linewidth"] = 0.7


def mm_to_inch(value_mm):
    return value_mm / 25.4


def ensure_dirs():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)


def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, fieldnames, rows):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_roles():
    role_map = {}
    for row in read_csv(TASK_TABLE):
        role_map[row["sample_id"]] = row["role"]
    return role_map


def read_cube(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        lines = handle.readlines()

    natoms = abs(int(float(lines[2].split()[0])))
    origin = np.array([float(x) for x in lines[2].split()[1:4]], dtype=float)

    grid = []
    axes = []
    for idx in range(3, 6):
        parts = lines[idx].split()
        grid.append(int(float(parts[0])))
        axes.append(np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=float))

    atoms = []
    for line in lines[6 : 6 + natoms]:
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
    return {"origin": origin, "grid": grid, "axes": axes, "atoms": atoms, "data": data}


def cube_extent(cube_info):
    origin = cube_info["origin"]
    nx, ny = cube_info["grid"][0], cube_info["grid"][1]
    step_x = float(np.linalg.norm(cube_info["axes"][0]))
    step_y = float(np.linalg.norm(cube_info["axes"][1]))
    return [origin[0], origin[0] + step_x * (nx - 1), origin[1], origin[1] + step_y * (ny - 1)]


def project_cube(cube_info, mode):
    data = cube_info["data"]
    if mode == "CDD":
        pos = np.max(np.clip(data, 0.0, None), axis=2)
        neg = np.max(np.clip(-data, 0.0, None), axis=2)
        return pos - neg
    return np.max(data, axis=2)


def atom_projection(cube_info):
    coords = np.array([[a["x"], a["y"]] for a in cube_info["atoms"]], dtype=float)
    z_vals = [a["Z"] for a in cube_info["atoms"]]
    return coords, z_vals


def crop_limits(coords, pad=1.25):
    xmin = float(np.min(coords[:, 0]) - pad)
    xmax = float(np.max(coords[:, 0]) + pad)
    ymin = float(np.min(coords[:, 1]) - pad)
    ymax = float(np.max(coords[:, 1]) + pad)
    return xmin, xmax, ymin, ymax


def draw_structure_overlay(ax, coords, z_vals, system_name):
    bond_threshold = SYSTEM_BOND_THRESHOLD[system_name]
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist <= bond_threshold:
                ax.plot(
                    [coords[i, 0], coords[j, 0]],
                    [coords[i, 1], coords[j, 1]],
                    color="white",
                    linewidth=0.45,
                    alpha=0.75,
                    zorder=3,
                )
    for xy, z_val in zip(coords, z_vals):
        style = ATOM_STYLE.get(z_val, {"size": 20, "color": "#888888"})
        ax.scatter(
            xy[0],
            xy[1],
            s=style["size"],
            c=style["color"],
            edgecolors="white" if z_val == 29 else "#1F2937",
            linewidths=0.35,
            zorder=4,
        )


def add_panel_label(ax, label):
    ax.text(-0.08, 1.03, label, transform=ax.transAxes, fontsize=8, fontweight="bold", ha="left", va="bottom")


def save_pub(fig, stem):
    base = FIG_DIR / stem
    fig.savefig(f"{base}.svg", bbox_inches="tight")
    fig.savefig(f"{base}.pdf", bbox_inches="tight")
    fig.savefig(f"{base}.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(f"{base}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_pub_fixed(fig, stem):
    base = FIG_DIR / stem
    fig.savefig(f"{base}.svg")
    fig.savefig(f"{base}.pdf")
    fig.savefig(f"{base}.tiff", dpi=600)
    fig.savefig(f"{base}.png", dpi=300)
    plt.close(fig)


def load_cdd_half_from_main_style_panel(sample_id):
    image_path = PANEL_IMAGE_BY_SAMPLE[sample_id]
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    return np.asarray(image.crop((0, 0, width // 2, height)))


def load_elf_half_from_main_style_panel(sample_id):
    image_path = PANEL_IMAGE_BY_SAMPLE[sample_id]
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    return np.asarray(image.crop((width // 2, 0, width, height)))


def build_cdd_from_main_style_panels(summary_rows):
    fig = plt.figure(figsize=(mm_to_inch(183), mm_to_inch(180)))
    gs = GridSpec(
        4,
        3,
        figure=fig,
        height_ratios=[1.0, 1.0, 1.0, 0.10],
        hspace=0.34,
        wspace=0.12,
    )

    for row_idx, (system_name, items) in enumerate(SYSTEM_ROWS):
        for col_idx, (sample_id, _cdd_path, _elf_path) in enumerate(items):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            ax.imshow(load_cdd_half_from_main_style_panel(sample_id))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)
            ax.set_title(sample_id.replace("_sample", ""), fontsize=7.2, pad=2)
            if col_idx == 0:
                ax.text(
                    -0.11,
                    0.5,
                    system_name,
                    transform=ax.transAxes,
                    rotation=90,
                    fontsize=8.0,
                    fontweight="bold",
                    ha="center",
                    va="center",
                )
            if row_idx == 0 and col_idx == 0:
                add_panel_label(ax, "a")

    cbar_host = fig.add_subplot(gs[3, :])
    cbar_host.axis("off")
    cax = cbar_host.inset_axes([0.18, 0.36, 0.64, 0.22])
    cdd_map = ScalarMappable(norm=Normalize(vmin=CDD_VMIN, vmax=CDD_VMAX), cmap="coolwarm")
    cbar = fig.colorbar(cdd_map, cax=cax, orientation="horizontal")
    cbar.set_label("CDD projection (a.u.)", fontsize=7.0, labelpad=2)
    cbar.set_ticks([CDD_VMIN, 0.0, CDD_VMAX])
    cbar.ax.tick_params(labelsize=6.5, length=2.0, pad=1.5)

    fig.subplots_adjust(left=0.06, right=0.985, top=0.985, bottom=0.055)
    save_pub_fixed(fig, "FigureS4_cdd_nine_panel")

    write_csv(
        SOURCE_DIR / "FigureS4_cdd_nine_panel_source_data.csv",
        ["system", "sample_id", "role", "cdd_min_au", "cdd_max_au", "elf_max_projection"],
        summary_rows,
    )


def collect_summary_rows():
    role_map = load_roles()
    summary_rows = []

    for system_name, items in SYSTEM_ROWS:
        for sample_id, cdd_path, elf_path in items:
            cdd = read_cube(cdd_path)
            elf = read_cube(elf_path)
            summary_rows.append(
                {
                    "system": system_name,
                    "sample_id": sample_id,
                    "role": role_map.get(sample_id, ""),
                    "cdd_min_au": f"{float(np.min(cdd['data'])):.6f}",
                    "cdd_max_au": f"{float(np.max(cdd['data'])):.6f}",
                    "elf_max_projection": f"{float(project_cube(elf, 'ELF').max()):.6f}",
                }
            )
    return summary_rows


def build_s4():
    apply_style()
    ensure_dirs()
    build_cdd_from_main_style_panels(collect_summary_rows())


def build_elf_from_main_style(summary_rows):
    fig = plt.figure(figsize=(mm_to_inch(183), mm_to_inch(188)))
    gs = GridSpec(
        4,
        3,
        figure=fig,
        height_ratios=[1.0, 1.0, 1.0, 0.09],
        hspace=0.06,
        wspace=0.20,
    )

    for row_idx, (system_name, items) in enumerate(SYSTEM_ROWS):
        for col_idx, (sample_id, cdd_path, elf_path) in enumerate(items):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            ax.imshow(load_elf_half_from_main_style_panel(sample_id))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)

    cbar_host = fig.add_subplot(gs[3, :])
    cbar_host.axis("off")
    cax = cbar_host.inset_axes([0.18, 0.34, 0.64, 0.27])
    elf_map = ScalarMappable(norm=Normalize(vmin=ELF_VMIN, vmax=ELF_VMAX), cmap="hot")
    cbar = fig.colorbar(elf_map, cax=cax, orientation="horizontal")
    cbar.set_label("ELF projection", fontsize=6.8, labelpad=2)
    cbar.set_ticks([ELF_VMIN, 0.45, ELF_VMAX])
    cbar.ax.tick_params(labelsize=6.2, width=0.55, length=2.0, pad=1.5)

    fig.subplots_adjust(left=0.025, right=0.985, top=0.975, bottom=0.055)
    save_pub_fixed(fig, "FigureS5_elf_nine_panel")

    write_csv(
        SOURCE_DIR / "FigureS5_elf_nine_panel_source_data.csv",
        ["system", "sample_id", "role", "cdd_min_au", "cdd_max_au", "elf_max_projection"],
        summary_rows,
    )


def build():
    apply_style()
    ensure_dirs()
    summary_rows = collect_summary_rows()
    build_cdd_from_main_style_panels(summary_rows)
    build_elf_from_main_style(summary_rows)


if __name__ == "__main__":
    build()
