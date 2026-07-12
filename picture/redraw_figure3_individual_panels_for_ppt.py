from __future__ import annotations

import csv
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import lines as mlines
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
PICTURE_DIR = ROOT / "picture"
FIG3_DATA = PICTURE_DIR / "figure3_origin_data"
DFT_DIR = ROOT / "phase_6" / "final_nested_router" / "02_dft_candidate_selection"
PUB_DIR = PICTURE_DIR / "publication_figures"
OUT_DIR = PUB_DIR / "Figure3_redrawn_individual_panels_20260709"
TIF_DIR = OUT_DIR / "tif_uncompressed"
PNG_DIR = OUT_DIR / "png_safe"
QA_DIR = OUT_DIR / "qa"
SPLIT_DIR = PUB_DIR / "PPT_split_panels_20260709"
SPLIT_TIF = SPLIT_DIR / "tif"
SPLIT_PNG = SPLIT_DIR / "png_preview"
BACKUP_DIR = SPLIT_DIR / "_backup_old_fig3_crops_20260709"

DPI = 600

PALETTE = {
    "AgB8": "#8D98C8",
    "AuB8": "#B9C4E6",
    "LaB8": "#B5B5B5",
    "LaSe8": "#7D7D7D",
    "LaCu12": "#E53935",
    "LaSi9": "#1E88E5",
    "BSe9": "#FF8C00",
    "dft": "#2E7D32",
    "replay": "#4A4A4A",
    "threshold": "#9E9E9E",
    "text": "#262626",
    "low_pred": "#DFF1FA",
    "high_risk": "#F7E4C8",
    "diversity": "#E6E6E6",
}

SYSTEM_LABELS = {
    "AgB8": r"AgB$_8$",
    "AuB8": r"AuB$_8$",
    "LaB8": r"LaB$_8$",
    "LaSe8": r"LaSe$_8$",
    "LaCu12": r"LaCu$_{12}$",
    "LaSi9": r"LaSi$_9$",
    "BSe9": r"BSe$_9$",
}

MAIN_SYSTEMS = ["LaCu12", "LaSi9", "BSe9"]
SI_SYSTEMS = ["AgB8", "AuB8", "LaB8", "LaSe8"]

OUTPUT_NAME_MAP = {
    "Fig3a_budget_recall": "Fig3_panel_a_budget_recall",
    "Fig3b_main_system_gap": "Fig3_panel_b_main_system_gap",
    "Fig3c_bse9_dft": "Fig3_panel_c_bse9_dft",
    "Fig3d_lasi9_dft": "Fig3_panel_d_lasi9_dft",
    "Fig3e_lacu12_dft": "Fig3_panel_e_lacu12_dft",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(value: str | float | int | None, default: float = np.nan) -> float:
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


def to_int(value: str | float | int | None, default: int = 0) -> int:
    number = to_float(value, np.nan)
    if not np.isfinite(number):
        return default
    return int(round(number))


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8.2,
            "axes.linewidth": 0.9,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "xtick.color": PALETTE["text"],
            "ytick.color": PALETTE["text"],
            "axes.labelcolor": PALETTE["text"],
            "text.color": PALETTE["text"],
            "legend.frameon": False,
        }
    )


def save_ppt_safe(fig: plt.Figure, stem: str) -> tuple[Path, Path]:
    png_path = PNG_DIR / f"{stem}.png"
    tif_path = TIF_DIR / f"{stem}.tif"
    svg_path = OUT_DIR / f"{stem}.svg"
    pdf_path = OUT_DIR / f"{stem}.pdf"

    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.18)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.18)
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight", pad_inches=0.18, facecolor="white")
    plt.close(fig)

    img = Image.open(png_path).convert("RGB")
    img.save(tif_path, dpi=(DPI, DPI))
    return tif_path, png_path


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.075,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
    )


def load_curve_table(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    rows = read_csv(path)
    ks = np.asarray([to_int(row["K"]) for row in rows], dtype=int)
    systems = [key for key in rows[0].keys() if key != "K"]
    curves = {system: np.asarray([to_float(row[system]) for row in rows], dtype=float) for system in systems}
    return ks, curves


def plot_budget_recall() -> tuple[Path, Path]:
    ks, curves = load_curve_table(FIG3_DATA / "figure3_budget_recall_all_systems_for_origin.csv")
    keep = ks <= 100
    fig, ax = plt.subplots(figsize=(7.6, 2.65))

    for system in SI_SYSTEMS + MAIN_SYSTEMS:
        is_main = system in MAIN_SYSTEMS
        ax.plot(
            ks[keep],
            curves[system][keep],
            color=PALETTE[system],
            linestyle="-" if is_main else "--",
            linewidth=2.3 if is_main else 1.45,
            alpha=0.98 if is_main else 0.82,
            label=SYSTEM_LABELS[system],
        )

    ax.set_xlim(1, 100)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Budget K")
    ax.set_ylabel("Recall within 0.10 eV")
    ax.set_title("Budget recall depends strongly on system chemistry", loc="left", fontsize=9.2, pad=8)
    ax.grid(alpha=0.24, linewidth=0.65)
    ax.legend(loc="upper left", fontsize=7.0, ncol=4, handlelength=2.2, columnspacing=0.9)
    ax.text(
        0.03,
        0.62,
        "Zero recall (all K):\nLaSi9, LaCu12, LaB8, LaSe8",
        transform=ax.transAxes,
        fontsize=7.2,
        ha="left",
        va="center",
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "#C8C8C8", "linewidth": 0.6},
    )
    add_panel_label(ax, "a")
    return save_ppt_safe(fig, "Fig3a_budget_recall")


def plot_main_system_gap() -> tuple[Path, Path]:
    ks, curves = load_curve_table(FIG3_DATA / "figure3_best_gap_main_systems_for_origin.csv")
    keep = ks <= 100
    rows = read_csv(FIG3_DATA / "figure3_budget_recall_summary.csv")
    summary = {row["system"]: row for row in rows}

    fig, ax = plt.subplots(figsize=(7.6, 2.55))
    for system in ["BSe9", "LaSi9", "LaCu12"]:
        ax.step(
            ks[keep],
            curves[system][keep],
            where="post",
            color=PALETTE[system],
            linewidth=2.45,
            label=SYSTEM_LABELS[system],
        )

    ax.axhline(0.10, color=PALETTE["threshold"], linewidth=1.0, linestyle="--")
    bse9_rank = to_int(summary["BSe9"]["budget_to_first_hit"], -1)
    if bse9_rank > 0:
        ax.scatter([bse9_rank], [0.0], s=34, color=PALETTE["BSe9"], zorder=4)
        ax.annotate(
            "BSe9 hit at K=73",
            xy=(bse9_rank, 0.0),
            xytext=(48, 0.19),
            fontsize=6.4,
            ha="left",
            color=PALETTE["BSe9"],
            arrowprops={"arrowstyle": "-", "lw": 0.8, "color": PALETTE["BSe9"]},
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.9},
        )
    ax.text(
        40,
        0.122,
        "0.10 eV",
        fontsize=6.3,
        color="#666666",
        bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.9},
    )
    label_offsets = {"LaCu12": 0.055, "LaSi9": -0.06, "BSe9": 0.16}
    label_x = {"LaCu12": 98.5, "LaSi9": 95.0, "BSe9": 95.0}
    for system in MAIN_SYSTEMS:
        end_value = curves[system][keep][-1]
        ax.text(
            label_x[system],
            end_value + label_offsets[system],
            f"{system} {end_value:.2f}",
            fontsize=6.4,
            ha="right",
            va="center",
            color=PALETTE[system],
        )

    ax.set_xlim(1, 100)
    ax.set_ylim(-0.02, 1.28)
    ax.set_xlabel("Budget K")
    ax.set_ylabel("Best-of-K replay gap (eV)")
    ax.set_title("Main-system best-of-K replay gap", loc="left", fontsize=9.2, pad=8)
    ax.grid(alpha=0.24, linewidth=0.65)
    ax.legend(loc="upper right", fontsize=7.2, ncol=3, handlelength=2.2)
    add_panel_label(ax, "b")
    return save_ppt_safe(fig, "Fig3b_main_system_gap")


def reason_spans(reasons: list[str]) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    if not reasons:
        return spans
    start = 1
    current = reasons[0]
    for idx, reason in enumerate(reasons[1:], start=2):
        if reason != current:
            spans.append((start, idx - 1, current))
            start = idx
            current = reason
    spans.append((start, len(reasons), current))
    return spans


def short_reason(reason: str) -> str:
    return {
        "low_predicted_energy": "low pred",
        "low_energy_high_risk": "high risk",
        "diversity_fill": "div",
    }.get(reason, reason.replace("_", " "))


def reason_color(reason: str) -> str:
    return {
        "low_predicted_energy": PALETTE["low_pred"],
        "low_energy_high_risk": PALETTE["high_risk"],
        "diversity_fill": PALETTE["diversity"],
    }.get(reason, "#F0F0F0")


def plot_dft_panel(system: str, label: str, out_stem: str) -> tuple[Path, Path]:
    rows = read_csv(DFT_DIR / f"phase6_final_dft_selection_{system}.csv")
    ranks = np.asarray([to_int(row["selection_rank"]) for row in rows], dtype=int)
    replay_gap = np.asarray([to_float(row["replay_relative_energy"]) for row in rows], dtype=float)
    dft_gap = np.asarray([to_float(row.get("dft_relative_energy_eV")) for row in rows], dtype=float)
    status = np.asarray([row.get("dft_status", "") for row in rows], dtype=object)
    reasons = [row.get("selection_reason", "") for row in rows]

    success = np.asarray([s == "dft_success" for s in status], dtype=bool) & np.isfinite(dft_gap)
    hit = success & (dft_gap <= 0.10)
    replay_best = np.minimum.accumulate(replay_gap)
    dft_for_best = np.where(success, dft_gap, np.inf)
    dft_best = np.minimum.accumulate(dft_for_best)
    dft_best[~np.isfinite(dft_best)] = np.nan

    fig, ax = plt.subplots(figsize=(7.6, 2.65))

    ymax = np.nanmax(np.concatenate([replay_best[np.isfinite(replay_best)], dft_gap[success]]))
    if system == "LaCu12":
        ymax = max(26.0, ymax)
    elif system == "LaSi9":
        ymax = max(2.2, ymax)
    else:
        ymax = max(2.0, ymax)

    for start, end, reason in reason_spans(reasons):
        ax.axvspan(start - 0.5, end + 0.5, ymin=0.92, ymax=1.0, color=reason_color(reason), alpha=0.75, linewidth=0)
        ax.text((start + end) / 2, ymax * 0.965, short_reason(reason), ha="center", va="center", fontsize=5.8, color="#666666")

    ax.step(ranks, replay_best, where="post", color=PALETTE["replay"], linewidth=2.2, label="Replay best", zorder=2)
    ax.step(ranks, dft_best, where="post", color=PALETTE["dft"], linewidth=2.3, label="DFT best", zorder=3)
    ax.scatter(
        ranks[success],
        dft_gap[success],
        s=30,
        marker="o",
        facecolor="white",
        edgecolor="#666666",
        linewidth=0.8,
        label="DFT completed",
        zorder=4,
    )
    ax.scatter(
        ranks[hit],
        dft_gap[hit],
        s=42,
        marker="o",
        facecolor=PALETTE["dft"],
        edgecolor="#262626",
        linewidth=0.7,
        label="DFT <=0.10 eV",
        zorder=5,
    )
    ax.axhline(0.10, color=PALETTE["threshold"], linewidth=1.0, linestyle="--")

    completed_n = int(success.sum())
    hit_n = int(hit.sum())
    ax.set_xlim(0.5, 30.5)
    ax.set_ylim(-0.03, ymax * 1.05)
    ax.set_xlabel("Selection rank in 30-candidate queue")
    ax.set_ylabel("Relative energy gap (eV)")
    ax.set_title(f"{label} - {system} ({completed_n}/30 DFT, {hit_n} hits <=0.10 eV)", loc="left", fontsize=9.2, pad=8)
    ax.grid(alpha=0.24, linewidth=0.65)
    add_panel_label(ax, label)

    handles = [
        mlines.Line2D([], [], color=PALETTE["replay"], linewidth=2.2, label="Replay best"),
        mlines.Line2D([], [], color=PALETTE["dft"], linewidth=2.3, label="DFT best"),
        mlines.Line2D([], [], color="none", marker="o", markerfacecolor="white", markeredgecolor="#666666", markersize=5.0, label="DFT completed"),
        mlines.Line2D([], [], color="none", marker="o", markerfacecolor=PALETTE["dft"], markeredgecolor="#262626", markersize=5.3, label="DFT <=0.10 eV"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=6.0, handlelength=2.0, labelspacing=0.22, borderaxespad=0.2)

    return save_ppt_safe(fig, out_stem)


def make_contact_sheet(paths: list[Path]) -> None:
    thumb_w = 760
    label_h = 36
    pad = 22
    font = ImageFont.load_default()
    thumbs: list[Image.Image] = []
    for path in paths:
        img = Image.open(path).convert("RGB")
        img.thumbnail((thumb_w, 280), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (thumb_w, img.height + label_h), "WHITE")
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 8), path.stem, fill=(0, 0, 0), font=font)
        canvas.paste(img, ((thumb_w - img.width) // 2, label_h))
        thumbs.append(canvas)

    sheet_h = sum(t.height for t in thumbs) + (len(thumbs) + 1) * pad
    sheet = Image.new("RGB", (thumb_w + 2 * pad, sheet_h), "WHITE")
    y = pad
    for thumb in thumbs:
        sheet.paste(thumb, (pad, y))
        y += thumb.height + pad
    sheet.save(QA_DIR / "Figure3_redrawn_individual_panels_contact.png", dpi=(200, 200))


def backup_and_replace_split_outputs(generated: list[tuple[str, Path, Path]]) -> None:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    for old in list(SPLIT_TIF.glob("Fig3*.tif")) + list(SPLIT_PNG.glob("Fig3*.png")):
        shutil.copy2(old, BACKUP_DIR / old.name)

    for stem, tif_path, png_path in generated:
        old_stem = OUTPUT_NAME_MAP[stem]
        shutil.copy2(tif_path, SPLIT_TIF / f"{old_stem}.tif")
        shutil.copy2(png_path, SPLIT_PNG / f"{old_stem}.png")


def write_manifest(rows: list[tuple[str, Path, Path]]) -> None:
    with (OUT_DIR / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["panel", "tif_uncompressed", "png_safe", "dpi", "copied_to_ppt_split_stem"])
        for stem, tif_path, png_path in rows:
            writer.writerow([stem, tif_path, png_path, DPI, OUTPUT_NAME_MAP[stem]])


def main() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    TIF_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    QA_DIR.mkdir(parents=True, exist_ok=True)

    setup_style()
    generated = [
        ("Fig3a_budget_recall", *plot_budget_recall()),
        ("Fig3b_main_system_gap", *plot_main_system_gap()),
        ("Fig3c_bse9_dft", *plot_dft_panel("BSe9", "c1", "Fig3c_bse9_dft")),
        ("Fig3d_lasi9_dft", *plot_dft_panel("LaSi9", "c2", "Fig3d_lasi9_dft")),
        ("Fig3e_lacu12_dft", *plot_dft_panel("LaCu12", "c3", "Fig3e_lacu12_dft")),
    ]
    make_contact_sheet([png for _, _, png in generated])
    backup_and_replace_split_outputs(generated)
    write_manifest(generated)

    print(f"Redrawn Figure 3 panels: {OUT_DIR}")
    print(f"Uncompressed TIFF: {TIF_DIR}")
    print(f"PNG backup: {PNG_DIR}")
    print(f"Old PPT-split Fig3 crops backed up to: {BACKUP_DIR}")
    print(f"Replaced Fig3 files in: {SPLIT_TIF} and {SPLIT_PNG}")


if __name__ == "__main__":
    main()
