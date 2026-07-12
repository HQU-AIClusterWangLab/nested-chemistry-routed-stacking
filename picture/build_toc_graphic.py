from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image


OUT_DIR = Path(r"D:\lunwen\2.1sci\picture")
BASE_IMAGE = OUT_DIR / "TOC_AI_base_illustration.png"
BASE = OUT_DIR / "TOC_graphic_publication"

OUTPUT_DPI = 600
TARGET_W = 1950
TARGET_H = 1050


def center_crop_to_aspect(img, target_ratio):
    width, height = img.size
    current_ratio = width / height
    if current_ratio > target_ratio:
        new_width = int(height * target_ratio)
        left = (width - new_width) // 2
        return img.crop((left, 0, left + new_width, height))
    new_height = int(width / target_ratio)
    top = (height - new_height) // 2
    return img.crop((0, top, width, top + new_height))


def add_overlay_label(ax, x, y, text, size, color, box_fc=None, box_ec=None, weight="bold"):
    bbox = None
    if box_fc:
        bbox = dict(boxstyle="round,pad=0.14,rounding_size=0.06", fc=box_fc, ec=box_ec, lw=1.05, alpha=0.9)
    txt = ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=size,
        fontweight=weight,
        color=color,
        family="Arial",
        bbox=bbox,
    )
    txt.set_path_effects([pe.withStroke(linewidth=1.4, foreground="white", alpha=0.62)])
    return txt


def build():
    if not BASE_IMAGE.exists():
        raise FileNotFoundError(f"Missing AI base illustration: {BASE_IMAGE}")

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )

    img = Image.open(BASE_IMAGE).convert("RGB")
    img = center_crop_to_aspect(img, TARGET_W / TARGET_H)
    img = img.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)

    fig, ax = plt.subplots(figsize=(3.25, 1.75), dpi=OUTPUT_DPI)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.imshow(img, extent=[0, 1, 0, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Keep scientific text deterministic; the AI base image intentionally has none.
    add_overlay_label(ax, 0.528, 0.790, "route", 7.7, "#173B54", box_fc="#E7F1F8", box_ec="#315E7C")
    add_overlay_label(ax, 0.600, 0.560, "DFT", 7.6, "#6B4E16", box_fc="#FFF0C9", box_ec="#95723D")
    add_overlay_label(ax, 0.842, 0.225, "low-E", 7.2, "#23482E", box_fc="#E8F3E7", box_ec="#557A5A")

    ax.add_patch(
        FancyArrowPatch(
            (0.45, 0.48),
            (0.51, 0.48),
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=1.1,
            color="#20465E",
            alpha=0.9,
        )
    )
    ax.add_patch(
        FancyArrowPatch(
            (0.635, 0.49),
            (0.735, 0.37),
            connectionstyle="arc3,rad=-0.16",
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=1.05,
            color="#4C6C4E",
            alpha=0.88,
        )
    )

    fig.savefig(BASE.with_suffix(".png"), dpi=OUTPUT_DPI)
    fig.savefig(BASE.with_suffix(".tiff"), dpi=OUTPUT_DPI)
    fig.savefig(BASE.with_suffix(".pdf"))
    fig.savefig(BASE.with_suffix(".svg"))
    plt.close(fig)
    print(BASE)


if __name__ == "__main__":
    build()
