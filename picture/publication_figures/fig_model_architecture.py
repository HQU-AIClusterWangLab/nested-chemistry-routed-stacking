"""
Figure 5: Model architecture for the final nested-router workflow.

Design goal:
make the composition read as one clear argument:
shared inputs -> three experts -> nested router -> final energy.

Outputs:
SVG, PDF, PNG, TIFF
"""

from __future__ import annotations

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7.2,
        "axes.linewidth": 0.8,
    }
)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
BASENAME = "Figure5_model_architecture"


C = {
    "bg": "#FFFFFF",
    "ink": "#2C2C2C",
    "muted": "#6E6E6E",
    "line": "#6A6A6A",
    "panel": "#F4F4F4",
    "input": "#DCEBF7",
    "schnet": "#DFF0D8",
    "paa": "#FFF2CC",
    "painn": "#E7D8F4",
    "router": "#F7D7D0",
    "accent": "#C24E4B",
    "accent2": "#E28E2C",
    "blue_dark": "#1F5AA6",
    "green": "#3C8D40",
}


def rbox(ax, x, y, w, h, fc, ec, lw=1.1, radius=0.02, z=2):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=z,
    )
    ax.add_patch(box)
    return box


def txt(ax, x, y, s, size=7, weight="normal", color=None, ha="center", va="center", z=5):
    ax.text(
        x,
        y,
        s,
        fontsize=size,
        fontweight=weight,
        color=color or C["ink"],
        ha=ha,
        va=va,
        zorder=z,
    )


def arrow(ax, x1, y1, x2, y2, color=None, lw=1.0, style="-|>", ms=10, z=3):
    a = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle=style,
        mutation_scale=ms,
        linewidth=lw,
        color=color or C["line"],
        shrinkA=0,
        shrinkB=0,
        connectionstyle="arc3,rad=0",
        zorder=z,
    )
    ax.add_patch(a)
    return a


fig, ax = plt.subplots(figsize=(183 / 25.4, 130 / 25.4))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")
ax.set_facecolor(C["bg"])

# Background bands.
ax.add_patch(Rectangle((0, 68), 100, 32, facecolor=C["panel"], edgecolor="none", zorder=0))
ax.add_patch(Rectangle((0, 32), 100, 36, facecolor="#FBFBFB", edgecolor="none", zorder=0))

# Input block: widened to prevent text from touching the frame.
rbox(ax, 27.5, 82.0, 45.0, 11.7, C["input"], C["line"], lw=1.35, radius=0.03)
txt(
    ax,
    50,
    88.0,
    "Input: {Z, R, phys. features}\ndistance matrix, e-neg, cov. radius, valence",
    size=7.6,
    weight="bold",
)
arrow(ax, 50, 82.0, 50, 73.7, lw=1.15)
txt(ax, 61.2, 76.8, "shared features", size=6.0, color=C["muted"], ha="left")

# Expert pool label: a compact side tag separated from the first card border.
rbox(ax, 0.9, 57.6, 3.4, 6.6, "#FFFFFF", "#BDBDBD", lw=0.75, radius=0.015, z=3)
txt(ax, 2.6, 60.9, "Expert\npool", size=4.45, weight="bold")

cards = [
    {
        "x": 5,
        "y": 39,
        "w": 28,
        "h": 31,
        "fc": C["schnet"],
        "title": "SchNet-static",
        "blocks": [
            ("dist.\nembed", 8.0, 55.8, 10.2, "#EEF6EE"),
            ("cfconv\n(filter)", 20.0, 55.8, 11.5, "#EEF6EE"),
            ("atom-\nwise", 8.0, 47.4, 10.2, "#D8EFD8"),
            ("E, F\nhead", 20.0, 47.4, 11.5, "#D8EFD8"),
        ],
        "foot": "[distance only]",
        "mid": 18,
    },
    {
        "x": 36,
        "y": 39,
        "w": 28,
        "h": 31,
        "fc": C["paa"],
        "title": "PAA-SchNet-dynamic",
        "blocks": [
            ("dist.\nembed", 39.0, 55.8, 10.2, "#FFF9D8"),
            ("PAA cfconv\n(dual-weighted)", 51.2, 55.8, 11.5, "#FFE18B"),
            ("atom-\nwise", 39.0, 47.4, 10.2, "#FFD58E"),
            ("E, F\nhead", 51.2, 47.4, 11.5, "#FFD58E"),
        ],
        "foot": "[+ dynamic phys. features]",
        "mid": 50,
    },
    {
        "x": 67,
        "y": 39,
        "w": 28,
        "h": 31,
        "fc": C["painn"],
        "title": "PaiNN",
        "blocks": [
            ("message\n(s+t)", 70.0, 55.8, 10.2, "#E9D8F3"),
            ("update\n(s+t)", 82.2, 55.8, 10.2, "#E9D8F3"),
            ("scalar\nreadout", 70.0, 47.4, 10.2, "#DAB5EB"),
            ("vector\nreadout", 82.2, 47.4, 10.2, "#DAB5EB"),
        ],
        "foot": "[equivariant msg-passing]",
        "mid": 82,
    },
]

for c in cards:
    rbox(ax, c["x"], c["y"], c["w"], c["h"], c["fc"], "#A8A8A8", lw=1.15, radius=0.03)
    txt(ax, c["x"] + c["w"] / 2, 66.6, c["title"], size=8.5, weight="bold")
    for label, bx, by, bw, fc in c["blocks"]:
        rbox(ax, bx, by, bw, 5.7, fc, "#A8A8A8", lw=0.8, radius=0.02)
        txt(ax, bx + bw / 2, by + 2.85, label, size=5.7)
    txt(ax, c["x"] + c["w"] / 2, c["y"] + 3.6, c["foot"], size=5.6, color=C["muted"])
    arrow(ax, c["mid"], 80.7, c["mid"], 70.0, lw=1.0)

# Internal horizontal connectors.
arrow(ax, 18.1, 58.5, 19.9, 58.5, lw=0.8)
arrow(ax, 50.1, 58.5, 51.4, 58.5, lw=0.8)
arrow(ax, 82.3, 58.5, 83.4, 58.5, lw=0.8)

# PAA gate-scale inset: label is internal, no text hangs outside the box.
rbox(ax, 38.0, 30.3, 24.0, 8.5, "#FFE2BB", C["accent2"], lw=1.2, radius=0.02, z=2)
txt(ax, 50.0, 37.55, "gate-scale coupling", size=5.25, weight="bold", color=C["accent2"])
rbox(ax, 39.1, 31.15, 10.3, 4.3, "#FFF1C6", "#A8A8A8", lw=0.7, radius=0.015, z=3)
rbox(ax, 50.8, 31.15, 10.3, 4.3, "#FFD5C5", "#A8A8A8", lw=0.7, radius=0.015, z=3)
txt(ax, 44.25, 34.15, "gate g_ij", size=4.45)
txt(ax, 44.25, 32.55, "r_ij, phys", size=4.45, color=C["muted"])
txt(ax, 55.95, 34.15, "scale s_ij", size=4.45)
txt(ax, 55.95, 32.05, "g_ij * s_ij", size=4.55, weight="bold", color=C["accent2"])
arrow(ax, 50, 39.0, 50, 38.8, lw=1.0)
arrow(ax, 50, 30.3, 50, 28.4, lw=1.0)

# Expert outputs.
for x in [18, 50, 82]:
    rbox(ax, x - 5, 22.0, 10, 6.2, "#E0E0E0", "#A6A6A6", lw=1.0, radius=0.02)
    txt(ax, x, 25.1, "E, F", size=7.3, weight="bold")

# Non-PAA experts route directly down to their outputs.
arrow(ax, 18, 39.0, 18, 28.2, lw=1.0)
arrow(ax, 82, 39.0, 82, 28.2, lw=1.0)

# Nested router block: taller bottom region prevents arrows from crossing labels.
rbox(ax, 8, 1.8, 84, 18.8, "#FAEAEA", C["accent"], lw=1.4, radius=0.02)
txt(ax, 50, 19.1, "Nested Router (Two-Stage)", size=8.4, weight="bold", color=C["accent"])

rbox(ax, 16.0, 10.6, 26.5, 6.8, "#F6CFCB", "#A6A6A6", lw=1.0, radius=0.02)
txt(
    ax,
    29.25,
    13.9,
    "Stage 1: Branch Selection\nrule-based routing\nLa-gate / non-La / extreme-OOD",
    size=4.85,
)
rbox(ax, 57.5, 10.6, 26.5, 6.8, "#F6CFCB", "#A6A6A6", lw=1.0, radius=0.02)
txt(
    ax,
    70.75,
    13.9,
    "Stage 2: Gated Fusion\ntrained MLP gate (32->16->4)\nsoftmax weights {w_i}",
    size=4.85,
)

arrow(ax, 42.5, 13.9, 57.5, 13.9, color=C["accent"], lw=1.15, ms=10)
rbox(ax, 37.2, 3.5, 25.6, 5.5, "#F8D1D6", C["accent"], lw=1.2, radius=0.02)
txt(ax, 50, 6.25, "Final Prediction:\nE (total energy)", size=6.65, weight="bold")

for x in [18, 50, 82]:
    arrow(ax, x, 22.0, x, 20.6, lw=1.0)

arrow(ax, 50, 13.8, 50, 9.0, color=C["accent"], lw=1.0)

fig.tight_layout(pad=0.2)

for ext, dpi, kw in [
    ("svg", None, {}),
    ("pdf", None, {}),
    ("png", 360, {}),
    ("tiff", 600, {"pil_kwargs": {"compression": "tiff_lzw"}}),
]:
    fig.savefig(os.path.join(OUT_DIR, f"{BASENAME}.{ext}"), dpi=dpi, bbox_inches="tight", pad_inches=0.03, **kw)

plt.close(fig)
