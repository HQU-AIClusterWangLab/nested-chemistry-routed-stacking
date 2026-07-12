"""
Figure X: Ablation ladder for the final nested-router workflow.

Goal:
show the model as a computational-chemistry step diagram, not a bar chart
with decorative arrows.
"""

from __future__ import annotations

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


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
BASENAME = "Figure6_ablation_ladder"

P = {
    "ink": "#2D2D2D",
    "muted": "#6E6E6E",
    "neutral": "#5A5A5A",
    "grid": "#E8E8E8",
    "blue1": "#5D9BD3",
    "blue2": "#2D67B0",
    "red": "#C24E4B",
    "green": "#3B8D40",
    "step_bg": "#FAFAFA",
}

steps = [
    dict(name="Bare\nSchNet", mae=4.13, desc="Pure distance features\nno physics prior", color=P["neutral"]),
    dict(name="SchNet\n+ Phys", mae=2.93, desc="+ static physics features\n(e-neg, cov. radius, valence)", color=P["blue1"], delta="−29%"),
    dict(name="PAA-SchNet\n(+ Attention)", mae=2.75, desc="+ PAA dual-branch attention\n(gate · scale joint weighting)", color=P["blue2"], delta="−6.1%"),
    dict(name="Nested Router\n(Stacked)", mae=1.93, desc="+ 3-expert stacking\n+ nested routing selection", color=P["red"], delta="−30%"),
]

fig, ax = plt.subplots(figsize=(183 / 25.4, 128 / 25.4))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")
ax.set_facecolor("white")

ax.text(1.6, 95.4, "Ablation Ladder  —  LOSO Mean MAE (Energy, eV)", fontsize=12.2, fontweight="bold", color=P["ink"], ha="left")

row_y = [72, 52, 32, 12]
for i, (step, y) in enumerate(zip(steps, row_y)):
    if i % 2 == 0:
        ax.add_patch(Rectangle((1.5, y - 8.6), 97, 17.2, facecolor=P["step_bg"], edgecolor="none", zorder=0))
    ax.plot([25, 72.4], [y, y], color=step["color"], lw=4.0, solid_capstyle="round", zorder=3)
    ax.plot([74.0], [y], marker="o", ms=12, color=step["color"], markeredgecolor="white", markeredgewidth=1.3, zorder=4)
    offset = -5.2 if i < len(steps) - 1 else -4.8
    ax.text(74.0, y + offset, f"{step['mae']:.2f} eV", fontsize=12.0, fontweight="bold", color=step["color"], ha="center", va="center")
    ax.text(20.5, y, step["name"], fontsize=9.2, fontweight="bold", color=step["color"], ha="right", va="center", linespacing=1.1)
    ax.text(79.2, y, step["desc"], fontsize=6.8, color=P["ink"], ha="left", va="center", linespacing=1.18)

for idx in range(len(row_y) - 1):
    y1, y2 = row_y[idx], row_y[idx + 1]
    ax.plot([50, 50], [y1 - 1.0, y2 + 1.0], color=P["muted"], lw=1.2, ls="--", zorder=1)
    ax.annotate("", xy=(50, y2 + 1.3), xytext=(50, y1 - 1.3), arrowprops=dict(arrowstyle="-|>", color=P["muted"], lw=1.2, mutation_scale=9))
    delta = steps[idx + 1].get("delta")
    if delta:
        ax.text(52.8, (y1 + y2) / 2, delta, fontsize=9.2, fontweight="bold", color=P["green"], ha="left", va="center")

fig.tight_layout(pad=0.2)
for ext, dpi, kw in [
    ("svg", None, {}),
    ("pdf", None, {}),
    ("png", 360, {}),
    ("tiff", 360, {"pil_kwargs": {"compression": "tiff_lzw"}}),
]:
    fig.savefig(os.path.join(OUT_DIR, f"{BASENAME}.{ext}"), dpi=dpi, bbox_inches="tight", pad_inches=0.03, **kw)

plt.close(fig)
