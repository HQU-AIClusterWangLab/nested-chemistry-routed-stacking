from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


DPI = 600
FIG_DIR = Path(__file__).resolve().parent / "publication_figures"
OUT_DIR = FIG_DIR / "PPT_split_panels_20260709"
TIF_DIR = OUT_DIR / "tif"
PNG_DIR = OUT_DIR / "png_preview"
SRC_DIR = OUT_DIR / "_source_renders_600dpi"

POPPLER = (
    Path.home()
    / ".cache"
    / "codex-runtimes"
    / "codex-primary-runtime"
    / "dependencies"
    / "native"
    / "poppler"
    / "Library"
    / "bin"
)


@dataclass(frozen=True)
class Panel:
    figure: str
    label: str
    title: str
    box: tuple[float, float, float, float]


FIGURE_FILES = {
    "Fig1": "Figure1_workflow_publication",
    "Fig2": "Figure2_nested_router_publication",
    "Fig3": "Figure3_replay_and_dft_publication",
    "Fig4": "Figure4_postdft_chemistry_publication",
    "Fig5": "Figure5_model_architecture",
    "Fig6": "Figure6_ablation_ladder",
}


PANELS: list[Panel] = [
    # Figure 1: workflow schematic blocks.
    Panel("Fig1", "top_workflow", "top workflow row", (0.015, 0.035, 0.985, 0.505)),
    Panel("Fig1", "supporting_evidence", "supporting evidence row", (0.015, 0.515, 0.985, 0.860)),
    Panel("Fig1", "title_and_track", "title and track labels", (0.030, 0.035, 0.970, 0.205)),
    Panel("Fig1", "step1_dataset", "step 1 dataset", (0.012, 0.218, 0.205, 0.495)),
    Panel("Fig1", "step2_split", "step 2 leakage-safe split", (0.205, 0.218, 0.398, 0.495)),
    Panel("Fig1", "step3_branch_stack", "step 3 branch stack", (0.398, 0.218, 0.590, 0.495)),
    Panel("Fig1", "step4_nested_router", "step 4 nested router", (0.590, 0.218, 0.782, 0.495)),
    Panel("Fig1", "step5_dft_validation", "step 5 DFT validation", (0.782, 0.218, 0.974, 0.495)),
    Panel("Fig1", "support_main_systems", "main-text systems block", (0.015, 0.605, 0.340, 0.855)),
    Panel("Fig1", "support_metrics", "performance metrics block", (0.340, 0.605, 0.655, 0.855)),
    Panel("Fig1", "support_claim_boundary", "claim boundary block", (0.655, 0.605, 0.985, 0.855)),

    # Figure 2: original scientific panels.
    Panel("Fig2", "panel_a_outerheldout_mae", "panel a outer-held-out MAE", (0.040, 0.050, 0.630, 0.955)),
    Panel("Fig2", "panel_b_workflow_summary", "panel b workflow summary", (0.630, 0.055, 0.985, 0.470)),
    Panel("Fig2", "panel_c_router_regret", "panel c router regret", (0.630, 0.470, 0.985, 0.955)),

    # Figure 3: stacked panels.
    Panel("Fig3", "panel_a_budget_recall", "panel a budget recall", (0.085, 0.050, 0.955, 0.230)),
    Panel("Fig3", "panel_b_main_system_gap", "panel b main-system gap", (0.085, 0.235, 0.955, 0.385)),
    Panel("Fig3", "panel_c_bse9_dft", "panel c BSe9 DFT", (0.085, 0.395, 0.955, 0.575)),
    Panel("Fig3", "panel_d_lasi9_dft", "panel d LaSi9 DFT", (0.085, 0.585, 0.955, 0.765)),
    Panel("Fig3", "panel_e_lacu12_dft", "panel e LaCu12 DFT", (0.085, 0.775, 0.955, 0.955)),

    # Figure 4: each row plus CDD/ELF subpanels and colorbars.
    Panel("Fig4", "panel_a_row_bse9", "panel a BSe9 row", (0.120, 0.055, 0.875, 0.320)),
    Panel("Fig4", "panel_b_row_lasi9", "panel b LaSi9 row", (0.120, 0.330, 0.875, 0.595)),
    Panel("Fig4", "panel_c_row_lacu12", "panel c LaCu12 row", (0.120, 0.605, 0.875, 0.875)),
    Panel("Fig4", "panel_a_cdd", "panel a CDD map", (0.230, 0.040, 0.505, 0.285)),
    Panel("Fig4", "panel_a_elf", "panel a ELF map", (0.525, 0.040, 0.795, 0.285)),
    Panel("Fig4", "panel_b_cdd", "panel b CDD map", (0.230, 0.315, 0.505, 0.560)),
    Panel("Fig4", "panel_b_elf", "panel b ELF map", (0.525, 0.315, 0.795, 0.560)),
    Panel("Fig4", "panel_c_cdd", "panel c CDD map", (0.230, 0.590, 0.505, 0.845)),
    Panel("Fig4", "panel_c_elf", "panel c ELF map", (0.525, 0.590, 0.795, 0.845)),
    Panel("Fig4", "colorbar_strip", "CDD and ELF colorbars", (0.165, 0.875, 0.805, 0.982)),

    # Figure 5: PPT-editable architecture blocks.
    Panel("Fig5", "input_block", "input block", (0.285, 0.105, 0.720, 0.285)),
    Panel("Fig5", "expert_pool", "expert pool block", (0.010, 0.350, 0.060, 0.435)),
    Panel("Fig5", "branch_schnet", "SchNet-static branch", (0.040, 0.275, 0.345, 0.765)),
    Panel("Fig5", "branch_paa", "PAA-SchNet-dynamic branch", (0.355, 0.275, 0.640, 0.765)),
    Panel("Fig5", "branch_painn", "PaiNN branch", (0.665, 0.275, 0.955, 0.765)),
    Panel("Fig5", "expert_branches_row", "three expert branches row", (0.040, 0.180, 0.970, 0.765)),
    Panel("Fig5", "nested_router", "nested router block", (0.070, 0.720, 0.930, 0.930)),
    Panel("Fig5", "final_prediction", "final prediction block", (0.370, 0.895, 0.635, 0.965)),

    # Figure 6: ablation ladder components.
    Panel("Fig6", "title", "title", (0.030, 0.050, 0.760, 0.165)),
    Panel("Fig6", "ladder_body", "full ablation ladder body", (0.050, 0.210, 0.975, 0.965)),
    Panel("Fig6", "step1_bare_schnet", "Bare SchNet step", (0.060, 0.220, 0.970, 0.385)),
    Panel("Fig6", "step2_schnet_phys", "SchNet + Phys step", (0.060, 0.390, 0.970, 0.555)),
    Panel("Fig6", "step3_paa_attention", "PAA-SchNet attention step", (0.060, 0.565, 0.970, 0.730)),
    Panel("Fig6", "step4_nested_router", "Nested router step", (0.060, 0.735, 0.970, 0.950)),
]


def flatten_white(img: Image.Image) -> Image.Image:
    if img.mode == "RGB":
        return img
    rgba = img.convert("RGBA")
    bg = Image.new("RGBA", rgba.size, "WHITE")
    bg.alpha_composite(rgba)
    return bg.convert("RGB")


def render_pdf_600(pdf_path: Path, out_path: Path) -> tuple[Image.Image, str]:
    try:
        from pdf2image import convert_from_path
    except Exception as exc:  # pragma: no cover - runtime dependency guard
        raise RuntimeError(f"pdf2image unavailable: {exc}") from exc

    pages = convert_from_path(
        str(pdf_path),
        dpi=DPI,
        first_page=1,
        last_page=1,
        poppler_path=str(POPPLER),
    )
    img = flatten_white(pages[0])
    img.save(out_path, dpi=(DPI, DPI))
    return img, f"pdf:{pdf_path.name}"


def load_source(fig_key: str) -> tuple[Image.Image, str]:
    stem = FIGURE_FILES[fig_key]
    pdf_path = FIG_DIR / f"{stem}.pdf"
    cached = SRC_DIR / f"{stem}_source600.png"
    if pdf_path.exists():
        try:
            return render_pdf_600(pdf_path, cached)
        except Exception as exc:
            print(f"[WARN] PDF render failed for {pdf_path.name}: {exc}")

    for ext in (".tiff", ".tif", ".png"):
        path = FIG_DIR / f"{stem}{ext}"
        if path.exists():
            img = flatten_white(Image.open(path))
            return img, f"raster:{path.name}"
    raise FileNotFoundError(stem)


def norm_to_box(size: tuple[int, int], box: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    width, height = size
    x0, y0, x1, y1 = box
    return (
        max(0, round(x0 * width)),
        max(0, round(y0 * height)),
        min(width, round(x1 * width)),
        min(height, round(y1 * height)),
    )


def save_panel(img: Image.Image, panel: Panel, source_desc: str) -> dict[str, str | int]:
    crop_box = norm_to_box(img.size, panel.box)
    crop = img.crop(crop_box)
    safe_title = panel.label.replace(" ", "_")
    out_stem = f"{panel.figure}_{safe_title}"
    tif_path = TIF_DIR / f"{out_stem}.tif"
    png_path = PNG_DIR / f"{out_stem}.png"
    crop.save(tif_path, compression="tiff_lzw", dpi=(DPI, DPI))
    crop.save(png_path, dpi=(DPI, DPI))
    return {
        "file_tif": str(tif_path),
        "file_png_preview": str(png_path),
        "figure": panel.figure,
        "panel": panel.label,
        "title": panel.title,
        "source": source_desc,
        "source_width_px": img.size[0],
        "source_height_px": img.size[1],
        "crop_left_px": crop_box[0],
        "crop_top_px": crop_box[1],
        "crop_right_px": crop_box[2],
        "crop_bottom_px": crop_box[3],
        "width_px": crop.size[0],
        "height_px": crop.size[1],
        "dpi": DPI,
    }


def contact_sheet(preview_paths: Iterable[Path], out_path: Path) -> None:
    paths = list(preview_paths)
    thumb_w = 460
    label_h = 44
    pad = 24
    cols = 4
    font = ImageFont.load_default()
    thumbs: list[tuple[Path, Image.Image]] = []
    for path in paths:
        img = Image.open(path).convert("RGB")
        img.thumbnail((thumb_w, 320), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (thumb_w, img.height + label_h), "WHITE")
        canvas.paste(img, ((thumb_w - img.width) // 2, label_h))
        draw = ImageDraw.Draw(canvas)
        draw.text((6, 8), path.stem, fill=(0, 0, 0), font=font)
        thumbs.append((path, canvas))

    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new(
        "RGB",
        (cols * thumb_w + (cols + 1) * pad, rows * (320 + label_h) + (rows + 1) * pad),
        "WHITE",
    )
    for idx, (_, thumb) in enumerate(thumbs):
        row, col = divmod(idx, cols)
        x = pad + col * (thumb_w + pad)
        y = pad + row * (320 + label_h)
        sheet.paste(thumb, (x, y))
    sheet.save(out_path, dpi=(200, 200))


def main() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    TIF_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    SRC_DIR.mkdir(parents=True, exist_ok=True)

    by_figure: dict[str, list[Panel]] = {}
    for panel in PANELS:
        by_figure.setdefault(panel.figure, []).append(panel)

    rows: list[dict[str, str | int]] = []
    preview_paths: list[Path] = []
    for fig_key, panels in by_figure.items():
        img, source_desc = load_source(fig_key)
        for panel in panels:
            row = save_panel(img, panel, source_desc)
            rows.append(row)
            preview_paths.append(Path(str(row["file_png_preview"])))
        img.close()

    manifest = OUT_DIR / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    contact_sheet(preview_paths, OUT_DIR / "_QA_contact_sheet.png")

    print(f"Saved {len(rows)} TIFF panels to: {TIF_DIR}")
    print(f"PNG previews: {PNG_DIR}")
    print(f"Manifest: {manifest}")
    print(f"QA contact sheet: {OUT_DIR / '_QA_contact_sheet.png'}")


if __name__ == "__main__":
    main()
