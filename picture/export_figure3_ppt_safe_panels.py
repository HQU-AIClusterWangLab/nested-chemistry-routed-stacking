from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


DPI = 600
ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "picture" / "publication_figures"
OUT_DIR = FIG_DIR / "Figure3_PPT_safe_panels_20260709"
TIF_DIR = OUT_DIR / "tif_uncompressed"
PNG_DIR = OUT_DIR / "png_safe"
QA_DIR = OUT_DIR / "qa"

PDF_PATH = FIG_DIR / "Figure3_replay_and_dft_publication.pdf"
SOURCE_RENDER = QA_DIR / "Figure3_source_600dpi.png"

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
class CropDef:
    name: str
    title: str
    box: tuple[float, float, float, float]


# These boxes intentionally include more white margin than the manuscript crop.
# PowerPoint can clip very thin line-art elements when a panel is tightly cropped
# and re-exported from slides, so these are PPT-safe strips rather than tight cuts.
CROPS = [
    CropDef("Fig3_full_figure", "full Figure 3", (0.000, 0.000, 1.000, 1.000)),
    CropDef("Fig3a_budget_recall_safe", "panel a budget recall, PPT-safe", (0.020, 0.030, 0.985, 0.250)),
    CropDef("Fig3b_main_system_gap_safe", "panel b main-system gap, PPT-safe", (0.020, 0.205, 0.985, 0.410)),
    CropDef("Fig3c_bse9_dft_safe", "panel c BSe9 DFT, PPT-safe", (0.020, 0.365, 0.985, 0.600)),
    CropDef("Fig3d_lasi9_dft_safe", "panel d LaSi9 DFT, PPT-safe", (0.020, 0.555, 0.985, 0.790)),
    CropDef("Fig3e_lacu12_dft_safe", "panel e LaCu12 DFT, PPT-safe", (0.020, 0.745, 0.985, 0.985)),
]


def flatten_white(img: Image.Image) -> Image.Image:
    if img.mode == "RGB":
        return img
    rgba = img.convert("RGBA")
    bg = Image.new("RGBA", rgba.size, "WHITE")
    bg.alpha_composite(rgba)
    return bg.convert("RGB")


def render_source() -> Image.Image:
    try:
        from pdf2image import convert_from_path
    except Exception as exc:
        raise RuntimeError(f"pdf2image is unavailable: {exc}") from exc

    if PDF_PATH.exists():
        pages = convert_from_path(
            str(PDF_PATH),
            dpi=DPI,
            first_page=1,
            last_page=1,
            poppler_path=str(POPPLER),
        )
        img = flatten_white(pages[0])
        img.save(SOURCE_RENDER, dpi=(DPI, DPI))
        return img

    # Fallback only if the PDF is absent.
    fallback = FIG_DIR / "Figure3_replay_and_dft_publication.tiff"
    return flatten_white(Image.open(fallback))


def norm_box(size: tuple[int, int], box: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    w, h = size
    x0, y0, x1, y1 = box
    return (
        max(0, round(x0 * w)),
        max(0, round(y0 * h)),
        min(w, round(x1 * w)),
        min(h, round(y1 * h)),
    )


def save_crop(source: Image.Image, crop_def: CropDef) -> dict[str, str | int]:
    box = norm_box(source.size, crop_def.box)
    crop = source.crop(box)
    tif_path = TIF_DIR / f"{crop_def.name}.tif"
    png_path = PNG_DIR / f"{crop_def.name}.png"

    # No TIFF compression: PowerPoint handles these more reliably than LZW for
    # very thin line-art panels.
    crop.save(tif_path, dpi=(DPI, DPI))
    crop.save(png_path, dpi=(DPI, DPI))

    return {
        "panel": crop_def.name,
        "title": crop_def.title,
        "tif_uncompressed": str(tif_path),
        "png_safe": str(png_path),
        "source": str(PDF_PATH),
        "source_width_px": source.size[0],
        "source_height_px": source.size[1],
        "left_px": box[0],
        "top_px": box[1],
        "right_px": box[2],
        "bottom_px": box[3],
        "width_px": crop.size[0],
        "height_px": crop.size[1],
        "dpi": DPI,
    }


def draw_crop_guide(source: Image.Image) -> None:
    guide = source.copy()
    guide.thumbnail((1100, 1500), Image.Resampling.LANCZOS)
    sx = guide.size[0] / source.size[0]
    sy = guide.size[1] / source.size[1]
    draw = ImageDraw.Draw(guide)
    font = ImageFont.load_default()
    colors = ["#D73027", "#4575B4", "#1A9850", "#984EA3", "#FF7F00", "#333333"]
    for idx, crop_def in enumerate(CROPS[1:]):
        left, top, right, bottom = norm_box(source.size, crop_def.box)
        rect = (round(left * sx), round(top * sy), round(right * sx), round(bottom * sy))
        color = colors[idx % len(colors)]
        draw.rectangle(rect, outline=color, width=3)
        draw.text((rect[0] + 6, rect[1] + 6), crop_def.name, fill=color, font=font)
    guide.save(QA_DIR / "Figure3_crop_guide.png", dpi=(200, 200))


def contact_sheet(paths: list[Path]) -> None:
    thumb_w = 620
    label_h = 36
    pad = 22
    cols = 2
    font = ImageFont.load_default()
    thumbs: list[Image.Image] = []
    for path in paths:
        img = Image.open(path).convert("RGB")
        img.thumbnail((thumb_w, 360), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (thumb_w, img.height + label_h), "WHITE")
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 8), path.stem, fill=(0, 0, 0), font=font)
        canvas.paste(img, ((thumb_w - img.width) // 2, label_h))
        thumbs.append(canvas)

    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb_w + (cols + 1) * pad, rows * 430 + (rows + 1) * pad), "WHITE")
    for i, thumb in enumerate(thumbs):
        row, col = divmod(i, cols)
        sheet.paste(thumb, (pad + col * (thumb_w + pad), pad + row * 430))
    sheet.save(QA_DIR / "Figure3_PPT_safe_contact_sheet.png", dpi=(200, 200))


def main() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    TIF_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    QA_DIR.mkdir(parents=True, exist_ok=True)

    source = render_source()
    rows = [save_crop(source, crop_def) for crop_def in CROPS]
    draw_crop_guide(source)
    contact_sheet([Path(str(row["png_safe"])) for row in rows])

    manifest = OUT_DIR / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} Figure 3 PPT-safe panels")
    print(f"Uncompressed TIFF: {TIF_DIR}")
    print(f"PNG backup: {PNG_DIR}")
    print(f"QA: {QA_DIR}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
