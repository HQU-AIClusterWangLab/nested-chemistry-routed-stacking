from pathlib import Path
import re

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "picture" / "publication_figures"

TITLE_CROPS_PX = {
    "FigureS6_low_energy_basin": 120,
    "FigureS7_dft_energy_ladder": 110,
    "FigureS8_mayer_wiberg_vs_attention": 135,
}

TITLE_PATTERNS = {
    "FigureS6_low_energy_basin": "Figure S6",
    "FigureS7_dft_energy_ladder": "Figure S7",
    "FigureS8_mayer_wiberg_vs_attention": "Figure S8",
}


def image_dpi(path: Path, default: int) -> tuple[int, int]:
    with Image.open(path) as image:
        dpi = image.info.get("dpi")
    if isinstance(dpi, tuple) and len(dpi) >= 2 and dpi[0] and dpi[1]:
        return int(round(dpi[0])), int(round(dpi[1]))
    return default, default


def crop_raster(stem: str, crop_px: int, suffix: str, dpi: int) -> None:
    path = FIG_DIR / f"{stem}.{suffix}"
    if not path.exists():
        return
    tmp_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
    with Image.open(path) as image:
        cropped = image.crop((0, crop_px, image.width, image.height))
        save_kwargs = {}
        if suffix.lower() in {"png", "tif", "tiff"}:
            save_kwargs["dpi"] = (dpi, dpi)
        cropped.save(tmp_path, **save_kwargs)
    try:
        tmp_path.replace(path)
    except PermissionError:
        print(f"Skipped locked raster file: {path}")
        if tmp_path.exists():
            tmp_path.unlink()


def svg_size_points(svg_text: str) -> tuple[float, float]:
    viewbox = re.search(r'viewBox="([^"]+)"', svg_text)
    if viewbox:
        parts = [float(x) for x in viewbox.group(1).replace(",", " ").split()]
        if len(parts) == 4:
            return parts[2], parts[3]
    width = re.search(r'width="([\d.]+)pt"', svg_text)
    height = re.search(r'height="([\d.]+)pt"', svg_text)
    if not width or not height:
        raise ValueError("Cannot determine SVG size in points")
    return float(width.group(1)), float(height.group(1))


def strip_title_text_groups(svg_text: str, title_pattern: str) -> str:
    lines = svg_text.splitlines()
    output: list[str] = []
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if title_pattern in line:
            while output and "<g" not in output[-1]:
                output.pop()
            if output and "<g" in output[-1]:
                output.pop()
            idx += 1
            depth = 1
            while idx < len(lines) and depth > 0:
                depth += lines[idx].count("<g")
                depth -= lines[idx].count("</g>")
                idx += 1
            continue
        output.append(line)
        idx += 1
    return "\n".join(output) + "\n"


def crop_svg(stem: str, crop_px: int, png_width_px: int) -> float:
    path = FIG_DIR / f"{stem}.svg"
    if not path.exists():
        return 0.0
    text = path.read_text(encoding="utf-8")
    if TITLE_PATTERNS[stem] not in text:
        return 0.0
    width_pt, height_pt = svg_size_points(text)
    crop_pt = crop_px / (png_width_px / width_pt)
    text = strip_title_text_groups(text, TITLE_PATTERNS[stem])
    new_height = height_pt - crop_pt
    text = re.sub(r'height="[\d.]+pt"', f'height="{new_height:.6f}pt"', text, count=1)
    if re.search(r'viewBox="[^"]+"', text):
        def repl(match: re.Match[str]) -> str:
            x0, y0, width, height = [float(x) for x in match.group(1).replace(",", " ").split()]
            return f'viewBox="{x0:.6f} {y0 + crop_pt:.6f} {width:.6f} {height - crop_pt:.6f}"'

        text = re.sub(r'viewBox="([^"]+)"', repl, text, count=1)
    path.write_text(text, encoding="utf-8")
    return crop_pt


def crop_pdf(stem: str, crop_pt: float) -> None:
    path = FIG_DIR / f"{stem}.pdf"
    if not path.exists() or crop_pt <= 0:
        return
    from pypdf import PdfReader, PdfWriter

    reader = PdfReader(str(path))
    writer = PdfWriter()
    for page in reader.pages:
        box = page.mediabox
        new_height = float(box.top) - crop_pt
        page.mediabox.upper_right = (float(box.right), new_height)
        page.cropbox.upper_right = (float(box.right), new_height)
        writer.add_page(page)
    tmp = path.with_suffix(".tmp.pdf")
    with tmp.open("wb") as handle:
        writer.write(handle)
    tmp.replace(path)


def main() -> None:
    for stem, crop_px in TITLE_CROPS_PX.items():
        png_path = FIG_DIR / f"{stem}.png"
        if not png_path.exists():
            raise FileNotFoundError(png_path)
        with Image.open(png_path) as image:
            png_width_px = image.width
            png_height_px = image.height
        dpi_x, _dpi_y = image_dpi(png_path, 300)
        crop_pt = crop_svg(stem, crop_px, png_width_px)
        crop_pdf(stem, crop_pt)
        crop_raster(stem, crop_px, "png", dpi_x)
        for suffix in ("tiff", "tif"):
            raster_path = FIG_DIR / f"{stem}.{suffix}"
            if raster_path.exists():
                with Image.open(raster_path) as raster:
                    scaled_crop = round(crop_px * raster.height / png_height_px)
                crop_raster(stem, scaled_crop, suffix, 600)


if __name__ == "__main__":
    main()
