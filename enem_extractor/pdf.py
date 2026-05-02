from __future__ import annotations

import subprocess
from pathlib import Path

import fitz  # PyMuPDF


def rasterize_pages(
    pdf_path: str | Path,
    out_dir: str | Path,
    dpi: int = 200,
    first_page: int | None = None,
    last_page: int | None = None,
    prefix: str = "page",
) -> list[Path]:
    """Rasterize PDF pages to PNG via pdftoppm. Returns sorted list of PNG paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = ["pdftoppm", "-png", "-r", str(dpi)]
    if first_page is not None:
        cmd += ["-f", str(first_page)]
    if last_page is not None:
        cmd += ["-l", str(last_page)]

    cmd += [str(pdf_path), str(out_dir / prefix)]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"pdftoppm failed: {result.stderr}")

    return sorted(out_dir.glob(f"{prefix}-*.png"))


def get_pdf_dimensions(pdf_path: str | Path) -> tuple[float, float]:
    """Returns (width_pts, height_pts) of the first page."""
    with fitz.open(str(pdf_path)) as doc:
        page = doc[0]
        rect = page.rect
        return rect.width, rect.height


def compute_scale(
    pdf_path: str | Path,
    png_path: str | Path,
) -> tuple[float, float]:
    """Compute (SCALE_X, SCALE_Y) from PDF points to PNG pixels (per-axis, not uniform)."""
    from PIL import Image

    pdf_w, pdf_h = get_pdf_dimensions(pdf_path)

    with Image.open(str(png_path)) as img:
        png_w, png_h = img.size

    scale_x = png_w / pdf_w
    scale_y = png_h / pdf_h
    return scale_x, scale_y
