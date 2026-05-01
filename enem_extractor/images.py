from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
from PIL import Image


@dataclass
class Bbox:
    x0: float
    y0: float
    x1: float
    y1: float

    def to_fitz_rect(self) -> fitz.Rect:
        return fitz.Rect(self.x0, self.y0, self.x1, self.y1)

    @classmethod
    def from_fitz_rect(cls, r: fitz.Rect) -> "Bbox":
        return cls(x0=r.x0, y0=r.y0, x1=r.x1, y1=r.y1)


@dataclass
class ImageRect:
    bbox_pts: Bbox
    xref: int


_MIN_SIZE_PTS = 50.0


def find_embedded_images(pdf_path: str | Path, page_idx: int) -> list[ImageRect]:
    """Return embedded raster images on a page, filtering out decorative ones (<50x50 pts)."""
    results: list[ImageRect] = []

    with fitz.open(str(pdf_path)) as doc:
        page = doc[page_idx]
        for img in page.get_images(full=True):
            xref = img[0]
            rects = page.get_image_rects(xref)
            for rect in rects:
                w = rect.width
                h = rect.height
                if w >= _MIN_SIZE_PTS and h >= _MIN_SIZE_PTS:
                    results.append(ImageRect(bbox_pts=Bbox.from_fitz_rect(rect), xref=xref))

    return results


def find_vector_figure_bbox(
    pdf_path: str | Path,
    page_idx: int,
    after_text: str,
    before_text: str,
) -> Optional[Bbox]:
    """
    Heuristic bbox for vector graphics (charts, diagrams) that don't appear in get_images().

    Finds the bbox of the last span containing `after_text`, then the bbox of the first span
    containing `before_text` that appears below it. Returns the vertical gap between them
    spanning the full column width.
    """
    with fitz.open(str(pdf_path)) as doc:
        page = doc[page_idx]
        w = page.rect.width
        text_dict = page.get_text("dict")

    after_pattern = re.compile(re.escape(after_text), re.IGNORECASE)
    before_pattern = re.compile(re.escape(before_text), re.IGNORECASE)

    after_bottom: Optional[float] = None
    before_top: Optional[float] = None

    for block in text_dict["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                span_text = span["text"]
                bbox = span["bbox"]
                if after_pattern.search(span_text):
                    after_bottom = bbox[3]  # y1 of this span
                if before_pattern.search(span_text) and after_bottom is not None:
                    candidate_top = bbox[1]
                    if candidate_top > after_bottom:
                        before_top = candidate_top
                        break
            if before_top is not None:
                break
        if before_top is not None:
            break

    if after_bottom is None or before_top is None:
        return None

    return Bbox(x0=0.0, y0=after_bottom, x1=w, y1=before_top)


def expand_bbox_to_caption(bbox: Bbox, text_dict: dict) -> Bbox:
    """
    Expand bbox downward to include figure captions (author attribution patterns),
    stopping before the next question header or alternatives marker.
    """
    caption_pattern = re.compile(r"(Disponível em:|VIEIRA|Acesso em:|Fonte:)", re.IGNORECASE)
    stop_pattern = re.compile(r"(QUEST[ÃA]O\s+\d+|^[A-E]\s)", re.IGNORECASE | re.MULTILINE)

    new_y1 = bbox.y1

    for block in text_dict["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                span_bbox = span["bbox"]
                span_top = span_bbox[1]
                span_bottom = span_bbox[3]
                span_text = span["text"]

                if span_top < bbox.y1:
                    continue
                if stop_pattern.search(span_text):
                    return Bbox(x0=bbox.x0, y0=bbox.y0, x1=bbox.x1, y1=new_y1)
                if caption_pattern.search(span_text):
                    new_y1 = max(new_y1, span_bottom)

    return Bbox(x0=bbox.x0, y0=bbox.y0, x1=bbox.x1, y1=new_y1)


def find_figure_regions_by_gaps(
    pdf_path: str | Path,
    page_idx: int,
    y_top: float,
    y_bottom: float,
    min_gap_pts: float = 35.0,
) -> list[Bbox]:
    """
    Detect figure regions by finding vertical gaps between text blocks.
    Works for both raster and vector figures since both create text-free zones.
    Returns a list of Bbox for each gap found within [y_top, y_bottom].
    """
    with fitz.open(str(pdf_path)) as doc:
        page = doc[page_idx]
        page_w = page.rect.width
        text_blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, type)

    # Only text blocks (type 0) inside the question region
    text_ys = []
    for b in text_blocks:
        if b[6] != 0:
            continue
        by0, by1 = b[1], b[3]
        # block must overlap the question region
        if by1 < y_top or by0 > y_bottom:
            continue
        text_ys.append((max(by0, y_top), min(by1, y_bottom)))

    if not text_ys:
        return []

    text_ys.sort()

    gaps: list[Bbox] = []
    prev_end = y_top
    for (start, end) in text_ys:
        if start - prev_end >= min_gap_pts:
            gaps.append(Bbox(x0=0.0, y0=prev_end, x1=page_w, y1=start))
        prev_end = max(prev_end, end)

    return gaps


def crop_figure(
    png_path: str | Path,
    bbox_pts: Bbox,
    scale_x: float,
    scale_y: float,
    out_path: str | Path,
    padding_pts: float = 5.0,
) -> Path:
    """Crop a figure from a rasterized PNG using per-axis scaling. Saves with optimize=True."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    px_x0 = max(0, int((bbox_pts.x0 - padding_pts) * scale_x))
    px_y0 = max(0, int((bbox_pts.y0 - padding_pts) * scale_y))
    px_x1 = int((bbox_pts.x1 + padding_pts) * scale_x)
    px_y1 = int((bbox_pts.y1 + padding_pts) * scale_y)

    with Image.open(str(png_path)) as img:
        px_x1 = min(px_x1, img.width)
        px_y1 = min(px_y1, img.height)
        cropped = img.crop((px_x0, px_y0, px_x1, px_y1))
        cropped.save(str(out_path), optimize=True)

    return out_path
