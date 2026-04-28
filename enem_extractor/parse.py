from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF


@dataclass
class Block:
    number: int
    raw_text: str
    start_pos: int
    end_pos: int


def _extract_column_text(page: fitz.Page, clip: fitz.Rect) -> str:
    """Extract text from a column bbox, sorted by (top, x0) within spans."""
    blocks = page.get_text("dict", clip=clip)["blocks"]

    spans: list[tuple[float, float, str]] = []
    for block in blocks:
        if block.get("type") != 0:  # 0 = text block
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                top = round(span["origin"][1], 1)
                x0 = span["bbox"][0]
                spans.append((top, x0, span["text"]))

    spans.sort(key=lambda s: (s[0], s[1]))

    # Group into lines by proximity of top coordinate
    lines: list[str] = []
    current_line_parts: list[str] = []
    current_top: Optional[float] = None

    for top, _x0, text in spans:
        if current_top is None:
            current_top = top
        if abs(top - current_top) > 5:
            if current_line_parts:
                lines.append(" ".join(current_line_parts))
            current_line_parts = []
            current_top = top
        current_line_parts.append(text)

    if current_line_parts:
        lines.append(" ".join(current_line_parts))

    return "\n".join(lines)


def extract_text(
    pdf_path: str | Path,
    page_range: tuple[int, int] | None = None,
) -> str:
    """PyMuPDF-based two-column text extraction."""
    full_text = ""

    with fitz.open(str(pdf_path)) as doc:
        pages = range(*page_range) if page_range else range(len(doc))

        for i in pages:
            page = doc[i]
            w = page.rect.width
            h = page.rect.height

            left_clip = fitz.Rect(0, 0, w * 0.48, h)
            right_clip = fitz.Rect(w * 0.52, 0, w, h)

            left_text = _extract_column_text(page, left_clip)
            right_text = _extract_column_text(page, right_clip)

            page_text = (left_text + "\n" + right_text).strip()
            full_text += "\n" + page_text

    return full_text


def normalize_text(text: str) -> str:
    """Clean up common PyMuPDF / PDF artifacts in ENEM text."""
    # Fix QUESTAO -> QUESTÃO
    text = re.sub(r"QUESTAO", "QUESTÃO", text, flags=re.IGNORECASE)

    # Remove ENEM watermark patterns like EM2024ENEM2024...
    text = re.sub(r"(EM\d{4})+", "", text)

    # Remove very long uppercase/digit sequences (encoded garbage)
    text = re.sub(r"[A-Z0-9]{20,}", "", text)

    # Fix soft hyphen + newline ligature joins (hyphenated words split across lines)
    text = re.sub(r"-\n(\S)", r"\1", text)
    text = re.sub(r"\xad", "", text)  # soft hyphen U+00AD

    # Collapse multiple blank lines to one
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text


# Matches QUESTÃO N at start of a segment — must be followed by whitespace or end-of-line,
# not by random digits inside quoted text.
_QUESTION_HEADER = re.compile(
    r"(QUEST[ÃA]O\s+(\d+)(?:\t| *\n| *$))",
    re.IGNORECASE | re.MULTILINE,
)


def split_by_question(text: str) -> list[Block]:
    """Split full exam text into question blocks, preserving position info."""
    matches = list(_QUESTION_HEADER.finditer(text))
    if not matches:
        return []

    blocks: list[Block] = []
    for idx, match in enumerate(matches):
        number = int(match.group(2))
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        raw_text = text[start:end].strip()
        blocks.append(Block(number=number, raw_text=raw_text, start_pos=start, end_pos=end))

    return blocks
