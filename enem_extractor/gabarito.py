from __future__ import annotations

import re
from pathlib import Path

import fitz  # PyMuPDF


_ANNULLED_PATTERN = re.compile(r"anulad[ao]", re.IGNORECASE)

# Matches lines like: "91 B" or "91. B" or "91 - B" (also with ANULADA)
_ANSWER_LINE = re.compile(
    r"(\d{2,3})\s*[.\-]?\s*([A-Ea-e]|anulad[ao])",
    re.IGNORECASE,
)


def parse_gabarito(pdf_path: str | Path) -> dict[int, str]:
    """
    Parse an ENEM answer key PDF and return a mapping of question number -> answer letter.

    Answer letters are lowercase. Annulled questions map to "annulled".
    """
    answers: dict[int, str] = {}

    with fitz.open(str(pdf_path)) as doc:
        for page in doc:
            text = page.get_text()
            for match in _ANSWER_LINE.finditer(text):
                number = int(match.group(1))
                raw_answer = match.group(2)
                if _ANNULLED_PATTERN.match(raw_answer):
                    answers[number] = "annulled"
                else:
                    answers[number] = raw_answer.lower()

    return answers
