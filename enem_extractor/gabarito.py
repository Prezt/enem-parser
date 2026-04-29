from __future__ import annotations

import re
from pathlib import Path

import fitz  # PyMuPDF


_ANNULLED_RE = re.compile(r"^anulad[ao]$", re.IGNORECASE)
_LETTER_RE = re.compile(r"^[A-Ea-e]$")
_NUMBER_RE = re.compile(r"^\d{1,3}$")

# Tokens that are part of section headers, not Q-A data
_HEADER_TOKENS = frozenset(
    ["QUESTÃO", "QUESTAO", "GABARITO", "INGLÊS", "INGLES", "ESPANHOL", "ESPAÑOL"]
)


def _normalise(tok: str) -> str:
    return tok.upper().strip()


def _parse_section(
    tokens: list[str], bilingual: bool, answers: dict[int, str]
) -> None:
    """
    Walk a flat token list extracted from one gabarito table section.

    Expected sequence (single-language):  <num> <letter> <num> <letter> ...
    Bilingual section (Q1-5, day 1):      <num> <EN-letter> <ES-letter> ...

    For bilingual questions we record the English answer and discard the Spanish one.
    """
    i = 0
    while i < len(tokens):
        tok = tokens[i]

        if not _NUMBER_RE.match(tok):
            i += 1
            continue

        num = int(tok)
        if not (1 <= num <= 180):
            i += 1
            continue

        if i + 1 >= len(tokens):
            break

        answer_tok = tokens[i + 1]

        if _LETTER_RE.match(answer_tok):
            answers[num] = answer_tok.lower()
            i += 2
            # Bilingual block: a second letter immediately follows (Spanish answer) — skip it
            if bilingual and num <= 5 and i < len(tokens) and _LETTER_RE.match(tokens[i]):
                i += 1
            continue

        if _ANNULLED_RE.match(answer_tok):
            answers[num] = "annulled"
            i += 2
            continue

        i += 1


def parse_gabarito(pdf_path: str | Path) -> dict[int, str]:
    """
    Parse an ENEM answer-key PDF and return {question_number: answer_letter}.

    Answers are lowercase. Annulled questions map to "annulled".

    Handles:
    - Single-digit question numbers (1–9).
    - Bilingual tables (INGLÊS / ESPANHOL) — English answer is kept.
    - Multi-page gabariots (tokens are processed page by page but section state
      is carried across pages within the same section header).
    """
    answers: dict[int, str] = {}

    with fitz.open(str(pdf_path)) as doc:
        full_text = "\n".join(page.get_text() for page in doc)

    # Tokenise on whitespace; preserve only relevant tokens for each section.
    # We walk the full text looking for QUESTÃO … GABARITO section openers and
    # collect Q-A pairs until the next section opener or end of text.
    raw_tokens = full_text.split()

    section_tokens: list[str] = []
    bilingual = False
    in_section = False

    i = 0
    while i < len(raw_tokens):
        norm = _normalise(raw_tokens[i])

        # Detect section header: QUESTÃO followed by GABARITO
        if norm in ("QUESTÃO", "QUESTAO"):
            if i + 1 < len(raw_tokens) and _normalise(raw_tokens[i + 1]) == "GABARITO":
                # Flush previous section
                if in_section and section_tokens:
                    _parse_section(section_tokens, bilingual, answers)

                section_tokens = []
                bilingual = False
                in_section = True
                i += 2  # skip QUESTÃO + GABARITO

                # Check for bilingual sub-header (INGLÊS … ESPANHOL)
                if i < len(raw_tokens) and _normalise(raw_tokens[i]) in ("INGLÊS", "INGLES"):
                    bilingual = True
                    i += 1  # skip INGLÊS
                    if i < len(raw_tokens) and _normalise(raw_tokens[i]) in ("ESPANHOL", "ESPAÑOL"):
                        i += 1  # skip ESPANHOL
                continue

        if in_section:
            # Collect only tokens that are numbers, single letters, or "ANULADA"
            if _NUMBER_RE.match(raw_tokens[i]) or _LETTER_RE.match(raw_tokens[i]) or _ANNULLED_RE.match(raw_tokens[i]):
                section_tokens.append(raw_tokens[i])

        i += 1

    # Flush last section
    if in_section and section_tokens:
        _parse_section(section_tokens, bilingual, answers)

    return answers
