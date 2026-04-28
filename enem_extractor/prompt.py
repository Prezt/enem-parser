from __future__ import annotations


def build_extraction_prompt(
    chunk: str,
    has_image_marker: bool = False,
    year: int | None = None,
    area: str | None = None,
) -> str:
    """
    Build the LLM prompt for extracting a single question chunk.

    Args:
        chunk: Raw question text (one question block).
        has_image_marker: If True, ask the model to insert bracket image placeholders.
        year: Exam year for context.
        area: Exam area (math, nature, linguagens, humanas).
    """
    context_parts = []
    if year:
        context_parts.append(f"year={year}")
    if area:
        context_parts.append(f"area={area}")
    context_note = f" ({', '.join(context_parts)})" if context_parts else ""

    image_instruction = ""
    if has_image_marker:
        image_instruction = """
- The question contains one or more figures (charts, diagrams, images).
  For each figure, insert a bracket placeholder in `text` at the exact position it appears.
  Use the most descriptive format that fits:
    [Figura: short description]
    [Gráfico - short description]
    [Infográfico - short description]
    [Esquema - short description]
  The number of placeholders in `text` must match the number of entries in `images`.
  Leave `images` as an empty array [] — the pipeline will fill in the actual file paths."""

    return f"""You are a strict JSON extractor for Brazilian ENEM exam questions{context_note}.

Extract the question below and return ONLY a valid JSON array — no explanation, no markdown, no preamble.

Each object in the array must follow this schema exactly:

[
  {{
    "number": <integer — question number from the exam booklet>,
    "text": <string — full question stem in Portuguese, verbatim>,
    "alternatives": {{
      "a": <string>,
      "b": <string>,
      "c": <string>,
      "d": <string>,
      "e": <string>
    }},
    "answer": null,
    "tags": [<2 to 4 short topic tags in Portuguese, e.g. "física", "cinemática">],
    "year": {year if year is not None else "null"},
    "test": "ENEM",
    "area": {f'"{area}"' if area else "null"},
    "images": [],
    "contextId": null,
    "contextIds": null,
    "language": null
  }}
]

Rules:
- Keep original Portuguese wording verbatim. Do not paraphrase.
- alternatives must have exactly keys a, b, c, d, e.
- Strip the letter prefix from each alternative (write the text after "A ", not "A texto").
- If the question has a reference text (Texto I, Texto II, or a named excerpt), extract it into
  `contextId` as a short key string like "enem_{year}_{area}_ctx1". If there are two texts, use
  `contextIds: ["enem_{year}_{area}_ctx1", "enem_{year}_{area}_ctx2"]` and set `contextId` to null.
- language: set to "en" or "es" only for foreign-language questions (English/Spanish); otherwise null.
- Ignore page headers, watermarks, page numbers, and unrelated footer text.
- Skip redação (essay) prompts entirely.
- images must always be present as an array (empty [] when no images).
- Ensure output is valid JSON (double-quote all strings, no trailing commas).{image_instruction}

QUESTION TEXT:
{chunk}
"""
