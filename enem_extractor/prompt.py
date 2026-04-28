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
        chunk: Raw question text (one question block, possibly multi-question for edge cases).
        has_image_marker: If True, ask the model to insert an image placeholder.
        year: Exam year for context.
        area: Exam area (linguagens, humanas, natureza, matematica).
    """
    context_parts = []
    if year:
        context_parts.append(f"year={year}")
    if area:
        context_parts.append(f"area={area}")
    context_note = f" ({', '.join(context_parts)})" if context_parts else ""

    image_instruction = ""
    if has_image_marker:
        image_instruction = (
            "\n- The question contains a figure. "
            "Insert a placeholder `[Imagem: q{number}_<short_descriptor>.png]` "
            "at the exact position in `text` where the figure appears."
        )

    return f"""You are a strict JSON extractor for Brazilian ENEM exam questions{context_note}.

Extract the question below and return ONLY valid JSON — no explanation, no markdown, no preamble.

Output a JSON array with one object per question found. Each object must follow this schema exactly:

[
  {{
    "number": <integer>,
    "text": <string, full question body in Portuguese>,
    "alternatives": {{
      "a": <string>,
      "b": <string>,
      "c": <string>,
      "d": <string>,
      "e": <string>
    }},
    "images": [<list of image placeholder strings, may be empty>],
    "tags": [<2 to 4 short topic strings inferred from content, in Portuguese>],
    "answer": null,
    "year": {year if year is not None else "null"},
    "test": "ENEM",
    "area": {f'"{area}"' if area else "null"},
    "language": null
  }}
]

Rules:
- Keep original Portuguese wording verbatim.
- Do not invent or paraphrase anything.
- alternatives must have exactly keys a, b, c, d, e — no more, no less.
- Strip alternative letter prefixes (e.g. write the text after "A ", not "A texto").
- Ignore headers, page numbers, watermarks, and unrelated footer text.
- If redação (essay prompt) is found, skip it entirely.
- Ensure output is valid JSON (double-quote all strings, no trailing commas).{image_instruction}

QUESTION TEXT:
{chunk}
"""
