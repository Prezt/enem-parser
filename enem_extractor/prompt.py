from __future__ import annotations


def build_extraction_prompt(
    chunk: str,
    has_image_marker: bool = False,
    image_count: int = 0,
    year: int | None = None,
    area: str | None = None,
    language_hint: str | None = None,
) -> str:
    context_parts = []
    if year:
        context_parts.append(f"year={year}")
    if area:
        context_parts.append(f"area={area}")
    if language_hint:
        context_parts.append(f"language variant={language_hint}")
    context_note = f" ({', '.join(context_parts)})" if context_parts else ""

    language_rule = ""
    if language_hint:
        lang_label = "English" if language_hint == "en" else "Spanish"
        language_rule = f'\n- This is the {lang_label} language variant. Set `"language": "{language_hint}"` on the output object.'

    if has_image_marker and image_count > 0:
        image_instruction = f"""
- This question has exactly {image_count} image(s). You MUST insert exactly {image_count} bracket \
placeholder(s) in `text`, one per image, at the position each figure appears in the question.
  Use the most descriptive format that fits:
    [Figura: short description]
    [Gráfico - short description]
    [Infográfico - short description]
    [Esquema - short description]
  The number of placeholders in `text` must equal the length of the `images` array ({image_count}).
  Leave `images` as an empty array [] — the pipeline fills in the actual file paths."""
    elif has_image_marker:
        image_instruction = """
- This question contains one or more figures. Insert one bracket placeholder per figure in `text`:
    [Figura: short description]  /  [Gráfico - short description]  /  [Esquema - short description]
  Leave `images` as []."""
    else:
        image_instruction = ""

    return f"""You are a strict JSON extractor for Brazilian ENEM exam questions{context_note}.

Extract the question below and return ONLY a valid JSON array — no explanation, no markdown, no preamble.

Each object must follow this schema with fields in this exact order:

[
  {{
    "number": <integer>,
    "text": <string — question stem in Portuguese>,
    "alternatives": {{
      "a": <string>,
      "b": <string>,
      "c": <string>,
      "d": <string>,
      "e": <string>
    }},
    "images": [],
    "tags": [<2 to 4 short topic tags in Portuguese>],
    "year": {year if year is not None else "null"},
    "test": "ENEM",
    "area": {f'"{area}"' if area else "null"},
    "answer": null
  }}
]

Rules:
- Keep original Portuguese wording verbatim. Do not paraphrase.
- Fix obvious text-extraction artifacts (dropped or swapped letters, e.g. "a eta" → "a reta", \
"dos nos" → "dos anos", "dessa ndústria" → "dessa indústria"). Keep corrections minimal and \
faithful to the original meaning.
- alternatives must have exactly keys a, b, c, d, e.
- Strip the letter prefix from each alternative (write the text after "A ", not "A texto").
- If an alternative is a bare number with a trailing period (e.g. "7.", "8."), strip the period.
- If the question has a reference text (Texto I, Texto II, or a named excerpt), set `contextId` \
to a short key like "enem_{year}_{area}_ctx1". For two texts use \
`contextIds: ["enem_{year}_{area}_ctx1", "enem_{year}_{area}_ctx2"]` and omit `contextId`.
- language: set "en" or "es" only for foreign-language questions; otherwise omit.{language_rule}
- images must always be present as [] (empty array).
- Ensure output is valid JSON (double-quote all strings, no trailing commas).{image_instruction}

QUESTION TEXT:
{chunk}
"""
