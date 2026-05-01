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
- This question has exactly {image_count} image(s). Insert one bracket placeholder per image \
at the exact position where the image appears in the original text:
  - If the image is inside a reference text (Texto I, Texto II, etc.), insert the placeholder \
inside that context's `text` field.
  - If the image appears in the question stem itself, insert it in the question's `text` field.
  Use the most descriptive format that fits:
    [Figura: short description]
    [Gráfico - short description]
    [Infográfico - short description]
    [Esquema - short description]
  Leave all `images` arrays as [] — the pipeline fills in the actual file paths."""
    elif has_image_marker:
        image_instruction = """
- This question contains one or more figures. For each figure, insert a bracket placeholder \
where it appears in the original:
  - Inside the matching context's `text` if the figure is part of a reference text.
  - In the question's `text` if the figure is in the question stem itself.
    [Figura: short description]  /  [Gráfico - short description]  /  [Esquema - short description]
  Leave all `images` arrays as []."""
    else:
        image_instruction = ""

    return f"""You are a strict JSON extractor for Brazilian ENEM exam questions{context_note}.

Extract the question below and return ONLY a valid JSON object — no explanation, no markdown, no preamble.

The object must have exactly two keys: "questions" and "contexts".

{{
  "questions": [
    {{
      "number": <integer>,
      "text": <string — question stem in Portuguese, WITHOUT the reference texts>,
      "alternatives": {{
        "a": <string>,
        "b": <string>,
        "c": <string>,
        "d": <string>,
        "e": <string>
      }},
      "images": [],
      "tags": [<2 to 4 tags from the approved taxonomy below>],
      "difficulty": <integer 1–10: 1–2=very easy (>80% pass rate), 3–4=easy, 5–6=medium, 7–8=hard, 9–10=very hard (<20% pass rate). Default 5 when uncertain.>,
      "year": {year if year is not None else "null"},
      "test": "ENEM",
      "area": {f'"{area}"' if area else "null"},
      "answer": null
    }}
  ],
  "contexts": {{}}
}}

Rules:
- Keep original Portuguese wording verbatim. Do not paraphrase.
- Fix obvious text-extraction artifacts (dropped or swapped letters, e.g. "a eta" → "a reta", \
"dos nos" → "dos anos", "dessa ndústria" → "dessa indústria"). Keep corrections minimal and \
faithful to the original meaning.
- alternatives must have exactly keys a, b, c, d, e.
- Strip the letter prefix from each alternative (write the text after "A ", not "A texto").
- If an alternative is a bare number with a trailing period (e.g. "7.", "8."), strip the period.
- Preserve fractions exactly as written (e.g. "35/64", "1/2"). Never reduce or split a fraction \
across lines — the numerator and denominator must stay together as "numerator/denominator".
- Do not omit any sentence from the question body. Every sentence in the original is needed \
to solve the question — if setup rules or constraints appear before the question stem, keep them.
- If the question has a reference text (Texto I, Texto II, or a named excerpt):
  - Set `contextId` on the question object to a short key like "enem_{year}_{area}_ctx1". \
For two texts use `contextIds: ["enem_{year}_{area}_ctx1", "enem_{year}_{area}_ctx2"]` and omit `contextId`.
  - Add a matching entry in the top-level `contexts` object:
    "enem_{year}_{area}_ctx1": {{
      "title": <string or null — label shown before the text, e.g. "Texto I", "Texto: Poema">,
      "subtitle": <string or null — secondary label if present, e.g. a poem/song title>,
      "text": <full verbatim text of the excerpt — every word, line break preserved>,
      "images": [],
      "reference": <string or null — author, work, publication, year of source if stated>
    }}
  - The `text` field in the question stem must NOT repeat the reference text; it should only \
contain the question itself (starting after the excerpt).
- tags: choose 2–4 tags exclusively from this approved taxonomy (do not invent new ones):
  Matemática: aritmética e números, álgebra, funções, geometria plana, geometria espacial, geometria analítica, trigonometria, estatística, probabilidade, análise combinatória, matemática financeira, sequências e progressões
  Física: mecânica newtoniana, termologia, óptica, ondulatória e acústica, eletricidade e magnetismo, física moderna, física nuclear e radioatividade, gravitação e astronomia
  Química: química geral e inorgânica, química orgânica, estequiometria, termoquímica e cinética, eletroquímica, equilíbrio químico, soluções e solubilidade
  Biologia: citologia, genética e hereditariedade, evolução, ecologia, fisiologia humana, botânica, zoologia, microbiologia e imunologia, biotecnologia
  História: história do brasil colonial, história do brasil imperial, história do brasil república, história antiga e medieval, história moderna, história contemporânea, geopolítica e relações internacionais, história da cultura e arte
  Geografia: geografia física e geologia, geografia humana e urbana, meio ambiente e sustentabilidade, geopolítica e território, cartografia, climatologia
  Filosofia e Sociologia: filosofia antiga e medieval, filosofia moderna e contemporânea, ética e política, sociologia e estrutura social, cultura e identidade, direitos humanos e cidadania, epistemologia e lógica
  Linguagens: interpretação de texto, gêneros textuais, linguística e variação linguística, literatura brasileira, literatura mundial, artes visuais, música e dança, comunicação e mídia, língua espanhola, língua inglesa
- language: set "en" or "es" only for foreign-language questions; otherwise omit.{language_rule}
- Preserve structured data as markdown lists. When the question body contains a sequence of \
data items (bullet points, numbered items, or inline lists separated by semicolons that each \
represent a distinct data point), format each item on its own line as "  - item". For example, \
inline data like "outubro/2020: 250 mm; novembro/2020: 150 mm" must become:\n\
  - outubro/2020: 250 mm;\n  - novembro/2020: 150 mm;\nDo this for any list of ≥3 structured items.
- Similarly, preserve bullet-point lists from the original text (e.g. lines starting with "•") \
as "  - item" in the output.
- images must always be present as [] (empty array).
- Ensure output is valid JSON (double-quote all strings, no trailing commas).{image_instruction}

QUESTION TEXT:
{chunk}
"""
