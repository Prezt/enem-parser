from enem_parser import parse_all


def build_prompt(chunk):

    text = parse_all(chunk)

    return f"""
You are a strict JSON generator.

Extract ENEM questions from the text.

Return ONLY valid JSON.
No explanations.
No markdown.
No extra text.

Schema:
[
  {{
    "number": integer,
    "text": string,
    "alternatives": {{
      "a": string,
      "b": string,
      "c": string,
      "d": string,
      "e": string
    }}
  }}
]

Rules:
- Keep original wording
- Ignore unrelated text
- Do not invent anything
- Ensure valid JSON

TEXT:
{text}
"""
