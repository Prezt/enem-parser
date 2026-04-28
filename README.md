# enem-extractor

Extracts structured JSON + cropped figures from ENEM PDF exams using a local LLM via Ollama.

## Install

### System dependencies

```bash
# macOS
brew install poppler

# Ubuntu / Debian
sudo apt install poppler-utils
```

### Ollama model

```bash
ollama pull qwen2.5:14b-instruct
# fallback
ollama pull qwen2.5:7b-instruct
```

### Python package

```bash
pip install -e .
```

## Usage

```bash
# Extract with gabarito (answer key)
enem-extract extract matematica_2024_enem.pdf gabarito_2024.pdf \
  --area matematica \
  --year 2024 \
  --pages 1-45 \
  --output output/matematica_2024

# Extract without answer key, debug mode
enem-extract extract linguagens_2024.pdf \
  --area linguagens \
  --year 2024 \
  --debug

# Validate output JSON
enem-extract validate output/matematica_2024/questions_matematica_2024.json

# Print questions that need manual review
enem-extract review output/matematica_2024/questions_matematica_2024.json
```

Output directory structure:

```
output/
├── questions_matematica_2024.json   # main output
├── figures/                         # cropped PNGs per question
│   ├── q091_fig1.png
│   └── ...
├── pages/                           # rasterized page PNGs (200 DPI)
├── .cache/                          # per-question cache for resumability
└── debug/                           # only with --debug
    ├── chunks/
    ├── prompts/
    └── responses/
```

## Output schema

```json
{
  "metadata": { "year": 2024, "area": "matematica", "model": "qwen2.5:14b-instruct", ... },
  "questions": [
    {
      "number": 91,
      "text": "...",
      "alternatives": { "a": "...", "b": "...", "c": "...", "d": "...", "e": "..." },
      "images": ["figures/q091_fig1.png"],
      "tags": ["física", "cinemática"],
      "answer": "b",
      "year": 2024,
      "test": "ENEM",
      "area": "matematica"
    }
  ],
  "figures": ["figures/q091_fig1.png"],
  "warnings": []
}
```

## When LLM extraction fails

Common failure modes and how to handle them:

**Long Texto I + Texto II questions** — These can exceed the model's context window or produce truncated JSON. Increase `num_ctx` by passing `--model qwen2.5:14b-instruct` (already at 16k context) or split the exam into smaller `--pages` ranges.

**Questions where alternatives are images** — The LLM cannot see images embedded in alternatives. These questions will have empty alternative strings. After extraction, use `enem-extract review output.json` to list them and fill them in manually.

**Malformed JSON from model** — The `extract_json` function tries several fallbacks (strip fences, extract array, extract object). If all fail, the question is logged as a warning with `_review_needed: True` and skipped. Re-run after the partial cache is saved — only failed questions will be retried.

**Model not available** — If `qwen2.5:14b-instruct` isn't pulled, pass `--model qwen2.5:7b-instruct` as fallback. Results will be less accurate on long questions.

**Resuming after partial failure** — Re-run the exact same command. Already-extracted questions are loaded from `.cache/` and skipped.

## Development

```bash
make install   # install package + dev tools
make test      # run tests
make lint      # ruff check
make clean     # remove output/debug/cache artifacts
```
