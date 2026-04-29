from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Optional

from .gabarito import parse_gabarito
from .images import find_embedded_images, find_vector_figure_bbox, crop_figure, expand_bbox_to_caption
from .llm import OllamaClient, extract_json, resolve_model
from .parse import extract_text, normalize_text, split_by_question, Block
from .pdf import rasterize_pages, get_pdf_dimensions, compute_scale
from .prompt import build_extraction_prompt
from .schema import Context, ExamMetadata, ExamResult, Question

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "qwen2.5:14b-instruct"

# ENEM question number ranges per day/area
# Day 1: Linguagens (1-45), Humanas (46-90)
# Day 2: Natureza (91-135), Matemática (136-180)
DAY_AREAS: dict[int, list[tuple[str, range]]] = {
    1: [("linguagens", range(1, 46)), ("humanas", range(46, 91))],
    2: [("nature", range(91, 136)), ("math", range(136, 181))],
}


def area_for_question(q_num: int, day: int) -> str | None:
    for area, r in DAY_AREAS.get(day, []):
        if q_num in r:
            return area
    return None
_FALLBACK_MODEL = "qwen2.5:7b-instruct"


def _pdf_hash(pdf_path: Path) -> str:
    h = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _cache_key(pdf_hash: str, question_number: int, occurrence: int = 0) -> str:
    suffix = f"_v{occurrence}" if occurrence > 0 else ""
    return f"{pdf_hash}_q{question_number:03d}{suffix}"


_ES_PATTERN = re.compile(r"[¿¡]|\b(que|está|una|por|del|las|los|se|en|con|como)\b", re.IGNORECASE)
_EN_PATTERN = re.compile(r"\b(the|is|are|was|were|have|has|this|that|with|from|which|their)\b", re.IGNORECASE)


def _detect_language(text: str) -> str | None:
    """Heuristic language detection for Q1-5 foreign language variants."""
    es_hits = len(_ES_PATTERN.findall(text))
    en_hits = len(_EN_PATTERN.findall(text))
    if es_hits > en_hits and es_hits >= 3:
        return "es"
    if en_hits > es_hits and en_hits >= 3:
        return "en"
    return None


def _load_cache(cache_dir: Path) -> dict[str, dict]:
    cache: dict[str, dict] = {}
    if not cache_dir.exists():
        return cache
    for f in cache_dir.glob("*.json"):
        try:
            cache[f.stem] = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return cache


def _save_cache(cache_dir: Path, key: str, data: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / f"{key}.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


_MARKER_RE = re.compile(
    r'\[(?:Figura|Gráfico|Infográfico|Esquema|Imagem)[^]]*\]',
    re.IGNORECASE,
)

_FIGURE_MENTION = re.compile(
    r"(conforme a figura|veja (o |a )?(figura|gráfico|imagem)|figura \d|observe (o |a )?)",
    re.IGNORECASE,
)


def _question_vertical_bounds(pdf_path: Path, page_idx: int, q_num: int, next_q_num: int | None) -> tuple[float, float]:
    """
    Return (y_top, y_bottom) in PDF points for question q_num on the given page.
    Searches for 'QUESTÃO {q_num}' and 'QUESTÃO {next_q_num}' text positions.
    Falls back to (0, page_height) if not found.
    """
    import fitz
    with fitz.open(str(pdf_path)) as doc:
        page = doc[page_idx]
        page_h = page.rect.height

        header = f"QUESTÃO {q_num}"
        hits = page.search_for(header)
        if not hits:
            hits = page.search_for(f"QUESTAO {q_num}")
        y_top = hits[0].y0 if hits else 0.0

        y_bottom = page_h
        if next_q_num is not None:
            next_header = f"QUESTÃO {next_q_num}"
            next_hits = page.search_for(next_header)
            if not next_hits:
                next_hits = page.search_for(f"QUESTAO {next_q_num}")
            if next_hits:
                y_bottom = next_hits[0].y0

    return y_top, y_bottom


def _images_for_block(pdf_path: Path, page_idx: int, y_top: float, y_bottom: float):
    """Return only embedded images whose bbox falls within [y_top, y_bottom]."""
    all_images = find_embedded_images(pdf_path, page_idx)
    return [
        img for img in all_images
        if img.bbox_pts.y1 > y_top and img.bbox_pts.y0 < y_bottom
    ]


def _detect_image_for_block(block: Block, images_in_block: list) -> bool:
    """Return True if the block has embedded images or mentions a figure."""
    if images_in_block:
        return True
    return bool(_FIGURE_MENTION.search(block.raw_text))


def _find_page_for_block(block: Block, page_texts: list[tuple[int, str]]) -> int:
    """Find which page index a block likely appears on (by matching question number)."""
    header = f"QUESTÃO {block.number}"
    for page_idx, page_text in page_texts:
        if header in page_text or f"QUESTAO {block.number}" in page_text.upper():
            return page_idx
    return 0  # fallback


def _process_block(
    block: Block,
    next_q_num: int | None,
    pdf_path: Path,
    png_paths: list[Path],
    page_texts: list[tuple[int, str]],
    scale_x: float,
    scale_y: float,
    figures_dir: Path,
    client: OllamaClient,
    model: str,
    gabarito: dict[int, str],
    year: Optional[int],
    area: Optional[str],
    debug_dir: Optional[Path],
    language_hint: Optional[str] = None,
) -> tuple[Optional[Question], dict, list[str]]:
    """Process one question block. Returns (Question | None, extracted_contexts, warnings)."""
    warnings: list[str] = []
    q_num = block.number

    page_idx = _find_page_for_block(block, page_texts)
    y_top, y_bottom = _question_vertical_bounds(pdf_path, page_idx, q_num, next_q_num)
    images_in_block = _images_for_block(pdf_path, page_idx, y_top, y_bottom)
    has_image = _detect_image_for_block(block, images_in_block)

    prompt = build_extraction_prompt(
        block.raw_text,
        has_image_marker=has_image,
        image_count=len(images_in_block),
        year=year,
        area=area,
        language_hint=language_hint,
    )

    if debug_dir:
        (debug_dir / "chunks").mkdir(parents=True, exist_ok=True)
        (debug_dir / "prompts").mkdir(parents=True, exist_ok=True)
        (debug_dir / "responses").mkdir(parents=True, exist_ok=True)
        (debug_dir / "chunks" / f"{q_num:03d}.txt").write_text(block.raw_text, encoding="utf-8")
        (debug_dir / "prompts" / f"{q_num:03d}.txt").write_text(prompt, encoding="utf-8")

    try:
        response = client.generate(prompt, model=model)
    except RuntimeError as exc:
        msg = f"Q{q_num}: LLM call failed: {exc}"
        warnings.append(msg)
        logger.warning("%s", msg)
        return None, {}, warnings

    if debug_dir:
        (debug_dir / "responses" / f"{q_num:03d}.txt").write_text(response, encoding="utf-8")

    extracted_contexts: dict = {}
    try:
        json_str = extract_json(response)
        parsed = json.loads(json_str)
        # Support both wrapper format {"questions": [...], "contexts": {...}} and bare array
        if isinstance(parsed, dict) and "questions" in parsed:
            data_list = parsed["questions"]
            extracted_contexts = parsed.get("contexts") or {}
        elif isinstance(parsed, list):
            data_list = parsed
        else:
            raise ValueError(f"Unexpected JSON shape: {type(parsed)}")
        if not data_list:
            raise ValueError("Empty questions list")
        data = data_list[0]
    except (ValueError, json.JSONDecodeError, KeyError) as exc:
        warnings.append(f"Q{q_num}: JSON parse failed: {exc}")
        return None, {}, warnings

    # Merge gabarito answer
    data["answer"] = gabarito.get(q_num)
    data.setdefault("year", year)
    data.setdefault("area", area)

    # Crop figures and assign them to where their placeholders landed
    # (either inside a context's text or in the question stem).
    cropped_figures: list[str] = []
    if has_image and page_idx < len(png_paths):
        png_path = png_paths[page_idx]
        # Sort images top-to-bottom so they match document reading order
        sorted_images = sorted(images_in_block, key=lambda img: img.bbox_pts.y0)

        # Build an ordered list of destinations matching each placeholder in document order:
        # contexts come before the question stem, so process them first.
        destinations: list[tuple[str, str | None]] = []  # ("ctx", ctx_id) | ("question", None)
        for ctx_id, ctx_data in extracted_contexts.items():
            for _ in _MARKER_RE.findall(ctx_data.get("text", "")):
                destinations.append(("ctx", ctx_id))
        for _ in _MARKER_RE.findall(data.get("text", "")):
            destinations.append(("question", None))

        ctx_images: dict[str, list[str]] = {}
        for i, (dest_type, ctx_id) in enumerate(destinations):
            if i >= len(sorted_images):
                break
            img_rect = sorted_images[i]
            year_tag = f"_{year}" if year else ""
            fig_name = f"q{q_num:03d}{year_tag}_fig{i+1}.png"
            fig_path = figures_dir / fig_name
            try:
                crop_figure(png_path, img_rect.bbox_pts, scale_x, scale_y, out_path=fig_path)
                logger.info("Q%d: cropped figure -> figuras/%s", q_num, fig_name)
                if dest_type == "ctx" and ctx_id is not None:
                    ctx_images.setdefault(ctx_id, []).append(f"figuras/{fig_name}")
                else:
                    cropped_figures.append(f"figuras/{fig_name}")
            except Exception as exc:
                warnings.append(f"Q{q_num}: figure crop failed: {exc}")

        # Write cropped paths back into the extracted context dicts
        for ctx_id, imgs in ctx_images.items():
            extracted_contexts[ctx_id]["images"] = imgs

    data["images"] = cropped_figures or data.get("images", [])

    # Validate placeholder ↔ image counts for question text and each context
    if cropped_figures:
        q_marker_count = len(_MARKER_RE.findall(data.get("text", "")))
        if q_marker_count != len(cropped_figures):
            warnings.append(
                f"Q{q_num}: question image/marker mismatch — {len(cropped_figures)} image(s) "
                f"but {q_marker_count} marker(s) in question text. Review needed."
            )
    for ctx_id, ctx_data in extracted_contexts.items():
        ctx_imgs = ctx_data.get("images", [])
        if ctx_imgs:
            ctx_marker_count = len(_MARKER_RE.findall(ctx_data.get("text", "")))
            if ctx_marker_count != len(ctx_imgs):
                warnings.append(
                    f"Q{q_num}: context {ctx_id} image/marker mismatch — {len(ctx_imgs)} "
                    f"image(s) but {ctx_marker_count} marker(s). Review needed."
                )

    try:
        question = Question(**data)
    except Exception as exc:
        warnings.append(f"Q{q_num}: Pydantic validation failed: {exc}")
        # Return a best-effort object with review flag
        try:
            question = Question(
                number=q_num,
                text=data.get("text", ""),
                alternatives=data.get("alternatives", {"a": "", "b": "", "c": "", "d": "", "e": ""}),
                answer=gabarito.get(q_num),
                year=year,
                area=area,
            )
            object.__setattr__(question, "_review_needed", True)
        except Exception:
            return None, {}, warnings

    logger.info("Q%d: OK", q_num) if not warnings else logger.warning("Q%d: %s", q_num, "; ".join(warnings))
    return question, extracted_contexts, warnings


def extract_exam(
    prova_pdf: str | Path,
    gabarito_pdf: str | Path | None = None,
    *,
    area: str | None = None,
    day: int | None = None,
    page_range: tuple[int, int] | None = None,
    output_dir: str | Path = "output",
    model: str = _DEFAULT_MODEL,
    refine_with_llm: bool = True,
    year: int | None = None,
    test_name: str = "ENEM",
    debug: bool = False,
    retry_failed: bool = False,
) -> ExamResult:
    prova_pdf = Path(prova_pdf)
    if gabarito_pdf and not str(gabarito_pdf).strip():
        gabarito_pdf = None
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figuras"
    cache_dir = output_dir / ".cache"
    debug_dir = output_dir / "debug" if debug else None

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)  # output_dir/figuras/

    # Rasterize pages
    png_dir = output_dir / "pages"
    first_page = page_range[0] if page_range else None
    last_page = page_range[1] if page_range else None
    logger.info("Rasterizing pages...")
    png_paths = rasterize_pages(prova_pdf, png_dir, dpi=200, first_page=first_page, last_page=last_page)

    # Compute per-axis scale using first PNG
    scale_x, scale_y = (1.0, 1.0)
    if png_paths:
        scale_x, scale_y = compute_scale(prova_pdf, png_paths[0])
        logger.info("Scale: X=%.4f Y=%.4f", scale_x, scale_y)

    # Extract text
    logger.info("Extracting text...")
    raw_text = extract_text(prova_pdf, page_range=page_range)
    clean_text = normalize_text(raw_text)

    # Split into blocks
    blocks = split_by_question(clean_text)
    logger.info("Found %d question blocks", len(blocks))

    if not blocks:
        logger.warning("No question blocks detected — check page_range and PDF structure")

    # Build page-level text index for page detection
    page_texts: list[tuple[int, str]] = []
    import fitz
    with fitz.open(str(prova_pdf)) as doc:
        pages = range(*page_range) if page_range else range(len(doc))
        for i in pages:
            page_texts.append((i, doc[i].get_text()))

    # Parse gabarito
    gabarito: dict[int, str] = {}
    if gabarito_pdf:
        logger.info("Parsing gabarito...")
        gabarito = parse_gabarito(gabarito_pdf)
        logger.info("Loaded %d answers from gabarito", len(gabarito))

    # Load cache
    pdf_hash = _pdf_hash(prova_pdf)
    cache = _load_cache(cache_dir)

    # LLM client — resolve model before starting
    model = resolve_model(model, _FALLBACK_MODEL)
    client = OllamaClient(options={"num_predict": 8192} if retry_failed else None)

    all_questions: list[Question] = []
    all_warnings: list[str] = []
    all_figures: list[str] = []
    all_contexts: dict[str, dict] = {}

    # Load or initialise the set of question numbers to (re-)process
    failed_file = output_dir / "failed_questions.json"
    failed_q_nums: set[int] = set()

    if retry_failed:
        if not failed_file.exists():
            logger.error("No failed_questions.json found in %s — nothing to retry", output_dir)
            return ExamResult(
                metadata=ExamMetadata(
                    year=year, test=test_name, area=area,
                    prova_pdf=str(prova_pdf),
                    gabarito_pdf=str(gabarito_pdf) if gabarito_pdf else None,
                    model=model, page_range=page_range,
                ),
                questions=[], figures=[], warnings=[],
            )
        retry_nums: set[int] = set(json.loads(failed_file.read_text(encoding="utf-8")))
        if not retry_nums:
            logger.info("failed_questions.json is empty — nothing to retry")
        else:
            logger.info("Retrying %d failed question(s): %s", len(retry_nums), sorted(retry_nums))
        failed_q_nums = set(retry_nums)  # will shrink as questions succeed

    # Track occurrences of each question number to handle EN/ES duplicates (Q1-5 day 1)
    q_num_occurrences: dict[int, int] = {}

    for idx, block in enumerate(blocks):
        next_q_num = blocks[idx + 1].number if idx + 1 < len(blocks) else None

        occurrence = q_num_occurrences.get(block.number, 0)
        q_num_occurrences[block.number] = occurrence + 1
        cache_key = _cache_key(pdf_hash, block.number, occurrence)

        # When retrying, skip questions that aren't in the failed list
        if retry_failed and block.number not in retry_nums:
            # Still load from cache so the output file stays complete
            if cache_key in cache:
                try:
                    cached = cache[cache_key]
                    q_area = area_for_question(block.number, day) if day else area
                    if q_area is not None:
                        cached["area"] = q_area
                    if year is not None:
                        cached["year"] = year
                    if gabarito.get(block.number):
                        cached["answer"] = gabarito[block.number]
                    all_questions.append(Question(**cached))
                except Exception:
                    pass
            continue

        # Detect language for foreign-language variants (day 1, Q1-5)
        is_foreign_q = day == 1 and block.number <= 5
        language_hint = _detect_language(block.raw_text) if is_foreign_q else None
        if is_foreign_q and occurrence > 0 and language_hint is None:
            # Second occurrence without clear detection — assume ES if first was EN, vice versa
            first_key = _cache_key(pdf_hash, block.number, 0)
            first_lang = cache.get(first_key, {}).get("language")
            language_hint = "es" if first_lang == "en" else "en"

        # Resume: use cached result if available (skip cache for questions being retried)
        if cache_key in cache and not (retry_failed and block.number in retry_nums):
            logger.info("Q%d: loaded from cache", block.number)
            try:
                cached = cache[cache_key]
                # Always apply current run's area/year/answer so stale cache entries are corrected
                q_area = area_for_question(block.number, day) if day else area
                if q_area is not None:
                    cached["area"] = q_area
                if year is not None:
                    cached["year"] = year
                if gabarito.get(block.number):
                    cached["answer"] = gabarito[block.number]
                all_questions.append(Question(**cached))
                failed_q_nums.discard(block.number)
                continue
            except Exception:
                logger.warning("Q%d: cache entry invalid, re-extracting", block.number)

        if not refine_with_llm:
            q_area = area_for_question(block.number, day) if day else area
            stub = Question(
                number=block.number,
                text=block.raw_text,
                alternatives={"a": "", "b": "", "c": "", "d": "", "e": ""},
                answer=gabarito.get(block.number),
                year=year,
                area=q_area,
            )
            all_questions.append(stub)
            continue

        q_area = area_for_question(block.number, day) if day else area
        question, q_contexts, warnings = _process_block(
            block=block,
            next_q_num=next_q_num,
            pdf_path=prova_pdf,
            png_paths=png_paths,
            page_texts=page_texts,
            scale_x=scale_x,
            scale_y=scale_y,
            figures_dir=figures_dir,
            client=client,
            model=model,
            gabarito=gabarito,
            year=year,
            area=q_area,
            debug_dir=debug_dir,
            language_hint=language_hint,
        )
        all_warnings.extend(warnings)
        # Merge extracted contexts; first extraction wins for a given ID
        for ctx_id, ctx_data in q_contexts.items():
            if ctx_id not in all_contexts:
                all_contexts[ctx_id] = ctx_data

        if question:
            all_figures.extend(question.images)
            _save_cache(cache_dir, cache_key, question.model_dump())
            all_questions.append(question)
            failed_q_nums.discard(block.number)
        else:
            reason = "; ".join(warnings) if warnings else "unknown reason"
            logger.warning("Q%d: skipped — %s", block.number, reason)
            failed_q_nums.add(block.number)

    # Persist (or clear) the failed question list
    if failed_q_nums:
        failed_file.write_text(
            json.dumps(sorted(failed_q_nums), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.warning("%d question(s) failed: %s — saved to %s", len(failed_q_nums), sorted(failed_q_nums), failed_file)
    elif failed_file.exists():
        failed_file.unlink()
        logger.info("All questions extracted successfully — removed %s", failed_file)

    # Write output JSON — one file per area (split by day), or single file
    test_slug = test_name.lower()
    year_slug = str(year) if year else "unknown"

    if day and day in DAY_AREAS:
        for area_slug, q_range in DAY_AREAS[day]:
            area_questions = [q for q in all_questions if q.number in q_range]
            out_json = output_dir / f"{area_slug}_{test_slug}_{year_slug}.json"
            out_json.write_text(
                json.dumps([_question_to_web(q) for q in area_questions], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info("Saved %d questions (%s) to %s", len(area_questions), area_slug, out_json)
    else:
        area_slug = area or "exam"
        out_json = output_dir / f"{area_slug}_{test_slug}_{year_slug}.json"
        out_json.write_text(
            json.dumps([_question_to_web(q) for q in all_questions], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Saved %d questions to %s", len(all_questions), out_json)

    # Build and write contexts.json from contextId/contextIds references
    _write_contexts_json(output_dir, all_questions, all_contexts)

    metadata = ExamMetadata(
        year=year,
        test=test_name,
        area=area,
        prova_pdf=str(prova_pdf),
        gabarito_pdf=str(gabarito_pdf) if gabarito_pdf else None,
        model=model,
        page_range=page_range,
    )

    return ExamResult(
        metadata=metadata,
        questions=all_questions,
        figures=all_figures,
        warnings=all_warnings,
    )


def _question_to_web(q: Question) -> dict:
    """Serialize a Question to the web app JSON format with canonical field order."""
    d: dict = {
        "number": q.number,
        "text": q.text,
        "alternatives": q.alternatives,
        "images": q.images,
        "tags": q.tags,
        "year": q.year,
        "test": q.test,
        "area": q.area,
        "answer": q.answer,
    }
    # Optional fields — only include when present
    if q.contextIds:
        d["contextIds"] = q.contextIds
    elif q.contextId:
        d["contextId"] = q.contextId
    if q.language:
        d["language"] = q.language
    return d


def _write_contexts_json(
    output_dir: Path,
    questions: list[Question],
    extracted_contexts: dict[str, dict] | None = None,
) -> None:
    """
    Collect all contextId/contextIds referenced by questions and write contexts.json.

    - Existing entries with non-empty text are never overwritten (preserves user edits).
    - Existing stub entries (empty text) are updated with freshly extracted content.
    - New IDs get the extracted content if available, otherwise an empty stub.
    - The file is always additive: no previously stored entry is removed.
    """
    contexts_path = output_dir / "contexts.json"
    existing: dict = {}
    if contexts_path.exists():
        try:
            existing = json.loads(contexts_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    extracted = extracted_contexts or {}

    all_ids: set[str] = set()
    for q in questions:
        if q.contextIds:
            all_ids.update(q.contextIds)
        elif q.contextId:
            all_ids.add(q.contextId)

    for ctx_id in sorted(all_ids):
        if ctx_id not in existing:
            # New entry — use extracted content or fall back to stub
            existing[ctx_id] = extracted.get(ctx_id) or {
                "title": None,
                "subtitle": None,
                "text": "",
                "images": [],
                "reference": None,
            }
        else:
            # Entry exists — only update if its text is empty and we have extracted content
            current_text = existing[ctx_id].get("text", "")
            if not current_text and ctx_id in extracted:
                existing[ctx_id].update(extracted[ctx_id])

    if existing:
        contexts_path.write_text(
            json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("Written %d context(s) to %s", len(existing), contexts_path)
