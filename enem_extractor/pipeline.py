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
_FALLBACK_MODEL = "qwen2.5:7b-instruct"


def _pdf_hash(pdf_path: Path) -> str:
    h = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _cache_key(pdf_hash: str, question_number: int) -> str:
    return f"{pdf_hash}_q{question_number:03d}"


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
) -> tuple[Optional[Question], list[str]]:
    """Process one question block. Returns (Question | None, warnings)."""
    warnings: list[str] = []
    q_num = block.number

    page_idx = _find_page_for_block(block, page_texts)
    y_top, y_bottom = _question_vertical_bounds(pdf_path, page_idx, q_num, next_q_num)
    images_in_block = _images_for_block(pdf_path, page_idx, y_top, y_bottom)
    has_image = _detect_image_for_block(block, images_in_block)

    prompt = build_extraction_prompt(block.raw_text, has_image_marker=has_image, year=year, area=area)

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
        return None, warnings

    if debug_dir:
        (debug_dir / "responses" / f"{q_num:03d}.txt").write_text(response, encoding="utf-8")

    try:
        json_str = extract_json(response)
        data_list = json.loads(json_str)
        if not isinstance(data_list, list) or len(data_list) == 0:
            raise ValueError("Empty or non-list JSON")
        data = data_list[0]
    except (ValueError, json.JSONDecodeError, KeyError) as exc:
        warnings.append(f"Q{q_num}: JSON parse failed: {exc}")
        return None, warnings

    # Merge gabarito answer
    data["answer"] = gabarito.get(q_num)
    data.setdefault("year", year)
    data.setdefault("area", area)

    # Crop figures if detected
    cropped_figures: list[str] = []
    if has_image and page_idx < len(png_paths):
        png_path = png_paths[page_idx]
        for i, img_rect in enumerate(images_in_block):
            year_tag = f"_{year}" if year else ""
            fig_name = f"q{q_num:03d}{year_tag}_fig{i+1}.png"
            fig_path = figures_dir / fig_name
            try:
                crop_figure(png_path, img_rect.bbox_pts, scale_x, scale_y, out_path=fig_path)
                cropped_figures.append(f"figuras/{fig_name}")
                logger.info("Q%d: cropped figure -> figuras/%s", q_num, fig_name)
            except Exception as exc:
                warnings.append(f"Q{q_num}: figure crop failed: {exc}")

    data["images"] = cropped_figures or data.get("images", [])

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
            return None, warnings

    logger.info("Q%d: OK", q_num) if not warnings else logger.warning("Q%d: %s", q_num, "; ".join(warnings))
    return question, warnings


def extract_exam(
    prova_pdf: str | Path,
    gabarito_pdf: str | Path | None = None,
    *,
    area: str | None = None,
    page_range: tuple[int, int] | None = None,
    output_dir: str | Path = "output",
    model: str = _DEFAULT_MODEL,
    refine_with_llm: bool = True,
    year: int | None = None,
    test_name: str = "ENEM",
    debug: bool = False,
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
    client = OllamaClient()

    all_questions: list[Question] = []
    all_warnings: list[str] = []
    all_figures: list[str] = []

    for idx, block in enumerate(blocks):
        next_q_num = blocks[idx + 1].number if idx + 1 < len(blocks) else None
        cache_key = _cache_key(pdf_hash, block.number)

        # Resume: use cached result if available
        if cache_key in cache:
            logger.info("Q%d: loaded from cache", block.number)
            try:
                cached = cache[cache_key]
                # Always apply current run's area/year/answer so stale cache entries are corrected
                if area is not None:
                    cached["area"] = area
                if year is not None:
                    cached["year"] = year
                if gabarito.get(block.number):
                    cached["answer"] = gabarito[block.number]
                all_questions.append(Question(**cached))
                continue
            except Exception:
                logger.warning("Q%d: cache entry invalid, re-extracting", block.number)

        if not refine_with_llm:
            # Skip LLM, produce a minimal question from regex parse
            from .parse import Block as _Block  # noqa: F401
            # Use the existing parse_question from enem_parser if available
            # Otherwise produce a stub
            stub = Question(
                number=block.number,
                text=block.raw_text,
                alternatives={"a": "", "b": "", "c": "", "d": "", "e": ""},
                answer=gabarito.get(block.number),
                year=year,
                area=area,
            )
            all_questions.append(stub)
            continue

        question, warnings = _process_block(
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
            area=area,
            debug_dir=debug_dir,
        )
        all_warnings.extend(warnings)

        if question:
            all_figures.extend(question.images)
            _save_cache(cache_dir, cache_key, question.model_dump())
            all_questions.append(question)
        else:
            logger.warning("Q%d: skipped (no valid output)", block.number)

    # Output filename: {area}_{test}_{year}.json  (web app convention)
    area_slug = area or "exam"
    test_slug = test_name.lower()
    year_slug = str(year) if year else "unknown"
    out_json = output_dir / f"{area_slug}_{test_slug}_{year_slug}.json"

    # Write flat questions array (web app format)
    questions_data = [_question_to_web(q) for q in all_questions]
    out_json.write_text(
        json.dumps(questions_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Saved %d questions to %s", len(all_questions), out_json)

    # Build and write contexts.json from contextId/contextIds references
    _write_contexts_json(output_dir, all_questions)

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
    """Serialize a Question to the web app JSON format."""
    d: dict = {
        "number": q.number,
        "text": q.text,
        "alternatives": q.alternatives,
        "answer": q.answer,
        "tags": q.tags,
        "year": q.year,
        "test": q.test,
        "area": q.area,
        "images": q.images,
    }
    if q.contextIds:
        d["contextIds"] = q.contextIds
    elif q.contextId:
        d["contextId"] = q.contextId
    if q.language:
        d["language"] = q.language
    return d


def _write_contexts_json(output_dir: Path, questions: list[Question]) -> None:
    """
    Collect all contextId/contextIds referenced by questions and write contexts.json.
    Stubs are written for any ID not already present; existing entries are preserved.
    """
    contexts_path = output_dir / "contexts.json"
    existing: dict = {}
    if contexts_path.exists():
        try:
            existing = json.loads(contexts_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    all_ids: set[str] = set()
    for q in questions:
        if q.contextIds:
            all_ids.update(q.contextIds)
        elif q.contextId:
            all_ids.add(q.contextId)

    for ctx_id in sorted(all_ids):
        if ctx_id not in existing:
            existing[ctx_id] = {
                "title": None,
                "subtitle": None,
                "text": "",
                "images": [],
                "reference": None,
            }

    if existing:
        contexts_path.write_text(
            json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("Written %d context(s) to %s", len(existing), contexts_path)
