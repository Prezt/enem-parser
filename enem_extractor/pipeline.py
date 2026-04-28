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
from .schema import ExamMetadata, ExamResult, Question

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


def _detect_image_for_block(
    pdf_path: Path,
    block: Block,
    page_idx: int,
) -> bool:
    """Return True if the block likely has an associated figure."""
    images = find_embedded_images(pdf_path, page_idx)
    if images:
        return True
    # Simple heuristic: text mentions "figura" or "gráfico"
    figure_pattern = re.compile(r"(conforme a figura|veja (o |a )?(figura|gráfico|imagem)|figura \d)", re.IGNORECASE)
    return bool(figure_pattern.search(block.raw_text))


def _find_page_for_block(block: Block, page_texts: list[tuple[int, str]]) -> int:
    """Find which page index a block likely appears on (by matching question number)."""
    header = f"QUESTÃO {block.number}"
    for page_idx, page_text in page_texts:
        if header in page_text or f"QUESTAO {block.number}" in page_text.upper():
            return page_idx
    return 0  # fallback


def _process_block(
    block: Block,
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
    has_image = _detect_image_for_block(pdf_path, block, page_idx)

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
        embedded = find_embedded_images(pdf_path, page_idx)
        for i, img_rect in enumerate(embedded):
            fig_name = f"q{q_num:03d}_fig{i+1}.png"
            fig_path = figures_dir / fig_name
            try:
                crop_figure(png_path, img_rect.bbox_pts, scale_x, scale_y, out_path=fig_path)
                cropped_figures.append(str(fig_path))
                logger.info("Q%d: cropped figure -> %s", q_num, fig_name)
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
    figures_dir = output_dir / "figures"
    cache_dir = output_dir / ".cache"
    debug_dir = output_dir / "debug" if debug else None

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

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

    for block in blocks:
        cache_key = _cache_key(pdf_hash, block.number)

        # Resume: use cached result if available
        if cache_key in cache:
            logger.info("Q%d: loaded from cache", block.number)
            try:
                all_questions.append(Question(**cache[cache_key]))
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

    metadata = ExamMetadata(
        year=year,
        test=test_name,
        area=area,
        prova_pdf=str(prova_pdf),
        gabarito_pdf=str(gabarito_pdf) if gabarito_pdf else None,
        model=model,
        page_range=page_range,
    )

    result = ExamResult(
        metadata=metadata,
        questions=all_questions,
        figures=all_figures,
        warnings=all_warnings,
    )

    # Save output JSON
    out_json = output_dir / f"questions_{area or 'exam'}_{year or 'unknown'}.json"
    out_json.write_text(
        json.dumps(result.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Saved %d questions to %s", len(all_questions), out_json)

    return result
