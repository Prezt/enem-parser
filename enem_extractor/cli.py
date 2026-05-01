from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .pipeline import extract_exam, _DEFAULT_MODEL, _FALLBACK_MODEL
from .schema import Question

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _parse_page_range(s: str) -> tuple[int, int]:
    parts = s.split("-")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("page range must be in format START-END (e.g. 1-45)")
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        raise argparse.ArgumentTypeError("page range values must be integers")


def cmd_extract(args: argparse.Namespace) -> None:
    prova_path = Path(args.prova)
    if not prova_path.exists():
        logger.error("Exam PDF not found: %s", prova_path.resolve())
        sys.exit(1)
    if args.gabarito and not Path(args.gabarito).exists():
        logger.error("Answer key PDF not found: %s", Path(args.gabarito).resolve())
        sys.exit(1)
    extract_exam(
        prova_pdf=args.prova,
        gabarito_pdf=args.gabarito,
        area=args.area,
        day=args.day,
        page_range=args.pages,
        output_dir=args.output,
        model=args.model,
        provider=args.provider,
        refine_with_llm=not args.no_llm,
        year=args.year,
        debug=args.debug,
        retry_failed=args.retry_failed,
    )


def cmd_validate(args: argparse.Namespace) -> None:
    path = Path(args.json_file)
    if not path.exists():
        logger.error("File not found: %s", path)
        sys.exit(1)

    data = json.loads(path.read_text(encoding="utf-8"))
    questions_data = data.get("questions", [])
    errors = 0

    for raw in questions_data:
        try:
            Question(**raw)
        except Exception as exc:
            logger.error("Q%s: validation failed: %s", raw.get("number", "?"), exc)
            errors += 1

    if errors:
        logger.error("%d question(s) failed validation", errors)
        sys.exit(1)
    else:
        logger.info("All %d questions passed validation", len(questions_data))


def cmd_review(args: argparse.Namespace) -> None:
    path = Path(args.json_file)
    if not path.exists():
        logger.error("File not found: %s", path)
        sys.exit(1)

    data = json.loads(path.read_text(encoding="utf-8"))
    warnings = data.get("warnings", [])

    if not warnings:
        print("No warnings recorded.")
        return

    print(f"\n{len(warnings)} warning(s):\n")
    for w in warnings:
        print(f"  - {w}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="enem-extract",
        description="Extract structured JSON + figures from ENEM PDF exams",
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- extract (default command) ---
    extract_parser = subparsers.add_parser("extract", help="Extract questions from a PDF")
    extract_parser.add_argument("prova", metavar="prova.pdf", help="Path to exam PDF")
    extract_parser.add_argument(
        "gabarito", metavar="gabarito.pdf", nargs="?", help="Path to answer key PDF (optional)"
    )
    extract_parser.add_argument(
        "--day",
        type=int,
        choices=[1, 2],
        help="Exam day (1 = linguagens+humanas, 2 = nature+math). Assigns area automatically by question range.",
    )
    extract_parser.add_argument(
        "--area",
        choices=["math", "nature", "linguagens", "humanas"],
        help="Exam area (used when --day is not set)",
    )
    extract_parser.add_argument(
        "--pages",
        metavar="START-END",
        type=_parse_page_range,
        help="1-indexed page range (e.g. 1-45)",
    )
    extract_parser.add_argument("--output", "-o", default="output", help="Output directory")
    extract_parser.add_argument(
        "--provider",
        choices=["ollama", "anthropic"],
        default="ollama",
        help="LLM provider: 'ollama' (local, default) or 'anthropic' (Claude API, reads ANTHROPIC_API_KEY)",
    )
    extract_parser.add_argument(
        "--model",
        default=_DEFAULT_MODEL,
        help=f"Model name — Ollama model (default: {_DEFAULT_MODEL}) or Claude model when --provider=anthropic (default: claude-haiku-4-5)",
    )
    extract_parser.add_argument("--year", type=int, help="Exam year (e.g. 2024)")
    extract_parser.add_argument("--no-llm", action="store_true", help="Skip LLM extraction")
    extract_parser.add_argument("--debug", action="store_true", help="Save debug files per stage")
    extract_parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Re-run only questions that failed in a previous extraction (reads failed_questions.json from output dir)",
    )
    extract_parser.set_defaults(func=cmd_extract)

    # --- validate ---
    validate_parser = subparsers.add_parser("validate", help="Re-run Pydantic validation on output JSON")
    validate_parser.add_argument("json_file", metavar="output.json")
    validate_parser.set_defaults(func=cmd_validate)

    # --- review ---
    review_parser = subparsers.add_parser("review", help="Print questions flagged for review")
    review_parser.add_argument("json_file", metavar="output.json")
    review_parser.set_defaults(func=cmd_review)

    args = parser.parse_args()

    # If no subcommand given but positional args look like a PDF path, run extract
    if args.command is None:
        if len(sys.argv) > 1 and sys.argv[1].endswith(".pdf"):
            # Re-parse treating first arg as positional for extract
            sys.argv.insert(1, "extract")
            args = parser.parse_args()
        else:
            parser.print_help()
            sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
