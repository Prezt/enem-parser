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
    extract_exam(
        prova_pdf=args.prova,
        gabarito_pdf=args.gabarito,
        area=args.area,
        page_range=args.pages,
        output_dir=args.output,
        model=args.model,
        refine_with_llm=not args.no_llm,
        year=args.year,
        debug=args.debug,
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
        "--area",
        choices=["linguagens", "humanas", "natureza", "matematica"],
        help="Exam area",
    )
    extract_parser.add_argument(
        "--pages",
        metavar="START-END",
        type=_parse_page_range,
        help="1-indexed page range (e.g. 1-45)",
    )
    extract_parser.add_argument("--output", "-o", default="output", help="Output directory")
    extract_parser.add_argument(
        "--model",
        default=_DEFAULT_MODEL,
        help=f"Ollama model name (default: {_DEFAULT_MODEL}, fallback: {_FALLBACK_MODEL})",
    )
    extract_parser.add_argument("--year", type=int, help="Exam year (e.g. 2024)")
    extract_parser.add_argument("--no-llm", action="store_true", help="Skip LLM extraction")
    extract_parser.add_argument("--debug", action="store_true", help="Save debug files per stage")
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
