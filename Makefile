.PHONY: install lint test clean

install:
	pip install -e ".[dev]" 2>/dev/null || pip install -e .
	pip install ruff pytest

lint:
	ruff check enem_extractor/ tests/

test:
	pytest tests/ -v

clean:
	rm -rf output/ debug/ __pycache__ enem_extractor/__pycache__ tests/__pycache__ .pytest_cache
	find . -name "*.pyc" -delete
