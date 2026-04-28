import json
import pytest

from enem_extractor.parse import normalize_text, split_by_question
from enem_extractor.llm import extract_json


# --- split_by_question ---

def test_split_by_question_basic():
    text = (
        "QUESTÃO 91\n"
        "Texto da questão 91.\n"
        "A alt a\nB alt b\nC alt c\nD alt d\nE alt e\n"
        "QUESTÃO 92\n"
        "Texto da questão 92.\n"
        "A alt a\nB alt b\nC alt c\nD alt d\nE alt e\n"
    )
    blocks = split_by_question(text)
    assert len(blocks) == 2
    assert blocks[0].number == 91
    assert blocks[1].number == 92
    assert "91" in blocks[0].raw_text
    assert "92" in blocks[1].raw_text


def test_split_by_question_with_inline_QUESTAO_in_caption():
    """Should not false-match 'QUESTÃO de gênero' inside quoted text."""
    text = (
        "QUESTÃO 45\n"
        "O texto aborda a QUESTÃO de gênero na sociedade contemporânea.\n"
        "A alt a\nB alt b\nC alt c\nD alt d\nE alt e\n"
        "QUESTÃO 46\n"
        "Texto da questão 46.\n"
        "A alt a\nB alt b\nC alt c\nD alt d\nE alt e\n"
    )
    blocks = split_by_question(text)
    assert len(blocks) == 2, f"Expected 2 blocks, got {len(blocks)}: {[b.number for b in blocks]}"
    assert blocks[0].number == 45
    assert blocks[1].number == 46
    # The inline "QUESTÃO de gênero" must be part of block 45, not a new block
    assert "QUESTÃO de gênero" in blocks[0].raw_text


def test_split_by_question_preserves_positions():
    text = "QUESTÃO 10\nTexto.\nA a\nB b\nC c\nD d\nE e\nQUESTÃO 11\nTexto2.\nA a\nB b\nC c\nD d\nE e\n"
    blocks = split_by_question(text)
    assert blocks[0].start_pos < blocks[1].start_pos
    assert blocks[0].end_pos == blocks[1].start_pos


def test_split_by_question_questao_variant():
    """Should handle QUESTAO (without accent) as well."""
    text = "QUESTAO 5\nTexto.\nA a\nB b\nC c\nD d\nE e\n"
    blocks = split_by_question(text)
    assert len(blocks) == 1
    assert blocks[0].number == 5


# --- normalize_text ---

def test_normalize_strips_watermark():
    text = "EM2024EM2024EM2024 Texto normal."
    result = normalize_text(text)
    assert "EM2024" not in result
    assert "Texto normal." in result


def test_normalize_fixes_questao():
    text = "QUESTAO 10"
    result = normalize_text(text)
    assert "QUESTÃO 10" in result


def test_normalize_removes_long_garbage():
    text = "Normal ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789 Normal"
    result = normalize_text(text)
    assert "ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789" not in result


def test_normalize_fixes_soft_hyphen_linebreak():
    text = "palavra-\ncontinuação"
    result = normalize_text(text)
    assert "palavra-\ncontinuação" not in result
    assert "palavracontinuação" in result


# --- extract_json ---

def test_extract_json_clean_array():
    raw = '[{"number": 91, "text": "Texto", "alternatives": {"a": "A", "b": "B", "c": "C", "d": "D", "e": "E"}}]'
    result = extract_json(raw)
    parsed = json.loads(result)
    assert isinstance(parsed, list)
    assert parsed[0]["number"] == 91


def test_extract_json_handles_markdown_fences():
    raw = '```json\n[{"number": 1, "text": "T", "alternatives": {}}]\n```'
    result = extract_json(raw)
    parsed = json.loads(result)
    assert isinstance(parsed, list)


def test_extract_json_handles_preamble():
    raw = 'Here is the JSON output:\n\n[{"number": 2, "text": "X", "alternatives": {}}]'
    result = extract_json(raw)
    parsed = json.loads(result)
    assert parsed[0]["number"] == 2


def test_extract_json_handles_single_object():
    raw = '{"number": 3, "text": "Y", "alternatives": {}}'
    result = extract_json(raw)
    parsed = json.loads(result)
    assert isinstance(parsed, list)
    assert parsed[0]["number"] == 3


def test_extract_json_raises_on_garbage():
    with pytest.raises(ValueError):
        extract_json("This is not JSON at all, just plain text.")
