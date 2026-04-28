from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, field_validator, model_validator


VALID_ANSWERS = {"a", "b", "c", "d", "e", "annulled"}
ALTERNATIVE_KEYS = {"a", "b", "c", "d", "e"}


class Alternative(BaseModel):
    key: str
    text: str

    @field_validator("key")
    @classmethod
    def key_must_be_valid(cls, v: str) -> str:
        if v not in ALTERNATIVE_KEYS:
            raise ValueError(f"Alternative key must be one of {ALTERNATIVE_KEYS}, got {v!r}")
        return v


class Question(BaseModel):
    number: int
    text: str
    alternatives: dict[str, str]
    images: list[str] = []
    tags: list[str] = []
    answer: Optional[str] = None
    year: Optional[int] = None
    test: str = "ENEM"
    area: Optional[str] = None
    language: Optional[str] = None
    _review_needed: bool = False

    @field_validator("alternatives")
    @classmethod
    def alternatives_must_be_complete(cls, v: dict[str, str]) -> dict[str, str]:
        if set(v.keys()) != ALTERNATIVE_KEYS:
            raise ValueError(
                f"alternatives must have exactly keys {ALTERNATIVE_KEYS}, got {set(v.keys())}"
            )
        return v

    @field_validator("answer")
    @classmethod
    def answer_must_be_valid(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in VALID_ANSWERS:
            raise ValueError(f"answer must be one of {VALID_ANSWERS}, got {v!r}")
        return v

    @field_validator("images")
    @classmethod
    def images_use_forward_slashes(cls, v: list[str]) -> list[str]:
        return [p.replace("\\", "/") for p in v]


class ExamMetadata(BaseModel):
    year: Optional[int] = None
    test: str = "ENEM"
    area: Optional[str] = None
    prova_pdf: str
    gabarito_pdf: Optional[str] = None
    model: str
    page_range: Optional[tuple[int, int]] = None


class ExamResult(BaseModel):
    metadata: ExamMetadata
    questions: list[Question]
    figures: list[str] = []
    warnings: list[str] = []
