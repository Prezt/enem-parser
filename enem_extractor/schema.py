from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, field_validator


VALID_ANSWERS = {"a", "b", "c", "d", "e", "annulled"}
ALTERNATIVE_KEYS = {"a", "b", "c", "d", "e"}
VALID_AREAS = {"math", "nature", "linguagens", "humanas"}
VALID_LANGUAGES = {"en", "es"}


class Question(BaseModel):
    # Required
    number: int
    text: str
    alternatives: dict[str, str]
    answer: Optional[str] = None
    tags: list[str] = []
    year: Optional[int] = None
    test: str = "ENEM"
    area: Optional[str] = None

    # Optional
    difficulty: Optional[int] = None
    images: list[str] = []
    contextId: Optional[str] = None
    contextIds: Optional[list[str]] = None
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

    @field_validator("difficulty")
    @classmethod
    def difficulty_must_be_valid(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and not (1 <= v <= 10):
            raise ValueError(f"difficulty must be between 1 and 10, got {v!r}")
        return v

    @field_validator("language")
    @classmethod
    def language_must_be_valid(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in VALID_LANGUAGES:
            raise ValueError(f"language must be one of {VALID_LANGUAGES}, got {v!r}")
        return v


class Context(BaseModel):
    title: Optional[str] = None
    subtitle: Optional[str] = None
    text: str
    images: list[str] = []
    reference: Optional[str] = None


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
