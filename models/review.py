from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class ReviewClass(str, Enum):
    valid = "valid"
    empty = "empty"
    speculative = "speculative"
    spam_offtopic = "spam_offtopic"
    low_effort = "low_effort"
    uncertain = "uncertain"


class ClassificationMethod(str, Enum):
    llm = "llm"
    heuristic = "heuristic"
    fallback = "fallback"


class Review(BaseModel):
    review_id: str
    rating: int = Field(ge=1, le=5)
    text: str = ""
    author: str = ""
    source: str = ""


class Classification(BaseModel):
    label: ReviewClass
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)
    method: ClassificationMethod = ClassificationMethod.heuristic


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0


class AnalyzedReview(BaseModel):
    review_id: str
    rating: int
    text: str
    author: str
    label: ReviewClass
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)
    method: ClassificationMethod = ClassificationMethod.heuristic


class AnalyzeRequest(BaseModel):
    source: Literal["google_maps", "serpapi"] = "google_maps"
    source_id: str = Field(min_length=1)
    reviews_limit: int | None = Field(default=None, ge=1)
    sort: Literal["most_relevant", "newest", "highest_rating", "lowest_rating"] = "newest"


class AnalysisResult(BaseModel):
    source: str
    source_id: str
    place_name: str | None = None
    naive_rating: float
    true_rating: float | None
    delta: float | None
    total: int
    sample_size: int
    official_rating: float | None = None
    official_review_count: int | None = None
    source_limit: int | None = None
    warning: str | None = None
    summary_ru: str | None = None
    pros_ru: list[str] = Field(default_factory=list)
    cons_ru: list[str] = Field(default_factory=list)
    excluded_count: int = 0
    excluded_by_class: dict[str, int] = Field(default_factory=dict)
    per_class_counts: dict[str, int]
    usage: Usage
    reviews: list[AnalyzedReview]
