from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class PairRecord(BaseModel):
    query: str
    doc: str
    score: int = Field(ge=0, le=3)
    rationale: str
    generation_seed: int
    generation_mode: Literal["offline", "teacher"]
    teacher_model: str | None = None


class PreferenceRecord(BaseModel):
    query: str
    doc_a: str
    doc_b: str
    preferred: Literal["A", "B"]
    confidence: float = Field(ge=0.0, le=1.0)
    generation_seed: int
    generation_mode: Literal["offline", "teacher"]
    teacher_model: str | None = None


class ContradictionRecord(BaseModel):
    subject: str
    doc_a: str
    doc_b: str
    contradicted_field: str
    value_a: str
    value_b: str
    is_contradiction: bool
    generation_seed: int
    generation_mode: Literal["offline", "teacher"]
    teacher_model: str | None = None


class ListwisePreferenceRecord(BaseModel):
    query: str
    docs: list[str]
    scores: list[float] = Field(description="Normalized relevance scores, one per doc")
    generation_seed: int
    generation_mode: Literal["offline", "teacher"]
    teacher_model: str | None = None


class HardNegativeRecord(BaseModel):
    query: str
    positive: str
    hard_negative: str
    easy_negative: str
    generation_seed: int
    generation_mode: Literal["offline", "teacher"]
    teacher_model: str | None = None


class QueryExpansionRecord(BaseModel):
    original_query: str
    expanded_queries: list[str]
    generation_seed: int
    generation_mode: Literal["offline", "teacher"]
    teacher_model: str | None = None


class DatasetManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    generated_at: str
    root: str
    seed: int
    generation_mode: Literal["offline", "teacher", "mixed"]
    teacher_model: str | None = None
    datasets: dict[str, dict[str, Any]]
