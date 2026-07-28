"""Immutable data contracts for retrieval evaluation artifacts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Mapping


_SPLITS = {"development", "regression", "holdout", "adversarial"}
_JUDGMENT_SOURCES = {
    "agent_pass_1",
    "agent_pass_2",
    "agent_adjudication",
    "human_calibration",
}


def _strict_fields(
    data: Mapping[str, Any], required: set[str], optional: set[str] | None = None
) -> None:
    if not isinstance(data, Mapping):
        raise ValueError("model input must be an object")
    optional = optional or set()
    missing = required - data.keys()
    extra = data.keys() - required - optional
    if missing:
        raise ValueError(f"missing fields: {', '.join(sorted(missing))}")
    if extra:
        raise ValueError(f"unexpected fields: {', '.join(sorted(extra))}")


def _nonblank(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _string_tuple(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be an array of strings")
    items = tuple(_nonblank(item, field) for item in value)
    if len(items) != len(set(items)):
        raise ValueError(f"{field} must not contain duplicates")
    return items


def _bool(value: Any, field: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{field} must be a boolean")
    return value


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{field} must be finite")
    return converted


@dataclass(frozen=True)
class EvalCase:
    query_id: str
    query: str
    category: str
    expected_facets: tuple[str, ...]
    is_hard_negative: bool
    requires_multiple_evidence: bool
    split: Literal["development", "regression", "holdout", "adversarial"]
    language: Literal["en"] = "en"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EvalCase:
        required = {
            "query_id",
            "query",
            "category",
            "expected_facets",
            "is_hard_negative",
            "requires_multiple_evidence",
            "split",
        }
        _strict_fields(data, required, {"language"})
        split = data["split"]
        if not isinstance(split, str):
            raise ValueError("split must be a string")
        if split not in _SPLITS:
            raise ValueError("split is not supported")
        language = data.get("language", "en")
        if language != "en":
            raise ValueError("language must be 'en'")
        return cls(
            query_id=_nonblank(data["query_id"], "query_id"),
            query=_nonblank(data["query"], "query"),
            category=_nonblank(data["category"], "category"),
            expected_facets=_string_tuple(data["expected_facets"], "expected_facets"),
            is_hard_negative=_bool(data["is_hard_negative"], "is_hard_negative"),
            requires_multiple_evidence=_bool(
                data["requires_multiple_evidence"], "requires_multiple_evidence"
            ),
            split=split,
            language=language,
        )


@dataclass(frozen=True)
class Qrel:
    query_id: str
    chunk_id: str
    relevance: Literal[0, 1, 2, 3]
    supported_facets: tuple[str, ...]
    confidence: float
    judgment_source: Literal[
        "agent_pass_1", "agent_pass_2", "agent_adjudication", "human_calibration"
    ]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Qrel:
        required = {
            "query_id",
            "chunk_id",
            "relevance",
            "supported_facets",
            "confidence",
            "judgment_source",
        }
        _strict_fields(data, required)
        relevance = data["relevance"]
        if type(relevance) is not int or relevance not in {0, 1, 2, 3}:
            raise ValueError("relevance must be an integer from 0 to 3")
        confidence = _finite_number(data["confidence"], "confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        source = data["judgment_source"]
        if not isinstance(source, str):
            raise ValueError("judgment_source must be a string")
        if source not in _JUDGMENT_SOURCES:
            raise ValueError("judgment_source is not supported")
        return cls(
            query_id=_nonblank(data["query_id"], "query_id"),
            chunk_id=_nonblank(data["chunk_id"], "chunk_id"),
            relevance=relevance,
            supported_facets=_string_tuple(data["supported_facets"], "supported_facets"),
            confidence=confidence,
            judgment_source=source,
        )


@dataclass(frozen=True)
class RankedEvidence:
    rank: int
    chunk_id: str
    score: float
    citation: str
    text: str
    source_id: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RankedEvidence:
        _strict_fields(data, {"rank", "chunk_id", "score", "citation", "text", "source_id"})
        rank = data["rank"]
        if type(rank) is not int or rank < 1:
            raise ValueError("rank must be a positive integer")
        return cls(
            rank=rank,
            chunk_id=_nonblank(data["chunk_id"], "chunk_id"),
            score=_finite_number(data["score"], "score"),
            citation=_nonblank(data["citation"], "citation"),
            text=_nonblank(data["text"], "text"),
            source_id=_nonblank(data["source_id"], "source_id"),
        )


@dataclass(frozen=True)
class RetrievalRun:
    run_id: str
    query_id: str
    artifact_id: str
    results: tuple[RankedEvidence, ...]
    degraded: bool
    confidence_threshold: float

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RetrievalRun:
        _strict_fields(
            data,
            {
                "run_id",
                "query_id",
                "artifact_id",
                "results",
                "degraded",
                "confidence_threshold",
            },
        )
        raw_results = data["results"]
        if not isinstance(raw_results, (list, tuple)):
            raise ValueError("results must be an array")
        results = tuple(RankedEvidence.from_dict(item) for item in raw_results)
        ranks = [item.rank for item in results]
        if len(ranks) != len(set(ranks)):
            raise ValueError("results must not contain duplicate ranks")
        chunk_ids = [item.chunk_id for item in results]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("results must not contain duplicate chunk IDs")
        return cls(
            run_id=_nonblank(data["run_id"], "run_id"),
            query_id=_nonblank(data["query_id"], "query_id"),
            artifact_id=_nonblank(data["artifact_id"], "artifact_id"),
            results=results,
            degraded=_bool(data["degraded"], "degraded"),
            confidence_threshold=_finite_number(
                data["confidence_threshold"], "confidence_threshold"
            ),
        )
