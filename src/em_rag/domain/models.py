"""Small immutable domain models used by the product runtime."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class RetrievalRequest:
    query: str
    top_k: int = 8
    retrieval_mode: str = "bm25_structured"


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    chunk_id: str
    source_id: str
    content: str
    citation: str
    rank: int
    score: float
    source_title: str = ""
    chapter: str = ""
    section: str = ""
    page_start: int | None = None
    page_end: int | None = None
    channel_scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievalResponse:
    request_id: str
    artifact_id: str
    query: str
    evidence: tuple[Evidence, ...]
    degraded: bool
    degraded_components: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["evidence"] = [item.to_dict() for item in self.evidence]
        value["degraded_components"] = list(self.degraded_components)
        value["warnings"] = list(self.warnings)
        return value


@dataclass(frozen=True)
class AnswerResponse:
    request_id: str
    artifact_id: str
    query: str
    answer: str
    evidence: tuple[Evidence, ...]
    degraded: bool
    degraded_components: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["evidence"] = [item.to_dict() for item in self.evidence]
        value["degraded_components"] = list(self.degraded_components)
        value["warnings"] = list(self.warnings)
        return value
