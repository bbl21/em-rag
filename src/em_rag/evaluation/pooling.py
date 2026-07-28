"""Blinded pooling and independent relevance-judgment reconciliation."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .models import EvalCase, RetrievalRun


def _nonblank(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _bool(value: Any, field: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{field} must be a boolean")
    return value


def _string_tuple(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be an array of strings")
    items = tuple(_nonblank(item, field) for item in value)
    if len(items) != len(set(items)):
        raise ValueError(f"{field} must not contain duplicates")
    return items


def _confidence(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("confidence must be a number")
    converted = float(value)
    if not math.isfinite(converted) or not 0.0 <= converted <= 1.0:
        raise ValueError("confidence must be a finite number between 0 and 1")
    return converted


@dataclass(frozen=True)
class JudgeRow:
    """Evidence presented to a judge without retriever identity or ranking signals."""

    judgment_id: str
    query_id: str
    query: str
    expected_facets: tuple[str, ...]
    chunk_id: str
    source_id: str
    citation: str
    text: str

    def __post_init__(self) -> None:
        for field in (
            "judgment_id",
            "query_id",
            "query",
            "chunk_id",
            "source_id",
            "citation",
            "text",
        ):
            _nonblank(getattr(self, field), field)
        object.__setattr__(
            self,
            "expected_facets",
            _string_tuple(self.expected_facets, "expected_facets"),
        )


@dataclass(frozen=True)
class Judgment:
    """One independently produced, source-grounded judgment."""

    judgment_id: str
    relevance: int
    supported_facets: tuple[str, ...]
    scope_correct: bool
    citation_supported: bool
    pollution: bool
    confidence: float
    source_quote: str
    reason: str

    def __post_init__(self) -> None:
        _nonblank(self.judgment_id, "judgment_id")
        if (
            type(self.relevance) is not int
            or self.relevance not in {0, 1, 2, 3}
        ):
            raise ValueError("relevance must be an integer from 0 to 3")
        object.__setattr__(
            self,
            "supported_facets",
            _string_tuple(self.supported_facets, "supported_facets"),
        )
        _bool(self.scope_correct, "scope_correct")
        _bool(self.citation_supported, "citation_supported")
        _bool(self.pollution, "pollution")
        object.__setattr__(self, "confidence", _confidence(self.confidence))
        _nonblank(self.source_quote, "source_quote")
        _nonblank(self.reason, "reason")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Judgment:
        required = {
            "judgment_id",
            "relevance",
            "supported_facets",
            "scope_correct",
            "citation_supported",
            "pollution",
            "confidence",
            "source_quote",
            "reason",
        }
        if not isinstance(data, Mapping):
            raise ValueError("judgment input must be an object")
        missing = required - data.keys()
        extra = data.keys() - required
        if missing:
            raise ValueError(f"missing fields: {', '.join(sorted(missing))}")
        if extra:
            raise ValueError(f"unexpected fields: {', '.join(sorted(extra))}")
        return cls(
            judgment_id=data["judgment_id"],
            relevance=data["relevance"],
            supported_facets=data["supported_facets"],
            scope_correct=data["scope_correct"],
            citation_supported=data["citation_supported"],
            pollution=data["pollution"],
            confidence=data["confidence"],
            source_quote=data["source_quote"],
            reason=data["reason"],
        )


@dataclass(frozen=True)
class MergedJudgment:
    judgment_id: str
    pass1: Judgment
    pass2: Judgment
    adjudication: Judgment | None
    human_calibration: Judgment | None
    final: Judgment | None


@dataclass(frozen=True)
class AgreementReport:
    total: int
    exact_agreement: float
    weighted_cohen_kappa: float
    release_eligible: bool
    status: str
    adjudication_ids: tuple[str, ...]
    unresolved_ids: tuple[str, ...]


@dataclass(frozen=True)
class JudgmentMerge:
    rows: tuple[MergedJudgment, ...]
    agreement: AgreementReport


def _judgment_id(query_id: str, chunk_id: str) -> str:
    digest = hashlib.sha256(f"{query_id}\0{chunk_id}".encode("utf-8")).hexdigest()
    return f"j_{digest[:16]}"


def build_blind_pool(
    runs: Sequence[tuple[EvalCase, Sequence[RetrievalRun]]], seed: int
) -> tuple[JudgeRow, ...]:
    """Build a deterministic shuffled pool from per-query retriever runs.

    Each input item pairs an evaluation case with independently generated runs for
    that query. Duplicate query/chunk evidence is emitted once. Conflicting copies
    are rejected so input order cannot silently select the text shown to judges.
    """

    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    pooled: dict[tuple[str, str], JudgeRow] = {}
    for case, query_runs in runs:
        if not isinstance(case, EvalCase):
            raise ValueError("each pool group must start with an EvalCase")
        for run in query_runs:
            if not isinstance(run, RetrievalRun):
                raise ValueError("pool groups must contain RetrievalRun values")
            if run.query_id != case.query_id:
                raise ValueError("RetrievalRun query_id must match its EvalCase")
            for evidence in run.results:
                row = JudgeRow(
                    judgment_id=_judgment_id(case.query_id, evidence.chunk_id),
                    query_id=case.query_id,
                    query=case.query,
                    expected_facets=case.expected_facets,
                    chunk_id=evidence.chunk_id,
                    source_id=evidence.source_id,
                    citation=evidence.citation,
                    text=evidence.text,
                )
                key = (case.query_id, evidence.chunk_id)
                existing = pooled.get(key)
                if existing is not None and existing != row:
                    raise ValueError(
                        "conflicting evidence for the same query_id and chunk_id"
                    )
                pooled[key] = row

    rows = [pooled[key] for key in sorted(pooled)]
    random.Random(seed).shuffle(rows)
    return tuple(rows)


def _coerce_judgments(
    values: Sequence[Judgment | Mapping[str, Any]], label: str
) -> dict[str, Judgment]:
    indexed: dict[str, Judgment] = {}
    for value in values:
        judgment = value if isinstance(value, Judgment) else Judgment.from_dict(value)
        if judgment.judgment_id in indexed:
            raise ValueError(f"{label} contains duplicate judgment_id values")
        indexed[judgment.judgment_id] = judgment
    return indexed


def _quadratic_weighted_kappa(pass1: Sequence[int], pass2: Sequence[int]) -> float:
    if not pass1:
        return 1.0
    total = len(pass1)
    counts1 = [pass1.count(grade) for grade in range(4)]
    counts2 = [pass2.count(grade) for grade in range(4)]
    observed = sum((left - right) ** 2 for left, right in zip(pass1, pass2)) / (
        9 * total
    )
    expected = sum(
        ((left - right) ** 2 / 9) * counts1[left] * counts2[right]
        for left in range(4)
        for right in range(4)
    ) / (total * total)
    if expected == 0.0:
        return 1.0 if observed == 0.0 else 0.0
    return 1.0 - observed / expected


def _requires_adjudication(left: Judgment, right: Judgment) -> bool:
    return (
        abs(left.relevance - right.relevance) > 1
        or left.confidence < 0.75
        or right.confidence < 0.75
        or left.scope_correct != right.scope_correct
        or left.citation_supported != right.citation_supported
    )


def _substantively_equal(left: Judgment, right: Judgment) -> bool:
    return (
        left.relevance == right.relevance
        and left.supported_facets == right.supported_facets
        and left.scope_correct == right.scope_correct
        and left.citation_supported == right.citation_supported
        and left.pollution == right.pollution
    )


def merge_judgments(
    pass1: Sequence[Judgment | Mapping[str, Any]],
    pass2: Sequence[Judgment | Mapping[str, Any]],
    adjudication: Sequence[Judgment | Mapping[str, Any]] | None = None,
    *,
    human_calibration: Sequence[Judgment | Mapping[str, Any]] | None = None,
    sampled_ids: Sequence[str] = (),
) -> JudgmentMerge:
    """Reconcile two auditable passes without averaging away disagreement."""

    first = _coerce_judgments(pass1, "pass1")
    second = _coerce_judgments(pass2, "pass2")
    if first.keys() != second.keys():
        raise ValueError("pass1 and pass2 must contain the same judgment_id values")
    adjudicated = _coerce_judgments(adjudication or (), "adjudication")
    if not adjudicated.keys() <= first.keys():
        raise ValueError("adjudication contains an unknown judgment_id")
    calibrated = _coerce_judgments(
        human_calibration or (), "human_calibration"
    )
    if not calibrated.keys() <= first.keys():
        raise ValueError("human_calibration contains an unknown judgment_id")
    if not isinstance(sampled_ids, (list, tuple)):
        raise ValueError("sampled_ids must be an array of judgment IDs")
    sampled = tuple(_nonblank(item, "sampled_ids") for item in sampled_ids)
    if len(sampled) != len(set(sampled)):
        raise ValueError("sampled_ids must not contain duplicates")
    sampled_set = set(sampled)
    if not sampled_set <= first.keys():
        raise ValueError("sampled_ids contains an unknown judgment_id")

    ids = sorted(first)
    required_ids = tuple(
        judgment_id
        for judgment_id in ids
        if _requires_adjudication(first[judgment_id], second[judgment_id])
    )
    if not adjudicated.keys() <= set(required_ids):
        raise ValueError("adjudication is limited to rows selected for adjudication")
    mandatory_unresolved = set(required_ids) - adjudicated.keys()
    non_mandatory_missing_final = {
        judgment_id
        for judgment_id in ids
        if judgment_id not in adjudicated
        and not _substantively_equal(
            first[judgment_id],
            second[judgment_id],
        )
    }
    unresolved_without_human = mandatory_unresolved | non_mandatory_missing_final
    allowed_calibration_ids = unresolved_without_human | sampled_set
    if not calibrated.keys() <= allowed_calibration_ids:
        raise ValueError(
            "human_calibration is limited to unresolved or sampled rows"
        )

    rows = []
    for judgment_id in ids:
        left = first[judgment_id]
        right = second[judgment_id]
        decided = adjudicated.get(judgment_id)
        human = calibrated.get(judgment_id)
        final = human or decided or (
            left if _substantively_equal(left, right) else None
        )
        rows.append(
            MergedJudgment(
                judgment_id=judgment_id,
                pass1=left,
                pass2=right,
                adjudication=decided,
                human_calibration=human,
                final=final,
            )
        )

    exact = (
        sum(first[item].relevance == second[item].relevance for item in ids) / len(ids)
        if ids
        else 1.0
    )
    kappa = _quadratic_weighted_kappa(
        [first[item].relevance for item in ids],
        [second[item].relevance for item in ids],
    )
    unresolved_mandatory = (
        set(required_ids) - adjudicated.keys() - calibrated.keys()
    )
    missing_final = {row.judgment_id for row in rows if row.final is None}
    unresolved = tuple(
        item for item in ids if item in unresolved_mandatory or item in missing_final
    )
    eligible = kappa >= 0.8 and not unresolved
    report = AgreementReport(
        total=len(ids),
        exact_agreement=exact,
        weighted_cohen_kappa=kappa,
        release_eligible=eligible,
        status="release_eligible" if eligible else "not_release_eligible",
        adjudication_ids=required_ids,
        unresolved_ids=unresolved,
    )
    return JudgmentMerge(rows=tuple(rows), agreement=report)
