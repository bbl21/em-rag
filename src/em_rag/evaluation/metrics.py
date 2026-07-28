"""Pure retrieval quality metrics."""

from __future__ import annotations

import math
from collections.abc import Sequence

from .models import EvalCase, Qrel, RankedEvidence, RetrievalRun


def _validate_k(k: int) -> None:
    if type(k) is not int or k <= 0:
        raise ValueError("k must be a positive integer")


def _relevance_by_chunk(qrels: Sequence[Qrel], query_id: str) -> dict[str, int]:
    return {
        qrel.chunk_id: qrel.relevance
        for qrel in qrels
        if qrel.query_id == query_id
    }


def _top_k(run: RetrievalRun, k: int) -> tuple[RankedEvidence, ...]:
    return tuple(
        evidence
        for evidence in sorted(run.results, key=lambda evidence: evidence.rank)
        if evidence.rank <= k
    )


def recall_at_k(qrels: Sequence[Qrel], run: RetrievalRun, k: int) -> float:
    _validate_k(k)
    relevance = _relevance_by_chunk(qrels, run.query_id)
    relevant_chunks = {chunk_id for chunk_id, grade in relevance.items() if grade > 0}
    if not relevant_chunks:
        return 0.0
    retrieved_relevant = sum(
        evidence.chunk_id in relevant_chunks for evidence in _top_k(run, k)
    )
    return retrieved_relevant / len(relevant_chunks)


def precision_at_k(qrels: Sequence[Qrel], run: RetrievalRun, k: int) -> float:
    _validate_k(k)
    relevance = _relevance_by_chunk(qrels, run.query_id)
    retrieved_relevant = sum(
        relevance.get(evidence.chunk_id, 0) > 0 for evidence in _top_k(run, k)
    )
    return retrieved_relevant / k


def mrr_at_k(qrels: Sequence[Qrel], run: RetrievalRun, k: int) -> float:
    _validate_k(k)
    relevance = _relevance_by_chunk(qrels, run.query_id)
    for evidence in _top_k(run, k):
        if relevance.get(evidence.chunk_id, 0) > 0:
            return 1.0 / evidence.rank
    return 0.0


def _dcg(ranked_relevances: Sequence[tuple[int, int]]) -> float:
    return sum(
        (2**relevance - 1) / math.log2(rank + 1)
        for rank, relevance in ranked_relevances
    )


def ndcg_at_k(qrels: Sequence[Qrel], run: RetrievalRun, k: int) -> float:
    _validate_k(k)
    relevance = _relevance_by_chunk(qrels, run.query_id)
    actual = [
        (evidence.rank, relevance.get(evidence.chunk_id, 0))
        for evidence in _top_k(run, k)
    ]
    ideal = list(enumerate(sorted(relevance.values(), reverse=True)[:k], start=1))
    ideal_dcg = _dcg(ideal)
    if ideal_dcg == 0.0:
        return 0.0
    return _dcg(actual) / ideal_dcg


def hard_negative_false_positive(run: RetrievalRun) -> bool:
    return any(
        evidence.score >= run.confidence_threshold for evidence in run.results
    )


def evaluate_run(
    case: EvalCase, qrels: Sequence[Qrel], run: RetrievalRun
) -> dict[str, str | float | bool]:
    if case.query_id != run.query_id:
        raise ValueError("case and run query IDs must match")
    return {
        "query_id": case.query_id,
        "recall_at_10": recall_at_k(qrels, run, 10),
        "recall_at_50": recall_at_k(qrels, run, 50),
        "mrr_at_10": mrr_at_k(qrels, run, 10),
        "ndcg_at_10": ndcg_at_k(qrels, run, 10),
        "precision_at_5": precision_at_k(qrels, run, 5),
        "hard_negative_false_positive": (
            case.is_hard_negative and hard_negative_false_positive(run)
        ),
    }
