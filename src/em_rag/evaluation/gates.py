"""Deterministic aggregate and paired retrieval quality release gates."""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from decimal import Decimal
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from typing import Any

from .mechanical import BLOCKING_CODES, WARNING_CODES, MechanicalFinding
from .metrics import evaluate_run, hard_negative_false_positive
from .models import EvalCase, Qrel, RetrievalRun


QUALITY_STATUSES = frozenset({"PASS", "FAIL", "NEEDS_CALIBRATION"})
RETRIEVAL_METRICS = (
    "recall_at_10",
    "recall_at_50",
    "mrr_at_10",
    "ndcg_at_10",
    "precision_at_5",
)
RELEASE_METRICS = (
    "recall_at_10",
    "recall_at_50",
    "mrr_at_10",
    "ndcg_at_10",
)
BOOTSTRAP_SEED = 1729
BOOTSTRAP_RESAMPLES = 10_000
_CITATION_BLOCKING_CODES = frozenset({"missing_chunk", "citation_mismatch"})
_FIXED_POLICY_THRESHOLDS = {
    "recall_at_10": 0.90,
    "recall_at_50": 0.95,
    "mrr_at_10": 0.80,
    "ndcg_at_10": 0.80,
    "citation_validity": 1.0,
    "hard_negative_false_positive_rate": 0.05,
    "weighted_cohen_kappa": 0.8,
    "max_category_regression": 0.05,
}


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    raise ValueError(f"{field} must be an object")


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    total = sum((Decimal(str(value)) for value in values), start=Decimal("0"))
    return float(total / Decimal(len(values)))


def _decimal_delta(candidate: float, baseline: float) -> float:
    return float(Decimal(str(candidate)) - Decimal(str(baseline)))


def _category_regressed(delta: float, maximum: float) -> bool:
    return Decimal(str(delta)) < -Decimal(str(maximum))


def _finite_unit_interval(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    converted = float(value)
    if not math.isfinite(converted) or not 0.0 <= converted <= 1.0:
        raise ValueError(f"{field} must be between 0 and 1")
    return converted


def _thresholds(values: Mapping[str, float]) -> dict[str, float]:
    if not isinstance(values, Mapping):
        raise ValueError("thresholds must be an object")
    missing = set(RELEASE_METRICS) - values.keys()
    if missing:
        raise ValueError(
            f"missing retrieval thresholds: {', '.join(sorted(missing))}"
        )
    unknown = values.keys() - _FIXED_POLICY_THRESHOLDS.keys()
    if unknown:
        raise ValueError(f"unsupported thresholds: {', '.join(sorted(unknown))}")
    merged = {**_FIXED_POLICY_THRESHOLDS, **values}
    normalized = {
        key: _finite_unit_interval(value, f"thresholds.{key}")
        for key, value in merged.items()
    }
    for key, fixed in _FIXED_POLICY_THRESHOLDS.items():
        if Decimal(str(normalized[key])) != Decimal(str(fixed)):
            raise ValueError(f"{key} is a fixed policy threshold")
    return normalized


def _validate_inputs(
    cases: Sequence[EvalCase],
    qrels: Sequence[Qrel],
    runs: Sequence[RetrievalRun],
) -> tuple[dict[str, EvalCase], dict[str, RetrievalRun], list[str]]:
    if not all(isinstance(case, EvalCase) for case in cases):
        raise ValueError("cases must contain EvalCase values")
    if not all(isinstance(qrel, Qrel) for qrel in qrels):
        raise ValueError("qrels must contain Qrel values")
    if not all(isinstance(run, RetrievalRun) for run in runs):
        raise ValueError("runs must contain RetrievalRun values")
    case_by_id = {case.query_id: case for case in cases}
    run_by_query = {run.query_id: run for run in runs}
    if len(case_by_id) != len(cases):
        raise ValueError("cases must not contain duplicate query IDs")
    if len(run_by_query) != len(runs):
        raise ValueError("runs must contain exactly one row per query ID")
    if case_by_id.keys() != run_by_query.keys():
        raise ValueError("cases and runs must contain the same query IDs")
    qrel_keys = [(qrel.query_id, qrel.chunk_id) for qrel in qrels]
    if len(qrel_keys) != len(set(qrel_keys)):
        raise ValueError("qrels must not contain duplicate query/chunk pairs")
    unknown_qrels = sorted({qrel.query_id for qrel in qrels} - case_by_id.keys())
    if unknown_qrels:
        raise ValueError("qrels contain unknown query IDs")
    judged_ids = {qrel.query_id for qrel in qrels}
    uncalibrated = sorted(
        case.query_id
        for case in cases
        if not case.is_hard_negative and case.query_id not in judged_ids
    )
    return case_by_id, run_by_query, uncalibrated


def aggregate_quality_gate(
    cases: Sequence[EvalCase],
    qrels: Sequence[Qrel],
    runs: Sequence[RetrievalRun],
    findings: Sequence[MechanicalFinding | Mapping[str, Any]],
    agreement: Mapping[str, Any] | Any,
    thresholds: Mapping[str, float],
    *,
    baseline_category_metrics: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Aggregate metrics without claiming retrieval quality before calibration."""

    case_by_id, run_by_query, uncalibrated = _validate_inputs(cases, qrels, runs)
    configured = _thresholds(thresholds)
    normalized_agreement = _mapping(agreement, "agreement")
    kappa = _finite_unit_interval(
        normalized_agreement.get("weighted_cohen_kappa"),
        "agreement.weighted_cohen_kappa",
    )
    if type(normalized_agreement.get("release_eligible")) is not bool:
        raise ValueError("agreement.release_eligible must be a boolean")
    unresolved = normalized_agreement.get("unresolved_ids", [])
    if not isinstance(unresolved, (list, tuple)) or not all(
        isinstance(item, str) and item for item in unresolved
    ):
        raise ValueError("agreement.unresolved_ids must be an array of IDs")
    agreement_ready = (
        normalized_agreement["release_eligible"]
        and kappa >= configured["weighted_cohen_kappa"]
        and not unresolved
    )

    normalized_findings = [_mapping(finding, "finding") for finding in findings]
    for finding in normalized_findings:
        if finding.get("query_id") not in case_by_id:
            raise ValueError("finding contains an unknown query ID")
        if finding.get("severity") not in {"blocking", "warning"}:
            raise ValueError("finding severity must be blocking or warning")
        code = finding.get("code")
        if not isinstance(code, str) or not code:
            raise ValueError("finding code must be a non-empty string")
        if code in BLOCKING_CODES:
            canonical_severity = "blocking"
        elif code in WARNING_CODES:
            canonical_severity = "warning"
        else:
            raise ValueError(f"unsupported mechanical finding code: {code}")
        if finding["severity"] != canonical_severity:
            raise ValueError(
                f"{code} must use canonical severity {canonical_severity}"
            )
    blocking_counts = Counter(
        str(finding["code"])
        for finding in normalized_findings
        if finding["severity"] == "blocking"
    )
    invalid_citations = {
        (str(finding["query_id"]), finding.get("chunk_id"))
        for finding in normalized_findings
        if finding["code"] in _CITATION_BLOCKING_CODES
    }
    evidence_count = sum(len(run.results) for run in runs)
    citation_validity = (
        max(0.0, 1.0 - len(invalid_citations) / evidence_count)
        if evidence_count
        else 1.0
    )
    hard_negative_count = sum(
        case.is_hard_negative
        and hard_negative_false_positive(run_by_query[case.query_id])
        for case in cases
    )
    hard_negative_total = sum(case.is_hard_negative for case in cases)
    hard_negative_rate = (
        hard_negative_count / hard_negative_total if hard_negative_total else 0.0
    )

    metrics_available = agreement_ready and not uncalibrated
    per_query: list[dict[str, Any]] = []
    category_metrics: dict[str, float] = {}
    category_deltas: dict[str, float] = {}
    retrieval_values: dict[str, float | None]
    if metrics_available:
        by_category: dict[str, list[float]] = defaultdict(list)
        for query_id in sorted(case_by_id):
            case = case_by_id[query_id]
            record = evaluate_run(case, qrels, run_by_query[query_id])
            record["category"] = case.category
            per_query.append(record)
            if not case.is_hard_negative:
                by_category[case.category].append(float(record["ndcg_at_10"]))
        answerable = [
            record
            for record in per_query
            if not case_by_id[str(record["query_id"])].is_hard_negative
        ]
        retrieval_values = {
            metric: _mean([float(record[metric]) for record in answerable])
            for metric in RETRIEVAL_METRICS
        }
        category_metrics = {
            category: _mean(values) for category, values in sorted(by_category.items())
        }
        baseline = dict(baseline_category_metrics or {})
        category_deltas = {
            category: _decimal_delta(
                value,
                _finite_unit_interval(baseline[category], f"baseline.{category}"),
            )
            for category, value in category_metrics.items()
            if category in baseline
        }
    else:
        retrieval_values = {metric: None for metric in RETRIEVAL_METRICS}

    metrics = {
        **retrieval_values,
        "citation_validity": citation_validity,
        "hard_negative_false_positive_rate": hard_negative_rate,
    }
    metric_checks = {
        metric: {
            "value": metrics[metric],
            "threshold": configured[metric],
            "passed": (
                None
                if metrics[metric] is None
                else (
                    metrics[metric] <= configured[metric]
                    if metric == "hard_negative_false_positive_rate"
                    else metrics[metric] >= configured[metric]
                )
            ),
        }
        for metric in (
            *RELEASE_METRICS,
            "citation_validity",
            "hard_negative_false_positive_rate",
        )
    }
    failures: list[str] = []
    if uncalibrated:
        failures.append("qrels_not_release_eligible")
    if not agreement_ready:
        failures.append("judgment_agreement_not_release_eligible")
    if metrics_available:
        failures.extend(
            f"metric_below_threshold:{metric}"
            for metric in RELEASE_METRICS
            if metric_checks[metric]["passed"] is False
        )
        failures.extend(
            f"category_regression:{category}"
            for category, delta in sorted(category_deltas.items())
            if _category_regressed(
                delta, configured["max_category_regression"]
            )
        )
    if metric_checks["citation_validity"]["passed"] is False:
        failures.append("citation_validity_below_threshold")
    if metric_checks["hard_negative_false_positive_rate"]["passed"] is False:
        failures.append("hard_negative_false_positive")
    failures.extend(
        f"mechanical_blocking:{code}"
        for code in sorted(blocking_counts)
        if code != "hard_negative_false_positive"
    )

    if not metrics_available:
        status = "NEEDS_CALIBRATION"
    elif failures:
        status = "FAIL"
    else:
        status = "PASS"
    if status not in QUALITY_STATUSES:
        raise AssertionError("quality gate emitted an unstable status")

    return {
        "status": status,
        "metrics_available": metrics_available,
        "metrics": metrics,
        "thresholds": dict(sorted(configured.items())),
        "metric_checks": metric_checks,
        "category_metrics": category_metrics,
        "category_deltas": category_deltas,
        "mechanical_blocking_counts": dict(sorted(blocking_counts.items())),
        "hard_negative_false_positives": hard_negative_count,
        "hard_negative_false_positive_rate": hard_negative_rate,
        "judgment_agreement": normalized_agreement,
        "uncalibrated_query_ids": uncalibrated,
        "gate_failures": failures,
        "artifact_ids": sorted({run.artifact_id for run in runs}),
        "run_ids": sorted({run.run_id for run in runs}),
        "per_query": per_query,
    }


def paired_bootstrap_ndcg(
    baseline: Sequence[float], candidate: Sequence[float]
) -> dict[str, int | float | str]:
    """Return a deterministic paired 95 percent bootstrap interval for nDCG delta."""

    if len(baseline) != len(candidate) or not baseline:
        raise ValueError("paired nDCG inputs must be non-empty and equal length")
    left = [_finite_unit_interval(value, "baseline nDCG") for value in baseline]
    right = [_finite_unit_interval(value, "candidate nDCG") for value in candidate]
    deltas = [new - old for old, new in zip(left, right)]
    random_source = random.Random(BOOTSTRAP_SEED)
    count = len(deltas)
    samples = [
        sum(deltas[random_source.randrange(count)] for _ in range(count)) / count
        for _ in range(BOOTSTRAP_RESAMPLES)
    ]
    samples.sort()
    lower = samples[int(0.025 * BOOTSTRAP_RESAMPLES)]
    upper = samples[int(0.975 * BOOTSTRAP_RESAMPLES) - 1]
    return {
        "seed": BOOTSTRAP_SEED,
        "resamples": BOOTSTRAP_RESAMPLES,
        "mean_delta": _mean(deltas),
        "lower_bound": lower,
        "upper_bound": upper,
        "improvement": (
            "confirmed_improvement"
            if lower > 0.0
            else "no_confirmed_improvement"
        ),
    }


def _metric_records(
    records: Sequence[Mapping[str, Any]], label: str
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError(f"{label} metric rows must be objects")
        query_id = record.get("query_id")
        category = record.get("category")
        if not isinstance(query_id, str) or not query_id:
            raise ValueError(f"{label} query_id must be a non-empty string")
        if not isinstance(category, str) or not category:
            raise ValueError(f"{label} category must be a non-empty string")
        if query_id in indexed:
            raise ValueError(f"{label} contains duplicate query IDs")
        indexed[query_id] = {
            **record,
            "ndcg_at_10": _finite_unit_interval(
                record.get("ndcg_at_10"), f"{label}.{query_id}.ndcg_at_10"
            ),
        }
    if not indexed:
        raise ValueError(f"{label} metrics must not be empty")
    return indexed


def compare_quality_runs(
    baseline: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
    *,
    max_category_regression: float = 0.05,
) -> dict[str, Any]:
    """Compare paired per-query metrics and block category regressions."""

    allowed_regression = _finite_unit_interval(
        max_category_regression, "max_category_regression"
    )
    fixed_regression = _FIXED_POLICY_THRESHOLDS["max_category_regression"]
    if Decimal(str(allowed_regression)) != Decimal(str(fixed_regression)):
        raise ValueError("max_category_regression is a fixed policy threshold")
    old = _metric_records(baseline, "baseline")
    new = _metric_records(candidate, "candidate")
    if old.keys() != new.keys():
        raise ValueError("baseline and candidate must contain the same query IDs")

    query_ids = sorted(old)
    category_deltas_by_name: dict[str, list[float]] = defaultdict(list)
    per_query_deltas: list[dict[str, str | float]] = []
    for query_id in query_ids:
        old_category = old[query_id]["category"]
        new_category = new[query_id]["category"]
        if old_category != new_category:
            raise ValueError("paired query categories must match")
        delta = _decimal_delta(
            new[query_id]["ndcg_at_10"], old[query_id]["ndcg_at_10"]
        )
        category_deltas_by_name[str(old_category)].append(delta)
        per_query_deltas.append(
            {
                "query_id": query_id,
                "category": str(old_category),
                "ndcg_at_10_delta": delta,
            }
        )

    category_deltas = {
        category: _mean(values)
        for category, values in sorted(category_deltas_by_name.items())
    }
    failures = [
        f"category_regression:{category}"
        for category, delta in category_deltas.items()
        if _category_regressed(delta, allowed_regression)
    ]
    paired = paired_bootstrap_ndcg(
        [old[query_id]["ndcg_at_10"] for query_id in query_ids],
        [new[query_id]["ndcg_at_10"] for query_id in query_ids],
    )
    return {
        "status": "FAIL" if failures else "PASS",
        "paired_query_ids": query_ids,
        "paired_ndcg": paired,
        "category_deltas": category_deltas,
        "per_query_deltas": per_query_deltas,
        "max_category_regression": allowed_regression,
        "gate_failures": failures,
    }
