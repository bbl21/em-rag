#!/usr/bin/env python3
"""Validate a completed human-review export and materialize calibration artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


JUDGMENT_FIELDS = (
    "judgment_id",
    "relevance",
    "supported_facets",
    "scope_correct",
    "citation_supported",
    "pollution",
    "confidence",
    "source_quote",
    "reason",
)
QUEUE_IDENTITY_FIELDS = (
    "judgment_id",
    "query",
    "expected_facets",
    "chunk_id",
    "source_id",
    "citation",
    "text",
)
QUEUE_REQUIRED_FIELDS = set(QUEUE_IDENTITY_FIELDS) | {
    "adjudication",
    "category",
    "pass1",
    "pass2",
    "query_id",
    "split",
}
COMPLETED_REQUIRED_FIELDS = set(QUEUE_IDENTITY_FIELDS) | {
    "full_evidence_text",
    "human_judgment",
}
QREL_FIELDS = (
    "query_id",
    "chunk_id",
    "relevance",
    "supported_facets",
    "confidence",
    "judgment_source",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True, type=Path)
    parser.add_argument("--completed", required=True, type=Path)
    parser.add_argument("--canonical-corpus", required=True, type=Path)
    parser.add_argument("--release-status", required=True, type=Path)
    parser.add_argument("--judgments-output", required=True, type=Path)
    parser.add_argument("--qrels-output", required=True, type=Path)
    parser.add_argument("--report-json", required=True, type=Path)
    parser.add_argument("--report-md", required=True, type=Path)
    return parser.parse_args(argv)


def _strict_object(
    value: Any,
    *,
    required: set[str],
    label: str,
    allow_extra: bool = False,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    missing = required - value.keys()
    if missing:
        raise ValueError(f"{label} missing fields: {', '.join(sorted(missing))}")
    if not allow_extra:
        extra = value.keys() - required
        if extra:
            raise ValueError(f"{label} unexpected fields: {', '.join(sorted(extra))}")
    return value


def _nonblank(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _string_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array of strings")
    result = [_nonblank(item, label) for item in value]
    if len(result) != len(set(result)):
        raise ValueError(f"{label} must not contain duplicates")
    return result


def _finite_confidence(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be between 0 and 1")
    return result


def validate_judgment(value: Any, *, label: str) -> dict[str, Any]:
    row = _strict_object(value, required=set(JUDGMENT_FIELDS), label=label)
    judgment_id = _nonblank(row["judgment_id"], f"{label}.judgment_id")
    relevance = row["relevance"]
    if type(relevance) is not int or relevance not in {0, 1, 2, 3}:
        raise ValueError(f"{label}.relevance must be an integer from 0 to 3")
    facets = _string_list(row["supported_facets"], f"{label}.supported_facets")
    booleans: dict[str, bool] = {}
    for field in ("scope_correct", "citation_supported", "pollution"):
        if type(row[field]) is not bool:
            raise ValueError(f"{label}.{field} must be a boolean")
        booleans[field] = row[field]
    return {
        "judgment_id": judgment_id,
        "relevance": relevance,
        "supported_facets": facets,
        **booleans,
        "confidence": _finite_confidence(row["confidence"], f"{label}.confidence"),
        "source_quote": _nonblank(row["source_quote"], f"{label}.source_quote"),
        "reason": _nonblank(row["reason"], f"{label}.reason"),
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{path.name}:{line_number} is blank")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path.name}:{line_number}: {error.msg}") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path.name}:{line_number} must be an object")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path.name} must not be empty")
    return rows


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain an object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary).replace(path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    text = "".join(
        json.dumps(dict(row), ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
        for row in rows
    )
    _atomic_write(path, text)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write(
        path,
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def _index_unique(rows: Iterable[Mapping[str, Any]], *, label: str) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row_number, row in enumerate(rows, start=1):
        judgment_id = _nonblank(row.get("judgment_id"), f"{label}[{row_number}].judgment_id")
        if judgment_id in indexed:
            raise ValueError(f"{label} contains duplicate judgment_id {judgment_id}")
        indexed[judgment_id] = row
    return indexed


def _load_canonical_chunks(path: Path, chunk_ids: set[str]) -> dict[str, Mapping[str, Any]]:
    chunks: dict[str, Mapping[str, Any]] = {}
    for row in read_jsonl(path):
        chunk_id = row.get("chunk_id")
        if chunk_id not in chunk_ids:
            continue
        if chunk_id in chunks:
            raise ValueError(f"canonical corpus contains duplicate chunk_id {chunk_id}")
        chunks[chunk_id] = row
    missing = chunk_ids - chunks.keys()
    if missing:
        raise ValueError(f"canonical corpus is missing {len(missing)} reviewed chunks")
    return chunks


def _quadratic_weighted_kappa(left: list[int], right: list[int]) -> float:
    if len(left) != len(right):
        raise ValueError("kappa inputs must have equal length")
    if not left:
        return 1.0
    total = len(left)
    counts_left = [left.count(grade) for grade in range(4)]
    counts_right = [right.count(grade) for grade in range(4)]
    observed = sum((a - b) ** 2 for a, b in zip(left, right)) / (9 * total)
    expected = sum(
        ((a - b) ** 2 / 9) * counts_left[a] * counts_right[b]
        for a in range(4)
        for b in range(4)
    ) / (total * total)
    if expected == 0.0:
        return 1.0 if observed == 0.0 else 0.0
    return 1.0 - observed / expected


def _agreement(human: list[int], agent: list[int]) -> dict[str, Any]:
    return {
        "exact_relevance_agreement": sum(a == b for a, b in zip(human, agent)) / len(human),
        "quadratic_weighted_kappa": _quadratic_weighted_kappa(human, agent),
    }


def _validate_release_identity(
    release_status: Mapping[str, Any], queue_path: Path, queue_count: int
) -> int:
    calibration = release_status.get("calibration_queue")
    if not isinstance(calibration, Mapping):
        raise ValueError("release status lacks calibration_queue metadata")
    sample_size = calibration.get("sample_size")
    population = calibration.get("population")
    expected_sha = calibration.get("sha256")
    if type(sample_size) is not int or sample_size != queue_count:
        raise ValueError("queue count does not match release-status sample_size")
    if type(population) is not int or population < queue_count:
        raise ValueError("release-status calibration population is invalid")
    if not isinstance(expected_sha, str) or sha256_file(queue_path) != expected_sha:
        raise ValueError("queue SHA-256 does not match release status")
    return population


def import_calibration(
    *,
    queue_path: Path,
    completed_path: Path,
    canonical_corpus_path: Path,
    release_status_path: Path,
    judgments_output: Path,
    qrels_output: Path,
    report_json: Path,
    report_md: Path,
) -> dict[str, Any]:
    queue_rows = read_jsonl(queue_path)
    completed_rows = read_jsonl(completed_path)
    release_status = read_json(release_status_path)
    population = _validate_release_identity(release_status, queue_path, len(queue_rows))

    queue_by_id = _index_unique(queue_rows, label="queue")
    completed_by_id = _index_unique(completed_rows, label="completed")
    if queue_by_id.keys() != completed_by_id.keys():
        missing = len(queue_by_id.keys() - completed_by_id.keys())
        extra = len(completed_by_id.keys() - queue_by_id.keys())
        raise ValueError(f"completed export identity mismatch: missing={missing}, extra={extra}")

    chunks = _load_canonical_chunks(
        canonical_corpus_path,
        {str(row["chunk_id"]) for row in queue_rows},
    )
    flat_judgments: list[dict[str, Any]] = []
    qrels: list[dict[str, Any]] = []
    pass1_relevance: list[int] = []
    pass2_relevance: list[int] = []
    human_relevance: list[int] = []
    confidence_values: list[float] = []
    split_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()

    for judgment_id in sorted(queue_by_id):
        queue = _strict_object(
            queue_by_id[judgment_id],
            required=QUEUE_REQUIRED_FIELDS,
            label=f"queue[{judgment_id}]",
        )
        completed = _strict_object(
            completed_by_id[judgment_id],
            required=COMPLETED_REQUIRED_FIELDS,
            label=f"completed[{judgment_id}]",
            allow_extra=True,
        )
        for field in QUEUE_IDENTITY_FIELDS:
            if completed[field] != queue[field]:
                raise ValueError(f"completed[{judgment_id}].{field} does not match queue")
        if queue["adjudication"] is not None:
            raise ValueError(f"queue[{judgment_id}] is not an unresolved calibration row")

        expected_facets = _string_list(queue["expected_facets"], f"queue[{judgment_id}].expected_facets")
        if not isinstance(completed["human_judgment"], Mapping):
            raise ValueError(f"completed[{judgment_id}].human_judgment must be an object")
        if "judgment_id" in completed["human_judgment"]:
            raise ValueError(
                f"completed[{judgment_id}].human_judgment must not override judgment_id"
            )
        human = validate_judgment(
            {"judgment_id": judgment_id, **completed["human_judgment"]},
            label=f"completed[{judgment_id}].human_judgment",
        )
        first = validate_judgment(queue["pass1"], label=f"queue[{judgment_id}].pass1")
        second = validate_judgment(queue["pass2"], label=f"queue[{judgment_id}].pass2")
        if not set(human["supported_facets"]).issubset(expected_facets):
            raise ValueError(f"completed[{judgment_id}] contains an unsupported facet")

        full_evidence = _nonblank(
            completed["full_evidence_text"],
            f"completed[{judgment_id}].full_evidence_text",
        )
        canonical = chunks[str(queue["chunk_id"])]
        if canonical.get("source_id") != queue["source_id"]:
            raise ValueError(f"canonical source mismatch for {judgment_id}")
        if canonical.get("content_md") != full_evidence:
            raise ValueError(f"authoritative evidence mismatch for {judgment_id}")
        if human["source_quote"] not in full_evidence:
            raise ValueError(f"source_quote is not verbatim canonical evidence for {judgment_id}")

        flat_judgments.append(human)
        qrels.append(
            {
                "query_id": _nonblank(queue["query_id"], f"queue[{judgment_id}].query_id"),
                "chunk_id": _nonblank(queue["chunk_id"], f"queue[{judgment_id}].chunk_id"),
                "relevance": human["relevance"],
                "supported_facets": human["supported_facets"],
                "confidence": human["confidence"],
                "judgment_source": "human_calibration",
            }
        )
        human_relevance.append(human["relevance"])
        pass1_relevance.append(first["relevance"])
        pass2_relevance.append(second["relevance"])
        confidence_values.append(human["confidence"])
        split_counts[_nonblank(queue["split"], f"queue[{judgment_id}].split")] += 1
        category_counts[_nonblank(queue["category"], f"queue[{judgment_id}].category")] += 1
        source_counts[_nonblank(queue["source_id"], f"queue[{judgment_id}].source_id")] += 1

    qrels.sort(key=lambda row: (row["query_id"], row["chunk_id"]))
    write_jsonl(judgments_output, flat_judgments)
    write_jsonl(qrels_output, qrels)
    imported_count = len(flat_judgments)
    remaining_unresolved = population - imported_count
    report = {
        "agreement": {
            "human_vs_pass1": _agreement(human_relevance, pass1_relevance),
            "human_vs_pass2": _agreement(human_relevance, pass2_relevance),
        },
        "artifact_policy": {
            "aggregate_only": True,
            "contains_query_ids": False,
            "contains_query_or_evidence_text": False,
            "controlled_outputs": ["human calibration judgments", "human gold qrels"],
        },
        "calibration": {
            "completed_count": imported_count,
            "completion_rate": imported_count / len(queue_rows),
            "population": population,
            "remaining_unresolved_count": remaining_unresolved,
            "sample_count": len(queue_rows),
            "sample_fraction": imported_count / population,
        },
        "counts": {
            "category": dict(sorted(category_counts.items())),
            "human_relevance": {str(grade): human_relevance.count(grade) for grade in range(4)},
            "source": dict(sorted(source_counts.items())),
            "split": dict(sorted(split_counts.items())),
        },
        "human_confidence": {
            "average": sum(confidence_values) / len(confidence_values),
            "maximum": max(confidence_values),
            "minimum": min(confidence_values),
        },
        "input_sha256": {
            "canonical_corpus": sha256_file(canonical_corpus_path),
            "completed_export": sha256_file(completed_path),
            "queue": sha256_file(queue_path),
            "release_status": sha256_file(release_status_path),
        },
        "output_sha256": {
            "human_calibration_judgments": sha256_file(judgments_output),
            "human_gold_qrels": sha256_file(qrels_output),
        },
        "release_eligible": remaining_unresolved == 0,
        "schema_version": "retrieval_quality_v2_human_calibration_v1",
        "status": "PASS" if remaining_unresolved == 0 else "NEEDS_CALIBRATION",
        "validation": {
            "canonical_evidence_exact_match_count": imported_count,
            "completed_id_match_count": imported_count,
            "facet_subset_valid_count": imported_count,
            "quote_verbatim_count": imported_count,
            "strict_judgment_schema_count": imported_count,
        },
    }
    write_json(report_json, report)
    markdown = "\n".join(
        [
            "# Human calibration status",
            "",
            f"- Status: `{report['status']}`",
            f"- Completed review rows: {imported_count}/{len(queue_rows)}",
            f"- Human gold qrels: {len(qrels)}",
            f"- Remaining unresolved judgments: {remaining_unresolved}",
            f"- Human vs pass 1 exact relevance agreement: {report['agreement']['human_vs_pass1']['exact_relevance_agreement']}",
            f"- Human vs pass 1 weighted kappa: {report['agreement']['human_vs_pass1']['quadratic_weighted_kappa']}",
            f"- Human vs pass 2 exact relevance agreement: {report['agreement']['human_vs_pass2']['exact_relevance_agreement']}",
            f"- Human vs pass 2 weighted kappa: {report['agreement']['human_vs_pass2']['quadratic_weighted_kappa']}",
            "",
            "All completed rows matched the declared queue, canonical chunk text, expected facets, and verbatim quote rule. Detailed judgments and qrels remain controlled local artifacts.",
            "",
        ]
    )
    _atomic_write(report_md, markdown)
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = import_calibration(
            queue_path=args.queue,
            completed_path=args.completed,
            canonical_corpus_path=args.canonical_corpus,
            release_status_path=args.release_status,
            judgments_output=args.judgments_output,
            qrels_output=args.qrels_output,
            report_json=args.report_json,
            report_md=args.report_md,
        )
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"human calibration import failed: {error}", file=os.sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "completed_count": report["calibration"]["completed_count"],
                "remaining_unresolved_count": report["calibration"]["remaining_unresolved_count"],
                "status": report["status"],
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
