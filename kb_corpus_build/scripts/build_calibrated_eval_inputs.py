#!/usr/bin/env python3
"""Materialize resolved qrels and mechanical findings after judgment calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from em_rag.evaluation.mechanical import inspect_run
from em_rag.evaluation.models import EvalCase, Qrel, RetrievalRun
from em_rag.evaluation.pooling import Judgment


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", required=True, type=Path)
    parser.add_argument("--merged", required=True, type=Path)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--docstore", required=True, type=Path)
    parser.add_argument("--qrels-output", required=True, type=Path)
    parser.add_argument("--findings-output", required=True, type=Path)
    parser.add_argument("--summary-output", required=True, type=Path)
    return parser.parse_args(argv)


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except json.JSONDecodeError as error:
        raise ValueError(f"{path.name} is not valid JSON: {error.msg}") from error


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{path.name}:{line_number} is blank")
            try:
                value = json.loads(line, parse_constant=_reject_constant)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path.name}:{line_number}: {error.msg}") from error
            if not isinstance(value, dict):
                raise ValueError(f"{path.name}:{line_number} must be an object")
            rows.append(value)
    if not rows:
        raise ValueError(f"{path.name} must not be empty")
    return rows


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
    _atomic_write(
        path,
        "".join(
            json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
            for row in rows
        ),
    )


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write(
        path,
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _eval_case(row: Mapping[str, Any]) -> EvalCase:
    return EvalCase.from_dict(
        {
            "query_id": row.get("query_id"),
            "query": row.get("query"),
            "category": row.get("category"),
            "expected_facets": row.get("expected_facets", row.get("expected_evidence_facets", [])),
            "is_hard_negative": row.get("is_hard_negative"),
            "requires_multiple_evidence": row.get(
                "requires_multiple_evidence",
                row.get("requires_multi_citation", False),
            ),
            "split": row.get("split"),
            "language": row.get("language", "en"),
        }
    )


def _judgment(value: Any, label: str) -> Judgment | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object or null")
    return Judgment.from_dict(value)


def _substantively_equal(left: Judgment, right: Judgment) -> bool:
    return (
        left.relevance == right.relevance
        and left.supported_facets == right.supported_facets
        and left.scope_correct == right.scope_correct
        and left.citation_supported == right.citation_supported
        and left.pollution == right.pollution
    )


def _index_pool(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        judgment_id = row.get("judgment_id")
        if not isinstance(judgment_id, str) or not judgment_id:
            raise ValueError("pool judgment_id must be a non-empty string")
        if judgment_id in indexed:
            raise ValueError(f"pool contains duplicate judgment_id {judgment_id}")
        query_id = row.get("query_id")
        chunk_id = row.get("chunk_id")
        if not isinstance(query_id, str) or not query_id or not isinstance(chunk_id, str) or not chunk_id:
            raise ValueError(f"pool identity is invalid for {judgment_id}")
        indexed[judgment_id] = row
    return indexed


def build_qrels(pool_rows: list[dict[str, Any]], merged: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    pool = _index_pool(pool_rows)
    merged_rows = merged.get("rows")
    if not isinstance(merged_rows, list):
        raise ValueError("merged judgment report must contain a rows array")
    qrels: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    seen_ids: set[str] = set()
    for row_number, row in enumerate(merged_rows, start=1):
        if not isinstance(row, Mapping):
            raise ValueError(f"merged row {row_number} must be an object")
        judgment_id = row.get("judgment_id")
        if not isinstance(judgment_id, str) or judgment_id not in pool:
            raise ValueError(f"merged row {row_number} has an unknown judgment_id")
        if judgment_id in seen_ids:
            raise ValueError(f"merged rows contain duplicate judgment_id {judgment_id}")
        seen_ids.add(judgment_id)
        final = _judgment(row.get("final"), f"merged[{judgment_id}].final")
        if final is None:
            continue
        human = _judgment(row.get("human_calibration"), f"merged[{judgment_id}].human_calibration")
        adjudication = _judgment(row.get("adjudication"), f"merged[{judgment_id}].adjudication")
        first = _judgment(row.get("pass1"), f"merged[{judgment_id}].pass1")
        second = _judgment(row.get("pass2"), f"merged[{judgment_id}].pass2")
        if first is None or second is None:
            raise ValueError(f"merged[{judgment_id}] lacks a judgment pass")
        if human is not None:
            source = "human_calibration"
            expected_final = human
        elif adjudication is not None:
            source = "agent_adjudication"
            expected_final = adjudication
        else:
            source = "agent_pass_1"
            expected_final = first
            if not _substantively_equal(first, second):
                raise ValueError(
                    f"merged[{judgment_id}] uses pass 1 without substantive pass agreement"
                )
        if final != expected_final:
            raise ValueError(f"merged[{judgment_id}] final judgment has inconsistent provenance")
        identity = pool[judgment_id]
        qrel = Qrel.from_dict(
            {
                "query_id": identity["query_id"],
                "chunk_id": identity["chunk_id"],
                "relevance": final.relevance,
                "supported_facets": list(final.supported_facets),
                "confidence": final.confidence,
                "judgment_source": source,
            }
        )
        qrels.append(asdict(qrel))
        source_counts[source] += 1
    if seen_ids != pool.keys():
        raise ValueError("merged judgment IDs do not match the blind pool")
    qrels.sort(key=lambda row: (row["query_id"], row["chunk_id"]))
    if len({(row["query_id"], row["chunk_id"]) for row in qrels}) != len(qrels):
        raise ValueError("resolved qrels contain duplicate query/chunk pairs")
    return qrels, dict(sorted(source_counts.items()))


def build_findings(
    case_rows: list[dict[str, Any]],
    run_rows: list[dict[str, Any]],
    docstore_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    cases: dict[str, EvalCase] = {}
    for row in case_rows:
        case = _eval_case(row)
        if case.query_id in cases:
            raise ValueError(f"cases contain duplicate query_id {case.query_id}")
        cases[case.query_id] = case
    runs: dict[str, RetrievalRun] = {}
    for row in run_rows:
        run = RetrievalRun.from_dict(row)
        if run.query_id in runs:
            raise ValueError(f"runs contain duplicate query_id {run.query_id}")
        runs[run.query_id] = run
    if cases.keys() != runs.keys():
        raise ValueError("cases and runs must contain identical query IDs")
    docstore: dict[str, dict[str, Any]] = {}
    for row in docstore_rows:
        chunk_id = row.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id:
            raise ValueError("docstore chunk_id must be a non-empty string")
        if chunk_id in docstore:
            raise ValueError(f"docstore contains duplicate chunk_id {chunk_id}")
        docstore[chunk_id] = {
            "citation": row.get("citation"),
            "group_id": row.get("duplicate_group_id"),
            "text": row.get("content_md"),
        }
    findings = [
        asdict(finding)
        for query_id in sorted(cases)
        for finding in inspect_run(cases[query_id], runs[query_id], docstore)
    ]
    return findings


def materialize(
    *,
    pool_path: Path,
    merged_path: Path,
    cases_path: Path,
    runs_path: Path,
    docstore_path: Path,
    qrels_output: Path,
    findings_output: Path,
    summary_output: Path,
) -> dict[str, Any]:
    merged = read_json(merged_path)
    if not isinstance(merged, Mapping):
        raise ValueError("merged judgment report must be an object")
    qrels, source_counts = build_qrels(read_jsonl(pool_path), merged)
    findings = build_findings(
        read_jsonl(cases_path),
        read_jsonl(runs_path),
        read_jsonl(docstore_path),
    )
    write_jsonl(qrels_output, qrels)
    write_jsonl(findings_output, findings)
    code_counts = Counter(str(row["code"]) for row in findings)
    severity_counts = Counter(str(row["severity"]) for row in findings)
    agreement = merged.get("agreement")
    if not isinstance(agreement, Mapping):
        raise ValueError("merged judgment report lacks agreement metadata")
    summary = {
        "artifact_policy": {
            "aggregate_only": True,
            "contains_query_ids": False,
            "contains_query_or_evidence_text": False,
            "controlled_outputs": ["resolved qrels", "per-finding mechanical details"],
        },
        "judgments": {
            "release_eligible": bool(agreement.get("release_eligible")),
            "resolved_qrel_count": len(qrels),
            "source_counts": source_counts,
            "unresolved_count": len(agreement.get("unresolved_ids", [])),
        },
        "mechanical": {
            "code_counts": dict(sorted(code_counts.items())),
            "finding_count": len(findings),
            "severity_counts": dict(sorted(severity_counts.items())),
        },
        "output_sha256": {
            "findings": sha256_file(findings_output),
            "qrels": sha256_file(qrels_output),
        },
        "schema_version": "retrieval_quality_v2_calibrated_inputs_v1",
        "status": "PASS" if bool(agreement.get("release_eligible")) else "NEEDS_CALIBRATION",
    }
    write_json(summary_output, summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = materialize(
            pool_path=args.pool,
            merged_path=args.merged,
            cases_path=args.cases,
            runs_path=args.runs,
            docstore_path=args.docstore,
            qrels_output=args.qrels_output,
            findings_output=args.findings_output,
            summary_output=args.summary_output,
        )
    except (FileNotFoundError, OSError, ValueError) as error:
        print(f"calibrated evaluation input build failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
