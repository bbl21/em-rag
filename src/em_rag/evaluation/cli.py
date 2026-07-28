"""Command-line entry points for retrieval quality validation and release gates."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .gates import aggregate_quality_gate, compare_quality_runs
from .io import read_jsonl
from .models import EvalCase, Qrel, RetrievalRun
from .pooling import Judgment, build_blind_pool, merge_judgments


_COMMANDS = (
    "validate-dataset",
    "build-pool",
    "validate-judgments",
    "evaluate-run",
    "compare-runs",
)


def command_names() -> tuple[str, ...]:
    """Return the stable public CLI command names."""

    return _COMMANDS


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def _read_json(path: str | Path) -> Any:
    file_path = Path(path)
    try:
        text = file_path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{file_path} is not valid UTF-8") from error
    try:
        return json.loads(text, parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{file_path} is not valid strict JSON") from error


def _json_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ) + "\n"
        payload = text.encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise ValueError("output is not UTF-8 JSON serializable") from error
    if b"\r\n" in payload or payload.decode("utf-8") != text:
        raise ValueError("output failed UTF-8/LF validation")
    return payload


def _atomic_write_bytes(path: str | Path, payload: bytes) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    written = output.read_bytes()
    if written != payload:
        raise ValueError(f"disk byte mismatch after writing {output}")


def _atomic_write_json(path: str | Path, value: Any) -> None:
    _atomic_write_bytes(path, _json_bytes(value))


def _atomic_write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        json.dumps(
            dict(row),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        for row in rows
    ]
    payload = b"\n".join(lines)
    if lines:
        payload += b"\n"
    if b"\r\n" in payload:
        raise ValueError("CRLF output is not allowed")
    payload.decode("utf-8")
    _atomic_write_bytes(path, payload)


_ACTIVE_CASE_FIELDS = frozenset(
    {
        "query_id",
        "query",
        "language",
        "category",
        "expected_intent",
        "expected_sources",
        "expected_evidence_facets",
        "is_hard_negative",
        "requires_web_check",
        "requires_multi_citation",
        "notes",
        "provenance",
        "split",
    }
)
_ACTIVE_CASE_MARKERS = frozenset(
    {"expected_evidence_facets", "requires_multi_citation"}
)


def _eval_case_payload(row: Mapping[str, Any]) -> Mapping[str, Any]:
    if not _ACTIVE_CASE_MARKERS.intersection(row):
        return row
    missing_active_fields = _ACTIVE_CASE_FIELDS.difference(row)
    if missing_active_fields:
        missing = ", ".join(sorted(missing_active_fields))
        raise ValueError(f"active evaluation case is missing fields: {missing}")
    unknown_active_fields = row.keys() - _ACTIVE_CASE_FIELDS
    if unknown_active_fields:
        unknown = ", ".join(sorted(unknown_active_fields))
        raise ValueError(f"active evaluation case has unexpected fields: {unknown}")
    return {
        "query_id": row["query_id"],
        "query": row["query"],
        "category": row["category"],
        "expected_facets": row["expected_evidence_facets"],
        "is_hard_negative": row["is_hard_negative"],
        "requires_multiple_evidence": row["requires_multi_citation"],
        "split": row["split"],
        "language": row["language"],
    }


def _load_cases(path: str | Path) -> list[EvalCase]:
    return [EvalCase.from_dict(_eval_case_payload(row)) for row in read_jsonl(path)]


def _load_qrels(path: str | Path) -> list[Qrel]:
    return [Qrel.from_dict(row) for row in read_jsonl(path)]


def _load_runs(path: str | Path) -> list[RetrievalRun]:
    return [RetrievalRun.from_dict(row) for row in read_jsonl(path)]


def _validate_dataset_links(
    cases: Sequence[EvalCase], qrels: Sequence[Qrel]
) -> list[str]:
    case_ids = {case.query_id for case in cases}
    if len(case_ids) != len(cases):
        raise ValueError("cases must not contain duplicate query IDs")
    qrel_keys = {(qrel.query_id, qrel.chunk_id) for qrel in qrels}
    if len(qrel_keys) != len(qrels):
        raise ValueError("qrels must not contain duplicate query/chunk pairs")
    if {qrel.query_id for qrel in qrels} - case_ids:
        raise ValueError("qrels contain unknown query IDs")
    judged_ids = {qrel.query_id for qrel in qrels}
    return sorted(
        case.query_id
        for case in cases
        if not case.is_hard_negative and case.query_id not in judged_ids
    )


def _run_validate_dataset(args: argparse.Namespace) -> int:
    cases = _load_cases(args.cases)
    qrels = _load_qrels(args.qrels)
    if not cases:
        raise ValueError("evaluation dataset must contain at least one case")
    missing = _validate_dataset_links(cases, qrels)
    status = "NEEDS_CALIBRATION" if missing else "PASS"
    report = {
        "status": status,
        "case_count": len(cases),
        "qrel_count": len(qrels),
        "category_count": len({case.category for case in cases}),
        "split_counts": {
            split: sum(case.split == split for case in cases)
            for split in sorted({case.split for case in cases})
        },
        "missing_qrel_query_ids": missing,
    }
    _atomic_write_json(args.output, report)
    return 0 if status == "PASS" else 1


def _run_build_pool(args: argparse.Namespace) -> int:
    cases = _load_cases(args.cases)
    if not cases:
        raise ValueError("evaluation dataset must contain at least one case")
    case_by_id = {case.query_id: case for case in cases}
    grouped: dict[str, list[RetrievalRun]] = defaultdict(list)
    seen_runs: set[tuple[str, str]] = set()
    for path in args.runs:
        for run in _load_runs(path):
            if run.query_id not in case_by_id:
                raise ValueError("run contains an unknown query ID")
            identity = (run.run_id, run.query_id)
            if identity in seen_runs:
                raise ValueError("runs contain duplicate run/query identities")
            seen_runs.add(identity)
            grouped[run.query_id].append(run)
    missing = sorted(case_by_id.keys() - grouped.keys())
    if missing:
        raise ValueError("every evaluation case must have at least one run")
    rows = build_blind_pool(
        [
            (case_by_id[query_id], tuple(grouped[query_id]))
            for query_id in sorted(case_by_id)
        ],
        seed=args.seed,
    )
    _atomic_write_jsonl(args.output, [asdict(row) for row in rows])
    return 0


def _load_judgments(path: str | Path | None) -> list[Judgment]:
    if path is None:
        return []
    return [Judgment.from_dict(row) for row in read_jsonl(path)]


def _run_validate_judgments(args: argparse.Namespace) -> int:
    merged = merge_judgments(
        _load_judgments(args.pass1),
        _load_judgments(args.pass2),
        _load_judgments(args.adjudication),
        human_calibration=_load_judgments(args.human_calibration),
        sampled_ids=tuple(args.sampled_id),
    )
    status = "PASS" if merged.agreement.release_eligible else "NEEDS_CALIBRATION"
    report = {
        "status": status,
        "agreement": asdict(merged.agreement),
        "rows": [asdict(row) for row in merged.rows],
    }
    _atomic_write_json(args.output, report)
    return 0 if status == "PASS" else 1


def _agreement_object(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("agreement input must be an object")
    nested = value.get("agreement")
    if nested is not None:
        if not isinstance(nested, Mapping):
            raise ValueError("agreement.agreement must be an object")
        return nested
    return value


def _baseline_categories(path: str | Path | None) -> Mapping[str, float] | None:
    if path is None:
        return None
    value = _read_json(path)
    if not isinstance(value, Mapping):
        raise ValueError("baseline metrics must be an object")
    categories = value.get("category_breakdown", value.get("category_metrics"))
    if not isinstance(categories, Mapping):
        raise ValueError("baseline metrics must contain category breakdown")
    return categories


def _quality_report(gate: Mapping[str, Any]) -> str:
    lines = [
        "# Retrieval Quality Report",
        "",
        f"Status: `{gate['status']}`",
        "",
        "## Aggregate metrics",
        "",
    ]
    if gate["metrics_available"]:
        for name, value in sorted(gate["metrics"].items()):
            lines.append(f"- {name}: {float(value):.6f}")
    else:
        lines.append(
            "Retrieval metrics are withheld until qrels and judgment agreement "
            "are release eligible."
        )
        lines.append(f"- citation_validity: {float(gate['metrics']['citation_validity']):.6f}")
    lines.extend(["", "## Category breakdown", ""])
    if gate["category_metrics"]:
        for category, value in sorted(gate["category_metrics"].items()):
            lines.append(f"- {category}: {float(value):.6f}")
    else:
        lines.append("- none")
    lines.extend(["", "## Gate failures", ""])
    failures = gate["gate_failures"]
    if failures:
        lines.extend(f"- {failure}" for failure in failures)
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Artifact identity",
            "",
            f"- artifact_ids: {', '.join(gate['artifact_ids']) or 'none'}",
            f"- run_ids: {', '.join(gate['run_ids']) or 'none'}",
            "",
        ]
    )
    text = "\n".join(lines)
    if "\r\n" in text:
        raise ValueError("quality report must use LF line endings")
    return text


def _gate_output_projection(
    gate: Mapping[str, Any],
    cases: Sequence[EvalCase],
    runs: Sequence[RetrievalRun],
    *,
    include_holdout_details: bool,
) -> dict[str, Any]:
    output = {key: value for key, value in gate.items() if key != "per_query"}
    if include_holdout_details:
        return output

    public_query_ids = {
        case.query_id for case in cases if case.split != "holdout"
    }
    public_runs = [run for run in runs if run.query_id in public_query_ids]
    output["artifact_ids"] = sorted({run.artifact_id for run in public_runs})
    output["run_ids"] = sorted({run.run_id for run in public_runs})
    output["uncalibrated_query_ids"] = [
        query_id
        for query_id in output["uncalibrated_query_ids"]
        if query_id in public_query_ids
    ]
    agreement = dict(output["judgment_agreement"])
    for key in ("adjudication_ids", "unresolved_ids"):
        identifiers = agreement.pop(key, [])
        agreement[key.removesuffix("_ids") + "_count"] = len(identifiers)
    output["judgment_agreement"] = agreement
    return output


def _run_evaluate(args: argparse.Namespace) -> int:
    if args.include_holdout_details and not args.rotation_flag:
        raise ValueError(
            "holdout detail output requires both --include-holdout-details "
            "and --rotation-flag"
        )

    cases = _load_cases(args.cases)
    qrels = _load_qrels(args.qrels)
    runs = _load_runs(args.runs)
    findings = read_jsonl(args.findings)
    agreement = _agreement_object(_read_json(args.agreement))
    thresholds = _read_json(args.thresholds)
    if not isinstance(thresholds, Mapping):
        raise ValueError("thresholds input must be an object")
    gate = aggregate_quality_gate(
        cases,
        qrels,
        runs,
        findings,
        agreement,
        thresholds,
        baseline_category_metrics=_baseline_categories(args.baseline_metrics),
    )
    split_by_query = {case.query_id: case.split for case in cases}
    visible = [
        record
        for record in gate["per_query"]
        if split_by_query[str(record["query_id"])] != "holdout"
    ]
    holdout = [
        record
        for record in gate["per_query"]
        if split_by_query[str(record["query_id"])] == "holdout"
    ]
    metrics = {
        "status": gate["status"],
        "metrics_available": gate["metrics_available"],
        "metrics": gate["metrics"],
        "category_breakdown": gate["category_metrics"],
        "per_query": visible,
    }
    if args.include_holdout_details:
        metrics["holdout_details"] = holdout

    output_dir = Path(args.output_dir)
    gate_output = _gate_output_projection(
        gate,
        cases,
        runs,
        include_holdout_details=args.include_holdout_details,
    )
    _atomic_write_json(output_dir / "metrics.json", metrics)
    _atomic_write_json(output_dir / "quality_gate.json", gate_output)
    _atomic_write_bytes(
        output_dir / "quality_report.md",
        _quality_report(gate_output).encode("utf-8"),
    )
    return 0 if gate["status"] == "PASS" else 1


def _metric_rows(value: Any, label: str) -> Sequence[Mapping[str, Any]]:
    rows = value.get("per_query") if isinstance(value, Mapping) else value
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise ValueError(f"{label} must contain a per_query array")
    return rows


def _run_compare(args: argparse.Namespace) -> int:
    comparison = compare_quality_runs(
        _metric_rows(_read_json(args.baseline), "baseline"),
        _metric_rows(_read_json(args.candidate), "candidate"),
        max_category_regression=args.max_category_regression,
    )
    _atomic_write_json(args.output, comparison)
    return 0 if comparison["status"] == "PASS" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m em_rag.evaluation.cli",
        description="Validate retrieval evaluation data and release quality.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_dataset = subparsers.add_parser("validate-dataset")
    validate_dataset.add_argument("--cases", required=True)
    validate_dataset.add_argument("--qrels", required=True)
    validate_dataset.add_argument("--output", required=True)
    validate_dataset.set_defaults(handler=_run_validate_dataset)

    build_pool = subparsers.add_parser("build-pool")
    build_pool.add_argument("--cases", required=True)
    build_pool.add_argument("--runs", required=True, nargs="+")
    build_pool.add_argument("--output", required=True)
    build_pool.add_argument("--seed", required=True, type=int)
    build_pool.set_defaults(handler=_run_build_pool)

    validate_judgments = subparsers.add_parser("validate-judgments")
    validate_judgments.add_argument("--pass1", required=True)
    validate_judgments.add_argument("--pass2", required=True)
    validate_judgments.add_argument("--adjudication")
    validate_judgments.add_argument("--human-calibration")
    validate_judgments.add_argument("--sampled-id", action="append", default=[])
    validate_judgments.add_argument("--output", required=True)
    validate_judgments.set_defaults(handler=_run_validate_judgments)

    evaluate = subparsers.add_parser("evaluate-run")
    evaluate.add_argument("--cases", required=True)
    evaluate.add_argument("--qrels", required=True)
    evaluate.add_argument("--runs", required=True)
    evaluate.add_argument("--findings", required=True)
    evaluate.add_argument("--agreement", required=True)
    evaluate.add_argument("--thresholds", required=True)
    evaluate.add_argument("--baseline-metrics")
    evaluate.add_argument("--output-dir", required=True)
    evaluate.add_argument("--include-holdout-details", action="store_true")
    evaluate.add_argument("--rotation-flag", action="store_true")
    evaluate.set_defaults(handler=_run_evaluate)

    compare = subparsers.add_parser("compare-runs")
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--candidate", required=True)
    compare.add_argument("--output", required=True)
    compare.add_argument("--max-category-regression", type=float, default=0.05)
    compare.set_defaults(handler=_run_compare)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
