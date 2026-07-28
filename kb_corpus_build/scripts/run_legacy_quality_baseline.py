#!/usr/bin/env python3
"""Replay legacy retrieval modes into the retrieval-quality v2 run contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for import_root in (REPO_ROOT / "src", SCRIPT_DIR):
    import_path = str(import_root)
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

import retrieve as legacy_retrieval
from em_rag.evaluation.io import read_jsonl, write_jsonl
from em_rag.evaluation.models import EvalCase, RetrievalRun


RETRIEVAL_MODES = ("bm25_structured", "hybrid", "vector_only")
VECTOR_MODES = frozenset({"hybrid", "vector_only"})
MAX_VECTOR_BATCH_SIZE = 8
VECTOR_CONCURRENCY = 1
DEFAULT_EVAL_CASES = Path("kb_corpus_build/eval/datasets/retrieval_quality_eval_cases.jsonl")
DEFAULT_RUNS_DIR = Path("kb_corpus_build/eval/retrieval_quality_v2/runs")


class BaselineError(RuntimeError):
    """Raised when a requested baseline cannot produce a valid run artifact."""


def _positive_int(value: str) -> int:
    converted = int(value)
    if converted < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return converted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--eval-cases", default=str(DEFAULT_EVAL_CASES))
    parser.add_argument("--output-dir", default=str(DEFAULT_RUNS_DIR))
    parser.add_argument("--modes", nargs="+", choices=RETRIEVAL_MODES, default=list(RETRIEVAL_MODES))
    parser.add_argument("--top-k", type=_positive_int, default=12)
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument("--split", choices=("development", "regression", "holdout", "adversarial"))
    parser.add_argument("--limit", type=_positive_int)
    parser.add_argument("--vector-batch-size", type=_positive_int, default=MAX_VECTOR_BATCH_SIZE)
    parser.add_argument(
        "--vector-timeout-seconds",
        type=_positive_int,
        default=legacy_retrieval.vector_timeout_seconds(),
    )
    return parser.parse_args()


def _resolve_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _display_path(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
    try:
        temporary_path.write_text(text, encoding="utf-8", newline="\n")
        written = temporary_path.read_bytes()
        if written.decode("utf-8") != text or b"\r\n" in written:
            raise BaselineError(f"UTF-8/LF roundtrip mismatch for {path.as_posix()}")
        decoded = json.loads(written.decode("utf-8"))
        if not isinstance(decoded, dict) or decoded != payload:
            raise BaselineError(f"JSON payload mismatch for {path.as_posix()}")
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _git_sha(project_root: Path) -> str:
    status = subprocess.run(
        [
            "git",
            "-C",
            str(project_root),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        text=True,
        capture_output=True,
        timeout=10,
    )
    if status.returncode != 0:
        error = (status.stderr or status.stdout or "").strip()
        raise BaselineError(error or "could not inspect Git worktree status")
    if status.stdout:
        raise BaselineError("repository is dirty; commit or remove all changes before running a baseline")

    completed = subprocess.run(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        timeout=10,
    )
    if completed.returncode != 0:
        error = (completed.stderr or completed.stdout or "").strip()
        raise BaselineError(error or "could not resolve git SHA")
    sha = completed.stdout.strip()
    if not sha:
        raise BaselineError("git returned an empty SHA")
    return sha


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize_modes(modes: Sequence[str]) -> list[str]:
    requested = set(modes)
    if not requested:
        raise ValueError("at least one retrieval mode is required")
    unsupported = requested - set(RETRIEVAL_MODES)
    if unsupported:
        raise ValueError(f"unsupported retrieval modes: {', '.join(sorted(unsupported))}")
    return [mode for mode in RETRIEVAL_MODES if mode in requested]


def _eval_case_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
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


def _load_cases(path: Path, split: str | None, limit: int | None) -> list[EvalCase]:
    cases = [EvalCase.from_dict(_eval_case_payload(row)) for row in read_jsonl(path)]
    if split is not None:
        cases = [case for case in cases if case.split == split]
    cases.sort(key=lambda case: case.query_id)
    if limit is not None:
        cases = cases[:limit]
    if not cases:
        raise BaselineError("no evaluation cases matched the requested selection")
    return cases


def _config(
    modes: list[str],
    top_k: int,
    confidence_threshold: float,
    split: str | None,
    limit: int | None,
    vector_batch_size: int,
    vector_timeout_seconds: int,
) -> dict[str, Any]:
    return {
        "confidence_threshold": float(confidence_threshold),
        "limit": limit,
        "retrieval_modes": modes,
        "split": split,
        "top_k": top_k,
        "vector_batch_size": vector_batch_size,
        "vector_concurrency": VECTOR_CONCURRENCY,
        "vector_timeout_seconds": vector_timeout_seconds,
    }


def _run_key(artifact_sha256: str, git_sha: str, config: dict[str, Any]) -> str:
    encoded_config = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
    digest = hashlib.sha256(f"{artifact_sha256}\n{git_sha}\n{encoded_config}".encode("utf-8")).hexdigest()
    return f"legacy-{digest[:20]}"


def _retrieval_run_row(
    run_key: str,
    mode: str,
    case: EvalCase,
    artifact_id: str,
    evidence: list[dict[str, Any]],
    confidence_threshold: float,
) -> dict[str, Any]:
    row = {
        "run_id": f"{run_key}:{mode}:{case.query_id}",
        "query_id": case.query_id,
        "artifact_id": artifact_id,
        "results": evidence,
        "degraded": False,
        "confidence_threshold": float(confidence_threshold),
    }
    try:
        return asdict(RetrievalRun.from_dict(row))
    except ValueError as error:
        raise BaselineError(f"invalid {mode} run for {case.query_id}: {error}") from error


def _legacy_evidence(results: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for result in results:
        evidence.append(
            {
                "rank": result.get("rank"),
                "chunk_id": result.get("chunk_id"),
                "score": result.get("final_score"),
                "citation": result.get("citation"),
                "text": result.get("content_preview"),
                "source_id": result.get("source_id"),
            }
        )
    return evidence


def _legacy_mode_rows(
    project_root: Path,
    cases: list[EvalCase],
    mode: str,
    top_k: int,
    confidence_threshold: float,
    artifact_id: str,
    run_key: str,
    vector_scores_by_query: dict[str, dict[str, float]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        kwargs: dict[str, Any] = {}
        if mode == "hybrid":
            if vector_scores_by_query is None:
                raise BaselineError("hybrid mode requires precomputed vector scores")
            kwargs = {
                "vector_scores_override": vector_scores_by_query[case.query_id],
                "vector_status_override": "ok",
                "vector_error_override": "",
            }
        output = legacy_retrieval.retrieve(
            project_root,
            case.query,
            top_k,
            retrieval_mode=mode,
            **kwargs,
        )
        if mode == "hybrid" and not output.get("out_of_scope"):
            vector_status = str(output.get("vector_status") or "missing")
            if vector_status != "ok" or not output.get("vector_index_used"):
                raise BaselineError(
                    f"hybrid silently skipped vector for {case.query_id}: status={vector_status}"
                )
        rows.append(
            _retrieval_run_row(
                run_key,
                mode,
                case,
                artifact_id,
                _legacy_evidence(output.get("results", [])),
                confidence_threshold,
            )
        )
    return rows


def _collect_vector_scores(
    project_root: Path,
    output_dir: Path,
    run_key: str,
    cases: list[EvalCase],
    batch_size: int,
    timeout_seconds: int,
) -> dict[str, dict[str, float]]:
    preflight_status, preflight_error = legacy_retrieval.run_vector_runtime_preflight(
        project_root,
        output_dir / f"{run_key}.vector_preflight",
        timeout_seconds=timeout_seconds,
    )
    if preflight_status != "ok":
        detail = preflight_error or "no error detail"
        raise BaselineError(f"vector preflight failed: status={preflight_status}; {detail}")

    query_items = [(case.query_id, legacy_retrieval.expand_query(case.query)) for case in cases]
    scores_by_query: dict[str, dict[str, float]] = {}
    for offset in range(0, len(query_items), batch_size):
        batch = dict(query_items[offset : offset + batch_size])
        batch_scores, status, error = legacy_retrieval.collect_batch_vector_scores_with_status(
            project_root / "kb_corpus_build",
            batch,
            timeout_seconds=timeout_seconds,
        )
        if status != "ok":
            detail = error or "no error detail"
            raise BaselineError(f"vector batch failed: status={status}; {detail}")
        if set(batch_scores) != set(batch):
            missing = sorted(set(batch) - set(batch_scores))
            raise BaselineError(f"vector batch silently skipped queries: {', '.join(missing)}")
        for query_id in batch:
            score_map = batch_scores[query_id]
            if not isinstance(score_map, dict) or not score_map:
                raise BaselineError(f"vector batch silently skipped vector for {query_id}")
            normalized: dict[str, float] = {}
            for chunk_id, score in score_map.items():
                converted = float(score)
                if not math.isfinite(converted):
                    raise BaselineError(f"non-finite vector score for {query_id}:{chunk_id}")
                normalized[str(chunk_id)] = converted
            scores_by_query[query_id] = normalized
    return scores_by_query


def _vector_only_rows(
    project_root: Path,
    cases: list[EvalCase],
    scores_by_query: dict[str, dict[str, float]],
    top_k: int,
    confidence_threshold: float,
    artifact_id: str,
    run_key: str,
) -> list[dict[str, Any]]:
    docstore_path = project_root / "kb_corpus_build" / "indexes" / "bm25" / "bm25_docstore.jsonl"
    docstore = {str(row.get("chunk_id")): row for row in legacy_retrieval.load_jsonl(docstore_path)}
    if not docstore:
        raise BaselineError(f"BM25 docstore is unavailable: {docstore_path.as_posix()}")

    rows: list[dict[str, Any]] = []
    for case in cases:
        ranked = sorted(scores_by_query[case.query_id].items(), key=lambda item: (-item[1], item[0]))[:top_k]
        evidence: list[dict[str, Any]] = []
        for rank, (chunk_id, raw_score) in enumerate(ranked, start=1):
            doc = docstore.get(chunk_id)
            if doc is None:
                raise BaselineError(f"vector chunk is missing from BM25 docstore: {chunk_id}")
            text = legacy_retrieval.content_preview_for_query(
                str(doc.get("content_md") or doc.get("retrieval_text") or doc.get("embedding_text") or ""),
                case.query,
            )
            evidence.append(
                {
                    "rank": rank,
                    "chunk_id": chunk_id,
                    "score": raw_score,
                    "citation": doc.get("citation"),
                    "text": text,
                    "source_id": doc.get("source_id"),
                }
            )
        rows.append(
            _retrieval_run_row(
                run_key,
                "vector_only",
                case,
                artifact_id,
                evidence,
                confidence_threshold,
            )
        )
    return rows


def _mode_run_path(output_dir: Path, run_key: str, mode: str) -> Path:
    if mode not in RETRIEVAL_MODES:
        raise ValueError(f"unsupported retrieval mode: {mode}")
    return output_dir / f"{run_key}.{mode}.jsonl"


def _remove_requested_mode_outputs(output_dir: Path, run_key: str, modes: Sequence[str]) -> None:
    for mode in modes:
        path = _mode_run_path(output_dir, run_key, mode)
        if path.is_dir():
            raise BaselineError(f"run output path is a directory: {path.as_posix()}")
        path.unlink(missing_ok=True)


def _write_mode_run(output_dir: Path, run_key: str, mode: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    filename = f"{run_key}.{mode}.jsonl"
    path = _mode_run_path(output_dir, run_key, mode)
    with tempfile.NamedTemporaryFile(
        dir=output_dir,
        prefix=f".{filename}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
    try:
        write_jsonl(temporary_path, rows)
        parsed = [RetrievalRun.from_dict(row) for row in read_jsonl(temporary_path)]
        if len(parsed) != len(rows):
            raise BaselineError(f"row count mismatch after writing {mode}")
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return {"file": filename, "row_count": len(parsed), "status": "ok"}


def run_baseline(
    project_root: Path,
    eval_cases_path: Path,
    output_dir: Path,
    *,
    modes: Sequence[str],
    top_k: int,
    confidence_threshold: float,
    split: str | None,
    limit: int | None,
    vector_batch_size: int,
    vector_timeout_seconds: int,
) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc)
    started_clock = time.monotonic()
    project_root = project_root.resolve()
    eval_cases_path = eval_cases_path.resolve()
    output_dir = output_dir.resolve()
    normalized_modes = _normalize_modes(modes)
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if not math.isfinite(float(confidence_threshold)):
        raise ValueError("confidence_threshold must be finite")
    if not 1 <= vector_batch_size <= MAX_VECTOR_BATCH_SIZE:
        raise ValueError(f"vector_batch_size must be between 1 and {MAX_VECTOR_BATCH_SIZE}")
    if vector_timeout_seconds < 1:
        raise ValueError("vector_timeout_seconds must be positive")

    git_sha = _git_sha(project_root)
    cases = _load_cases(eval_cases_path, split, limit)
    artifact_sha256 = _sha256(eval_cases_path)
    config = _config(
        normalized_modes,
        top_k,
        confidence_threshold,
        split,
        limit,
        vector_batch_size,
        vector_timeout_seconds,
    )
    run_key = _run_key(artifact_sha256, git_sha, config)
    artifact_id = f"sha256:{artifact_sha256}"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"{run_key}.manifest.json"
    if manifest_path.is_dir():
        raise BaselineError(f"manifest path is a directory: {manifest_path.as_posix()}")
    manifest_path.unlink(missing_ok=True)
    _remove_requested_mode_outputs(output_dir, run_key, normalized_modes)
    mode_runs: dict[str, dict[str, Any]] = {}

    if "bm25_structured" in normalized_modes:
        try:
            rows = _legacy_mode_rows(
                project_root,
                cases,
                "bm25_structured",
                top_k,
                confidence_threshold,
                artifact_id,
                run_key,
            )
            mode_runs["bm25_structured"] = _write_mode_run(
                output_dir,
                run_key,
                "bm25_structured",
                rows,
            )
        except (BaselineError, OSError, ValueError) as error:
            mode_runs["bm25_structured"] = {"status": "failed", "error": str(error)}

    vector_scores_by_query: dict[str, dict[str, float]] | None = None
    requested_vector_modes = [mode for mode in normalized_modes if mode in VECTOR_MODES]
    if requested_vector_modes:
        try:
            vector_scores_by_query = _collect_vector_scores(
                project_root,
                output_dir,
                run_key,
                cases,
                vector_batch_size,
                vector_timeout_seconds,
            )
        except (BaselineError, OSError, ValueError) as error:
            for mode in requested_vector_modes:
                mode_runs[mode] = {"status": "failed", "error": str(error)}

    if vector_scores_by_query is not None and "hybrid" in requested_vector_modes:
        try:
            rows = _legacy_mode_rows(
                project_root,
                cases,
                "hybrid",
                top_k,
                confidence_threshold,
                artifact_id,
                run_key,
                vector_scores_by_query,
            )
            mode_runs["hybrid"] = _write_mode_run(output_dir, run_key, "hybrid", rows)
        except (BaselineError, OSError, ValueError) as error:
            mode_runs["hybrid"] = {"status": "failed", "error": str(error)}

    if vector_scores_by_query is not None and "vector_only" in requested_vector_modes:
        try:
            rows = _vector_only_rows(
                project_root,
                cases,
                vector_scores_by_query,
                top_k,
                confidence_threshold,
                artifact_id,
                run_key,
            )
            mode_runs["vector_only"] = _write_mode_run(output_dir, run_key, "vector_only", rows)
        except (BaselineError, OSError, ValueError) as error:
            mode_runs["vector_only"] = {"status": "failed", "error": str(error)}

    finished_at = datetime.now(timezone.utc)
    status = "ok" if all(mode_runs.get(mode, {}).get("status") == "ok" for mode in normalized_modes) else "failed"
    manifest = {
        "artifact_id": artifact_id,
        "artifact_path": _display_path(eval_cases_path, project_root),
        "artifact_sha256": artifact_sha256,
        "config": config,
        "git_sha": git_sha,
        "mode_runs": mode_runs,
        "query_order": [case.query_id for case in cases],
        "run_key": run_key,
        "runtime": {
            "duration_seconds": round(time.monotonic() - started_clock, 6),
            "finished_at": finished_at.isoformat(),
            "platform": platform.platform(),
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "started_at": started_at.isoformat(),
        },
        "schema_version": "retrieval_quality_v2",
        "status": status,
    }
    _write_json(manifest_path, manifest)
    return manifest


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    eval_cases_path = _resolve_path(project_root, args.eval_cases)
    output_dir = _resolve_path(project_root, args.output_dir)
    try:
        manifest = run_baseline(
            project_root,
            eval_cases_path,
            output_dir,
            modes=args.modes,
            top_k=args.top_k,
            confidence_threshold=args.confidence_threshold,
            split=args.split,
            limit=args.limit,
            vector_batch_size=args.vector_batch_size,
            vector_timeout_seconds=args.vector_timeout_seconds,
        )
    except (BaselineError, OSError, ValueError) as error:
        print(f"legacy baseline failed: {error}", file=sys.stderr)
        return 2
    summary = {
        "manifest": f"{manifest['run_key']}.manifest.json",
        "mode_status": {mode: details["status"] for mode, details in manifest["mode_runs"].items()},
        "status": manifest["status"],
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if manifest["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
