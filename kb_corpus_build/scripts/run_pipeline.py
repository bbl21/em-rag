#!/usr/bin/env python3
"""Unified runner for the EM knowledge-base build pipeline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPORT_PATH = Path("kb_corpus_build/reports/pipeline_run_report.md")
RETRIEVAL_SMOKE_REPORT_PATH = Path("kb_corpus_build/eval/reports/retrieval_smoke_report.md")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one or more EM knowledge-base pipeline stages.")
    parser.add_argument("--project-root", default=".", help="Project root containing kb_corpus_build.")
    parser.add_argument(
        "--stage",
        required=True,
        choices=[
            "scan",
            "extract",
            "clean",
            "chunk",
            "contextualize",
            "dedup",
            "structured",
            "bm25",
            "vector",
            "retrieve-test",
            "eval",
            "final-report",
            "all",
        ],
        help="Stage to run.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Write planned commands without running them.")
    parser.add_argument("--skip-vector", action="store_true", help="Skip vector build inside --stage all.")
    return parser.parse_args(argv)


def write_text_checked(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    if path.read_text(encoding="utf-8") != text:
        raise ValueError(f"UTF-8 roundtrip mismatch for {path.as_posix()}")


def py(project_root: Path, script: str, *args: str) -> list[str]:
    return [sys.executable, str(project_root / "kb_corpus_build" / "scripts" / script), "--project-root", str(project_root), *args]


def command(name: str, argv: list[str]) -> dict[str, Any]:
    return {"name": name, "argv": argv}


def md_cell(value: Any) -> str:
    return str(value if value is not None else "").replace("|", "\\|").replace("\n", " ").strip()


def display_command(argv: list[str], project_root: Path) -> str:
    parts: list[str] = []
    for item in argv:
        text = str(item)
        if text == sys.executable:
            parts.append("kb_corpus_build/.venv/bin/python")
            continue
        if text == str(project_root):
            parts.append(".")
            continue
        prefix = str(project_root) + "/"
        if text.startswith(prefix):
            parts.append(text[len(prefix) :])
            continue
        parts.append(text)
    return " ".join(parts).replace("|", "\\|")


def write_retrieval_smoke_report(project_root: Path, stdout: str, status: str, stderr: str) -> None:
    output: dict[str, Any] = {}
    parse_error = ""
    if stdout.strip():
        try:
            parsed = json.loads(stdout)
            if isinstance(parsed, dict):
                output = parsed
        except json.JSONDecodeError as exc:
            parse_error = f"{type(exc).__name__}: {exc}"
    elif status == "PASS":
        parse_error = "retrieve.py produced empty stdout"

    effective_status = status if output and not parse_error else "FAIL"
    results = output.get("results") or []
    lines = [
        "# Retrieval Smoke Report",
        "",
        f"- overall_status：`{effective_status}`",
        f"- query：{md_cell(output.get('query') or '<unknown>')}",
        f"- top_k：{output.get('top_k', '<unknown>')}",
        f"- retrieval_mode：`{md_cell(output.get('retrieval_mode') or '<unknown>')}`",
        f"- vector_index_used：`{str(bool(output.get('vector_index_used'))).lower()}`",
        f"- out_of_scope：`{str(bool(output.get('out_of_scope'))).lower()}`",
        f"- result_count：{len(results)}",
    ]
    if parse_error:
        lines.append(f"- parse_error：`{md_cell(parse_error)}`")
    if stderr.strip():
        lines.append(f"- stderr_tail：`{md_cell(stderr[-500:])}`")
    lines.extend(
        [
            "",
            "| rank | chunk_id | source_id | final_score | citation |",
            "|---:|---|---|---:|---|",
        ]
    )
    if results:
        for row in results[:10]:
            lines.append(
                f"| {row.get('rank', '')} | `{md_cell(row.get('chunk_id'))}` | "
                f"`{md_cell(row.get('source_id'))}` | {row.get('final_score', 0.0)} | "
                f"{md_cell(row.get('citation'))} |"
            )
    else:
        lines.append("| <none> | <none> | <none> | 0.0 | <none> |")
    lines.append("")
    write_text_checked(project_root / RETRIEVAL_SMOKE_REPORT_PATH, "\n".join(lines))


def stage_commands(project_root: Path, stage: str, skip_vector: bool = False) -> list[dict[str, Any]]:
    retrieve_query = "What is the frequency range of ITU-R P.1411?"
    retrieve_args = ["--query", retrieve_query, "--top-k", "5"]
    if skip_vector:
        retrieve_args.extend(["--retrieval-mode", "bm25_structured"])
    stage_map: dict[str, list[dict[str, Any]]] = {
        "scan": [command("scan", py(project_root, "scan_sources.py"))],
        "extract": [
            command("extract_latex", py(project_root, "extract_latex.py")),
            command("extract_pdf", py(project_root, "extract_pdf.py")),
        ],
        "clean": [command("clean", py(project_root, "clean_corpus.py"))],
        "chunk": [command("chunk", py(project_root, "chunk_corpus.py"))],
        "contextualize": [command("contextualize", py(project_root, "contextualize_chunks.py"))],
        "dedup": [command("dedup", py(project_root, "deduplicate_chunks.py"))],
        "structured": [
            command("extract_structured_indexes", py(project_root, "extract_structured_indexes.py")),
            command("build_structured_indexes", py(project_root, "build_structured_indexes.py")),
        ],
        "bm25": [command("build_bm25_index", py(project_root, "build_bm25_index.py"))],
        "vector": [command("build_vector_index", py(project_root, "build_vector_index.py"))],
        "retrieve-test": [
            command(
                "retrieve_test",
                py(project_root, "retrieve.py", *retrieve_args),
            )
        ],
        "eval": [
            command(
                "retrieval_quality_eval",
                py(
                    project_root,
                    "eval_retrieval_quality.py",
                    "--top-k",
                    "10",
                    "--retrieval-mode",
                    "bm25_structured",
                    "--output-dir",
                    "kb_corpus_build/audit/retrieval_quality_evaluation",
                ),
            ),
            command("security_smoke", py(project_root, "eval_security_smoke.py")),
        ],
        "final-report": [command("final_report", py(project_root, "build_readiness_report.py"))],
    }
    if stage != "all":
        return stage_map[stage]
    commands: list[dict[str, Any]] = []
    for name in [
        "scan",
        "extract",
        "clean",
        "chunk",
        "contextualize",
        "dedup",
        "structured",
        "bm25",
    ]:
        commands.extend(stage_map[name])
    if not skip_vector:
        commands.extend(stage_map["vector"])
    commands.extend(stage_map["retrieve-test"])
    commands.extend(stage_map["eval"])
    commands.extend(stage_map["final-report"])
    return commands


def run_pipeline(project_root: Path, stage: str, dry_run: bool = False, skip_vector: bool = False) -> dict[str, Any]:
    commands = stage_commands(project_root, stage, skip_vector=skip_vector)
    results: list[dict[str, Any]] = []
    for item in commands:
        started = time.perf_counter()
        if dry_run:
            results.append({**item, "status": "DRY_RUN", "returncode": 0, "seconds": 0.0})
            continue
        completed = subprocess.run(item["argv"], cwd=project_root, text=True, capture_output=True)
        status = "PASS" if completed.returncode == 0 else "FAIL"
        row = {
            **item,
            "status": status,
            "returncode": completed.returncode,
            "seconds": round(time.perf_counter() - started, 3),
            "stdout_tail": completed.stdout[-1000:],
            "stderr_tail": completed.stderr[-1000:],
        }
        results.append(row)
        if item["name"] == "retrieve_test":
            write_retrieval_smoke_report(project_root, completed.stdout, status, completed.stderr)
        if completed.returncode != 0:
            break
    if dry_run:
        overall = "DRY_RUN"
    elif all(row["status"] == "PASS" for row in results) and len(results) == len(commands):
        overall = "PASS"
    else:
        overall = "FAIL"
    write_pipeline_report(project_root / REPORT_PATH, stage, dry_run, results, overall)
    return {"overall_status": overall, "stage": stage, "results": results}


def write_pipeline_report(path: Path, stage: str, dry_run: bool, results: list[dict[str, Any]], overall: str) -> None:
    lines = [
        "# Pipeline Run Report",
        "",
        f"- stage：`{stage}`",
        f"- dry_run：`{str(dry_run).lower()}`",
        f"- overall_status：`{overall}`",
        "",
        "| step | status | seconds | command |",
        "|---|---|---:|---|",
    ]
    for row in results:
        command_text = display_command(row["argv"], path.parent.parent.parent)
        lines.append(f"| `{row['name']}` | `{row['status']}` | {row['seconds']} | `{command_text}` |")
    lines.append("")
    write_text_checked(path, "\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = Path(args.project_root).resolve()
    result = run_pipeline(project_root, args.stage, dry_run=args.dry_run, skip_vector=args.skip_vector)
    print(f"Pipeline {args.stage}: {result['overall_status']}")
    return 0 if result["overall_status"] in {"PASS", "DRY_RUN"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
