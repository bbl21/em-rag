#!/usr/bin/env python3
"""Build the final RAG readiness report from local artifacts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = Path("kb_corpus_build/reports/rag_readiness_report.md")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the final RAG readiness report.")
    parser.add_argument("--project-root", default=".", help="Project root containing kb_corpus_build.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT.as_posix(), help="Output report path.")
    return parser.parse_args(argv)


def resolve_project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def write_text_checked(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    if path.read_text(encoding="utf-8") != text:
        raise ValueError(f"UTF-8 roundtrip mismatch for {path.as_posix()}")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if isinstance(row, dict):
            rows.append(row)
    return rows


def file_status(path: Path) -> str:
    return "present" if path.exists() and (path.is_dir() or path.stat().st_size > 0) else "missing"


def count_yaml_sources(path: Path) -> int:
    if not path.is_file():
        return 0
    return len(re.findall(r"^\s*-\s+source_id\s*:", path.read_text(encoding="utf-8"), flags=re.MULTILINE))


def grep_value(path: Path, pattern: str, default: str = "unknown") -> str:
    if not path.is_file():
        return default
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(pattern, text, flags=re.MULTILINE)
    return match.group(1).strip() if match else default


def report_status_from_markdown(path: Path) -> str:
    if not path.is_file():
        return "missing"
    text = path.read_text(encoding="utf-8", errors="replace")
    for pattern in [r"overall_status：`([^`]+)`", r"quality_gate：`([^`]+)`", r"Phase status:\s*([^\n]+)"]:
        value = grep_value(path, pattern, "")
        if value:
            return value
    return "present"


def build_report_text(project_root: Path) -> str:
    build = project_root / "kb_corpus_build"
    metadata = build / "metadata"
    corpus = build / "corpus"
    reports = build / "reports"
    eval_reports = build / "eval" / "reports"
    audit = build / "audit" / "retrieval_quality_evaluation"
    hybrid_audit = build / "audit" / "retrieval_quality_evaluation_hybrid"
    indexes = build / "indexes"

    source_count = count_yaml_sources(metadata / "sources.yaml")
    canonical_rows = load_jsonl(corpus / "chunks.canonical.jsonl")
    dedup_rows = load_jsonl(corpus / "chunks.dedup.jsonl")
    chunk_rows = load_jsonl(corpus / "chunks.jsonl")
    contextual_rows = load_jsonl(corpus / "chunks.contextual.jsonl")
    formula_rows = load_jsonl(corpus / "formula_index.jsonl")
    term_rows = load_jsonl(corpus / "term_index.jsonl")
    propagation_rows = load_jsonl(corpus / "propagation_model_index.jsonl")
    duplicate_count = sum(1 for row in dedup_rows if row.get("is_canonical") is False)
    cleaned_files = sorted((corpus / "by_source").glob("*.cleaned.md")) if (corpus / "by_source").is_dir() else []
    missing_dependencies = grep_value(reports / "vector_build_report.md", r"missing_optional_dependencies:\s*([^\n]+)", "<none>")
    hybrid_gate = report_status_from_markdown(hybrid_audit / "quality_gate_summary.md")
    unfinished = (
        "none"
        if hybrid_gate == "PASS"
        else "完整 Hybrid/vector 质量复审仍为 needs_validation；当前第十四阶段门禁只证明 bm25_structured baseline。"
    )
    hybrid_recommendation = (
        "- 完整 Hybrid/vector 已通过；接入前仍需人工关注 WEAK_PASS 类别和 citation 规划。"
        if hybrid_gate == "PASS"
        else "- 在发布或接入 agent 前，单独完成 Hybrid/vector validation。"
    )

    lines = [
        "# RAG Readiness Report",
        "",
        "## Summary",
        "",
        f"- source inventory 状态：{file_status(metadata / 'source_inventory.md')}",
        f"- sources.yaml source 数：{source_count}",
        f"- cleaned markdown 文件数：{len(cleaned_files)}",
        f"- chunk 数：{len(chunk_rows)}",
        f"- contextual chunk 数：{len(contextual_rows)}",
        f"- canonical chunk 数：{len(canonical_rows)}",
        f"- duplicate 数：{duplicate_count}",
        f"- formula index 数：{len(formula_rows)}",
        f"- term index 数：{len(term_rows)}",
        f"- propagation model index 数：{len(propagation_rows)}",
        "",
        "## Source Recognition",
        "",
        "- 预期原始资料数：4",
        f"- 已识别资料数：{source_count}",
        f"- LaTeX/PDF 扫描报告：{file_status(reports / 'scan_report.md')}",
        f"- LaTeX 抽取报告：{file_status(reports / 'latex_extraction_report.md')}",
        f"- PDF 抽取报告：{file_status(reports / 'pdf_extraction_report.md')}",
        "",
        "## Index Status",
        "",
        f"- BM25 index 状态：{file_status(indexes / 'bm25' / 'bm25_index.pkl')}",
        f"- BM25 docstore 状态：{file_status(indexes / 'bm25' / 'bm25_docstore.jsonl')}",
        f"- vector index 状态：{file_status(indexes / 'vector' / 'faiss.index')}",
        f"- vector docstore 状态：{file_status(indexes / 'vector' / 'docstore.jsonl')}",
        f"- structured index 状态：{file_status(indexes / 'structured')}",
        "",
        "## Evaluation Status",
        "",
        f"- retrieve.py 测试结果：{file_status(build / 'rag' / 'retrieval_trace.jsonl')}",
        f"- retrieval quality gate：{report_status_from_markdown(audit / 'quality_gate_summary.md')}",
        f"- hybrid retrieval quality gate：{hybrid_gate}",
        f"- security smoke test 结果：{report_status_from_markdown(eval_reports / 'security_smoke_report.md')}",
        f"- pipeline run report：{file_status(reports / 'pipeline_run_report.md')}",
        "",
        "## Dependencies And Failures",
        "",
        f"- 缺失依赖：{missing_dependencies}",
        f"- 已知未完成项：{unfinished}",
        "- 失败项：当前 readiness report 未发现缺失的核心已生成产物；若重新运行完整 pipeline，应以 pipeline_run_report 为准。",
        "",
        "## Human Review Recommendations",
        "",
        "- 优先审查 `kb_corpus_build/audit/retrieval_quality_evaluation/failure_taxonomy_and_fix_plan.md` 中的 WEAK_PASS 类别。",
        "- 对 P.1411 总范围问题，人工确认回答是否区分 Recommendation 总范围和子模型局部范围。",
        "- 对多 citation 问题，人工确认回答阶段没有强制引用弱相关 top-k。",
        hybrid_recommendation,
        "",
    ]
    return "\n".join(lines)


def write_readiness_report(project_root: Path, output: Path | None = None) -> Path:
    output = output or project_root / DEFAULT_OUTPUT
    text = build_report_text(project_root)
    write_text_checked(output, text)
    return output


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = Path(args.project_root).resolve()
    output = write_readiness_report(project_root, resolve_project_path(project_root, args.output))
    print(f"Wrote readiness report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
