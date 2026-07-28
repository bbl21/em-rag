#!/usr/bin/env python3
"""Phase-3 PDF extractor for the EM knowledge corpus build."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import fitz
import yaml


BODY_TEXT_STARTERS = {"Thus", "Then", "Therefore", "Hence", "This", "These", "Those"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract structured text from PDF sources.")
    parser.add_argument("--project-root", default=".", help="Project root containing kb_corpus_build and reference.")
    return parser.parse_args()


def load_sources(path: Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    sources = data.get("sources")
    if not isinstance(sources, list):
        raise ValueError("sources.yaml missing top-level 'sources' list")
    return [item for item in sources if isinstance(item, dict)]


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def safe_rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def looks_like_equation(line: str) -> bool:
    if line.startswith("Equation:"):
        return True
    if "=" not in line:
        return False
    if len(line) > 240:
        return False
    score = 0
    if re.search(r"\b[a-zA-Z]\s*=\s*[-+0-9a-zA-Z(]", line):
        score += 1
    if re.search(r"[\^_{}]|\\[A-Za-z]+", line):
        score += 1
    if re.search(r"\b(sin|cos|tan|log|exp|min|max)\b", line, flags=re.IGNORECASE):
        score += 1
    if re.search(r"[+\-*/]", line):
        score += 1
    return score >= 2


def is_table_line(line: str) -> bool:
    return bool(re.match(r"^(Table|TABLE)\b", line))


def is_figure_line(line: str) -> bool:
    return bool(re.match(r"^(Figure|FIGURE|Fig\.)\b", line))


def looks_like_section_heading(line: str) -> bool:
    if len(line) > 120 or "=" in line:
        return False
    if looks_like_equation(line) or is_table_line(line) or is_figure_line(line):
        return False
    if re.search(r"[.:;]\s+\S", line):
        return False

    match = re.match(r"^([1-9]\d*(?:\.[1-9]\d*)*)\s+(.+)$", line)
    if not match:
        return False

    number, title = match.groups()
    title = title.strip()
    if not title:
        return False
    if not re.match(r"^[A-Z][A-Za-z0-9()/-]*\b", title):
        return False
    first_word = re.match(r"^([A-Za-z]+)", title)
    if first_word and len(first_word.group(1)) <= 2:
        return False
    if "." not in number and int(number) > 30:
        return False
    return True


def looks_like_section_number(line: str) -> bool:
    if not re.match(r"^[1-9]\d*(?:\.[1-9]\d*)*$", line):
        return False
    if "." not in line and int(line) > 30:
        return False
    return True


def looks_like_heading_title(line: str) -> bool:
    if len(line) > 75 or "=" in line:
        return False
    if line.startswith(("Rec.", "Recommendation", "RECOMMENDATION")):
        return False
    if re.search(r"\.{3,}", line):
        return False
    if re.search(r"[.;:]\s+\S", line):
        return False
    if " " not in line and line.isupper() and len(line) <= 6:
        return False
    if looks_like_equation(line) or is_table_line(line) or is_figure_line(line):
        return False
    if not re.match(r"^[A-Z][A-Za-z0-9()/-]*\b", line):
        return False
    first_word = re.match(r"^([A-Za-z]+)", line)
    if first_word and len(first_word.group(1)) <= 2:
        return False
    if first_word and first_word.group(1) in BODY_TEXT_STARTERS:
        return False
    return True


def detect_heading(lines: list[str]) -> tuple[str | None, str | None]:
    chapter: str | None = None
    section: str | None = None
    candidates = lines[:12]
    for index, line in enumerate(candidates):
        if re.match(r"^Chapter\b", line, flags=re.IGNORECASE):
            chapter = line
            continue
        if re.match(r"^(Recommendation|RECOMMENDATION|Annex)\b", line):
            chapter = line
            continue
        if section is not None:
            continue
        if line == "Scope":
            section = line
            continue
        if looks_like_section_heading(line):
            section = line
            continue
        if index + 1 < len(candidates) and looks_like_section_number(line) and looks_like_heading_title(candidates[index + 1]):
            section = f"{line} {candidates[index + 1]}"
            continue
    return chapter, section


def merge_heading_state(
    current_chapter: str | None,
    current_section: str | None,
    page_chapter: str | None,
    page_section: str | None,
) -> tuple[str | None, str | None]:
    next_chapter = current_chapter
    next_section = current_section

    if page_chapter is not None:
        next_chapter = page_chapter
        if page_section is None:
            next_section = None

    if page_section is not None:
        next_section = page_section

    return next_chapter, next_section


class PdfExtractor:
    def __init__(self, project_root: Path, build_root: Path) -> None:
        self.project_root = project_root
        self.build_root = build_root
        self.warnings: list[str] = []
        self.source_reports: list[dict[str, Any]] = []

    def extract_all(self, sources: list[dict[str, Any]]) -> int:
        processed = 0
        for source in sources:
            source_type = str(source.get("source_type", ""))
            if not source_type.startswith("pdf"):
                continue
            self.extract_source(source)
            processed += 1
        self.write_warnings()
        self.write_report(processed)
        return processed

    def extract_source(self, source: dict[str, Any]) -> None:
        source_id = str(source["source_id"])
        title = str(source["title"])
        raw_path = str(source["raw_path"])
        source_type = str(source["source_type"])
        pdf_path = (self.project_root / raw_path).resolve()
        pages_path = self.build_root / "intermediate" / "pdf" / f"{source_id}.pages.jsonl"
        cleaned_path = self.build_root / "corpus" / "by_source" / f"{source_id}.cleaned.md"
        pages_path.parent.mkdir(parents=True, exist_ok=True)
        cleaned_path.parent.mkdir(parents=True, exist_ok=True)

        records: list[dict[str, Any]] = []
        pages = 0
        text_pages = 0
        empty_pages = 0
        equation_count = 0
        table_count = 0
        figure_count = 0
        warning_count_before = len(self.warnings)
        current_chapter: str | None = None
        current_section: str | None = None

        with fitz.open(pdf_path) as document:
            for page_index in range(document.page_count):
                page_number = page_index + 1
                text = normalize_text(document.load_page(page_index).get_text("text"))
                lines = [line.strip() for line in text.splitlines() if line.strip()]
                page_chapter, page_section = detect_heading(lines)
                current_chapter, current_section = merge_heading_state(
                    current_chapter=current_chapter,
                    current_section=current_section,
                    page_chapter=page_chapter,
                    page_section=page_section,
                )
                equations = [line for line in lines if looks_like_equation(line)]
                tables = [line for line in lines if is_table_line(line)]
                figures = [line for line in lines if is_figure_line(line)]
                quality_flags: list[str] = []

                if not text:
                    empty_pages += 1
                    quality_flags.append("no_text_layer")
                    quality_flags.append("empty_text_page")
                    self.warnings.append(f"[{source_id}] page {page_number}: empty text extraction from PyMuPDF text layer")
                else:
                    text_pages += 1

                record = {
                    "source_id": source_id,
                    "source_title": title,
                    "source_type": source_type,
                    "raw_path": raw_path,
                    "page": page_number,
                    "chapter": current_chapter,
                    "section": current_section,
                    "content_md": text,
                    "equations": equations,
                    "tables": tables,
                    "figures": figures,
                    "quality_flags": quality_flags,
                }
                records.append(record)
                pages += 1
                equation_count += len(equations)
                table_count += len(tables)
                figure_count += len(figures)

        self.write_jsonl(pages_path, records)
        self.write_cleaned_markdown(cleaned_path, title, records)

        self.source_reports.append(
            {
                "source_id": source_id,
                "pages": pages,
                "text_pages": text_pages,
                "empty_pages": empty_pages,
                "equations": equation_count,
                "tables": table_count,
                "figures": figure_count,
                "pages_path": safe_rel(pages_path, self.project_root),
                "cleaned_path": safe_rel(cleaned_path, self.project_root),
                "warnings": len(self.warnings) - warning_count_before,
            }
        )

    def write_jsonl(self, path: Path, records: list[dict[str, Any]]) -> None:
        lines = [json.dumps(record, ensure_ascii=False) for record in records]
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8", newline="\n")
        _ = path.read_text(encoding="utf-8")

    def write_cleaned_markdown(self, path: Path, title: str, records: list[dict[str, Any]]) -> None:
        lines: list[str] = [f"# {title}"]
        for record in records:
            lines.extend(["", f"## Page {record['page']}"])
            if record["chapter"]:
                lines.extend(["", record["chapter"]])
            if record["section"] and record["section"] != record["chapter"]:
                lines.extend(["", record["section"]])
            if record["content_md"]:
                lines.extend(["", record["content_md"]])
        text = normalize_text("\n".join(lines)) + "\n"
        path.write_text(text, encoding="utf-8", newline="\n")
        _ = path.read_text(encoding="utf-8")

    def write_warnings(self) -> None:
        log_path = self.build_root / "logs" / "pdf_warnings.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        text = "\n".join(self.warnings)
        log_path.write_text(text + ("\n" if text else ""), encoding="utf-8", newline="\n")
        _ = log_path.read_text(encoding="utf-8")

    def write_report(self, processed: int) -> None:
        report_path = self.build_root / "reports" / "pdf_extraction_report.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        phase_status = "completed" if processed >= 1 else "blocked"
        warnings_path = self.build_root / "logs" / "pdf_warnings.log"
        lines = [
            "# Phase 3 PDF Extraction Report",
            "",
            f"Phase status: {phase_status}",
            f"warnings: {len(self.warnings)}",
            f"warnings_log: {safe_rel(warnings_path, self.project_root)}",
            "",
            "## Source outputs",
            "",
        ]
        for report in self.source_reports:
            lines.extend(
                [
                    f"### {report['source_id']}",
                    f"- pages: {report['pages']}",
                    f"- text_pages: {report['text_pages']}",
                    f"- empty_pages: {report['empty_pages']}",
                    f"- equations: {report['equations']}",
                    f"- tables: {report['tables']}",
                    f"- figures: {report['figures']}",
                    f"- pages_jsonl: {report['pages_path']}",
                    f"- cleaned_markdown: {report['cleaned_path']}",
                    f"- warnings: {report['warnings']}",
                    "",
                ]
            )
        if not self.source_reports:
            lines.extend(["No PDF sources were processed.", ""])
        report_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
        _ = report_path.read_text(encoding="utf-8")


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    build_root = project_root / "kb_corpus_build"
    sources_path = build_root / "metadata" / "sources.yaml"
    if not sources_path.is_file():
        raise FileNotFoundError(f"Missing sources file: {sources_path}")

    extractor = PdfExtractor(project_root, build_root)
    sources = load_sources(sources_path)
    extractor.extract_all(sources)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
