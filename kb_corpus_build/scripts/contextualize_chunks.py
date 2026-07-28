#!/usr/bin/env python3
"""Phase-6 contextual chunk augmentation for the EM knowledge corpus build."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


TOKEN_RE = re.compile(r"\S+")
HEADER_TOKEN_LIMIT = 120
INTERPRETATION_LINE = "Interpret this chunk as part of the above source and section."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add deterministic contextual headers to RAG chunks.")
    parser.add_argument("--project-root", default=".", help="Project root containing kb_corpus_build.")
    return parser.parse_args()


def write_text_checked(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    read_back = path.read_text(encoding="utf-8")
    if read_back != text:
        raise ValueError(f"UTF-8 roundtrip mismatch for {path.as_posix()}")


def write_jsonl_checked(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    text = "\n".join(lines) + ("\n" if lines else "")
    write_text_checked(path, text)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if isinstance(row, dict):
            rows.append(row)
    return rows


def normalize_scalar(value: Any) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_content_text(value: Any) -> str:
    text = str(value or "")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def normalize_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [normalize_scalar(item) for item in value if normalize_scalar(item)]
    scalar = normalize_scalar(value)
    return [scalar] if scalar else []


def normalize_equations(value: Any) -> list[str]:
    if isinstance(value, list):
        equations = [normalize_content_text(item).strip() for item in value]
    else:
        equations = [normalize_content_text(value).strip()]
    return [equation for equation in equations if equation]


def render_list(items: list[str]) -> str:
    return ", ".join(items) if items else "unknown"


def token_count(text: str) -> int:
    return len(TOKEN_RE.findall(text))


def trim_scalar_to_budget(prefix: str, value: str, max_tokens: int) -> str:
    value_tokens = TOKEN_RE.findall(value)
    value_budget = max_tokens - token_count(prefix)
    if value_budget <= 0 or not value_tokens:
        return f"{prefix}unknown"
    return f"{prefix}{' '.join(value_tokens[:value_budget])}"


def trim_location_to_budget(chapter: str, section: str, subsection: str, max_tokens: int) -> str:
    prefix = "Location: "
    scalar_budget = max_tokens - token_count(prefix)
    if scalar_budget <= 0:
        return f"{prefix}unknown > unknown > unknown"

    raw_fields = [chapter, section, subsection]
    kept_fields: list[str] = []
    remaining_budget = scalar_budget
    total_fields = len(raw_fields)
    for index, raw_field in enumerate(raw_fields):
        minimum_for_rest = total_fields - index - 1
        field_budget = remaining_budget - minimum_for_rest
        field_tokens = TOKEN_RE.findall(raw_field)
        if field_budget <= 0 or not field_tokens:
            kept = "unknown"
        else:
            kept = " ".join(field_tokens[:field_budget])
        kept_fields.append(kept)
        remaining_budget -= token_count(kept)

    return f"{prefix}{' > '.join(kept_fields)}"


def trim_list_to_budget(prefix: str, items: list[str], max_tokens: int) -> str:
    if max_tokens <= token_count(prefix):
        return f"{prefix}unknown"
    if not items:
        return f"{prefix}unknown"

    kept: list[str] = []
    for item in items:
        candidate_items = kept + [item]
        candidate = f"{prefix}{render_list(candidate_items)}"
        if token_count(candidate) <= max_tokens:
            kept.append(item)
            continue
        break

    if kept:
        return f"{prefix}{render_list(kept)}"
    return f"{prefix}unknown"


def build_contextual_header(row: dict[str, Any]) -> str:
    source_title = normalize_scalar(row.get("source_title")) or "unknown"
    chapter = normalize_scalar(row.get("chapter"))
    section = normalize_scalar(row.get("section"))
    subsection = normalize_scalar(row.get("subsection"))
    content_type = normalize_scalar(row.get("content_type")) or "unknown"
    domain_tags = normalize_list(row.get("domain_tags"))
    keywords = normalize_list(row.get("keywords"))

    line_minimums = [
        token_count("Source: unknown"),
        token_count("Location: unknown > unknown > unknown"),
        token_count("Type: unknown"),
        token_count("Domain tags: unknown"),
        token_count("Keywords: unknown"),
        token_count(INTERPRETATION_LINE),
    ]
    remaining_tokens = HEADER_TOKEN_LIMIT

    source_budget = remaining_tokens - sum(line_minimums[1:])
    source_line = trim_scalar_to_budget("Source: ", source_title, source_budget)
    remaining_tokens -= token_count(source_line)

    location_budget = remaining_tokens - sum(line_minimums[2:])
    location_line = trim_location_to_budget(chapter, section, subsection, location_budget)
    remaining_tokens -= token_count(location_line)

    type_budget = remaining_tokens - sum(line_minimums[3:])
    type_line = trim_scalar_to_budget("Type: ", content_type, type_budget)
    remaining_tokens -= token_count(type_line)

    domain_budget = remaining_tokens - sum(line_minimums[4:])
    domain_line = trim_list_to_budget("Domain tags: ", domain_tags, domain_budget)
    remaining_tokens -= token_count(domain_line)

    keywords_budget = remaining_tokens - line_minimums[5]
    keywords_line = trim_list_to_budget("Keywords: ", keywords, keywords_budget)

    header_lines = [
        source_line,
        location_line,
        type_line,
        domain_line,
        keywords_line,
        INTERPRETATION_LINE,
    ]
    header = "\n".join(header_lines)
    if token_count(header) > HEADER_TOKEN_LIMIT:
        header_lines = [
            "Source: unknown",
            "Location: unknown > unknown > unknown",
            "Type: unknown",
            "Domain tags: unknown",
            "Keywords: unknown",
            INTERPRETATION_LINE,
        ]
        header = "\n".join(header_lines)
    if token_count(header) > HEADER_TOKEN_LIMIT:
        raise ValueError("Contextual header exceeds token budget after deterministic fallback.")
    return header


def build_bm25_text(row: dict[str, Any]) -> str:
    content_md = normalize_content_text(row.get("content_md"))
    content_with_equations = append_missing_equations(content_md, row.get("equations"))
    fields = [
        normalize_scalar(row.get("source_title")) or "unknown",
        normalize_scalar(row.get("chapter")) or "unknown",
        normalize_scalar(row.get("section")) or "unknown",
        normalize_scalar(row.get("subsection")) or "unknown",
        render_list(normalize_list(row.get("keywords"))),
        content_with_equations,
    ]
    return "\n".join(fields)


def append_missing_equations(content_md: str, equations_value: Any) -> str:
    equations = normalize_equations(equations_value)
    missing_equations = [equation for equation in equations if equation not in content_md]
    if not missing_equations:
        return content_md
    appendix = "Equations:\n" + "\n".join(missing_equations)
    if not content_md:
        return appendix
    return f"{content_md}\n\n{appendix}"


def contextualize_row(row: dict[str, Any]) -> dict[str, Any]:
    output = dict(row)
    content_md = normalize_content_text(row.get("content_md"))
    content_with_equations = append_missing_equations(content_md, row.get("equations"))
    contextual_header = build_contextual_header(row)
    output["contextual_header"] = contextual_header
    output["embedding_text"] = f"{contextual_header}\n\n{content_with_equations}"
    output["bm25_text"] = build_bm25_text(row)
    return output


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    input_path = project_root / "kb_corpus_build" / "corpus" / "chunks.jsonl"
    output_path = project_root / "kb_corpus_build" / "corpus" / "chunks.contextual.jsonl"

    rows = load_jsonl(input_path)
    contextualized_rows = [contextualize_row(row) for row in rows]
    write_jsonl_checked(output_path, contextualized_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
