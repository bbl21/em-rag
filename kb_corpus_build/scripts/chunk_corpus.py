#!/usr/bin/env python3
"""Phase-5 structure-aware chunking for the EM knowledge corpus build."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


TARGET_TOKENS = 1200
MIN_TOKENS = 200
MAX_TOKENS = 1500
OVERLAP_TOKENS = 80
TOKEN_RE = re.compile(r"\S+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create structure-aware RAG chunks from cleaned corpus units.")
    parser.add_argument("--project-root", default=".", help="Project root containing kb_corpus_build.")
    return parser.parse_args()


def safe_rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


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


def normalize_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"[ \t]+\n", "\n", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def token_estimate(text: str) -> int:
    return len(tokenize(text))


def sentence_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    stripped = text.strip()
    if not stripped:
        return spans
    for part in SENTENCE_SPLIT_RE.split(stripped):
        sentence = part.strip()
        if not sentence:
            continue
        index = stripped.find(sentence, cursor)
        if index < 0:
            index = cursor
        start = index
        end = start + len(sentence)
        spans.append((start, end))
        cursor = end
    return spans


def choose_chunk_end_token(text: str, tokens: list[re.Match[str]], start_token: int, target_end_token: int) -> int:
    if not tokens:
        return 0
    if target_end_token >= len(tokens):
        return len(tokens)

    best_end_token = min(target_end_token, len(tokens))
    for sentence_start, sentence_end in sentence_spans(text):
        end_token_count = sum(1 for token in tokens if token.end() <= sentence_end)
        if end_token_count <= start_token:
            continue
        chunk_tokens = end_token_count - start_token
        if MIN_TOKENS <= chunk_tokens <= TARGET_TOKENS:
            best_end_token = end_token_count
        if chunk_tokens >= TARGET_TOKENS:
            break
    return max(start_token + 1, min(len(tokens), best_end_token))


def split_content(content_md: str) -> list[str]:
    text = normalize_text(content_md)
    if not text:
        return [""]

    total_tokens = token_estimate(text)
    if total_tokens <= TARGET_TOKENS:
        return [text]

    chunks: list[str] = []
    start_token = 0
    token_matches = list(TOKEN_RE.finditer(text))
    while start_token < len(token_matches):
        remaining = len(token_matches) - start_token
        if start_token > 0 and remaining <= MIN_TOKENS:
            overlap_start = max(0, len(token_matches) - TARGET_TOKENS)
            start_char = token_matches[overlap_start].start()
            merged = normalize_text(text[start_char:])
            if not chunks or chunks[-1] != merged:
                chunks.append(merged)
            break

        target_end = min(start_token + TARGET_TOKENS, len(token_matches))
        actual_end = choose_chunk_end_token(text, token_matches, start_token, target_end)
        start_char = token_matches[start_token].start()
        chunk_end_char = token_matches[actual_end - 1].end()
        chunk_text = normalize_text(text[start_char:chunk_end_char])
        if not chunk_text:
            break
        chunks.append(chunk_text)
        if actual_end >= len(token_matches):
            break
        next_start = max(actual_end - OVERLAP_TOKENS, start_token + 1)
        if next_start <= start_token:
            next_start = actual_end
        start_token = next_start

    return chunks or [text]


def dedupe_preserve(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        value = item.strip()
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def has_los(text: str) -> bool:
    return bool(re.search(r"(?<![a-z])los(?![a-z])", text)) or "line-of-sight" in text


def has_nlos(text: str) -> bool:
    return bool(re.search(r"(?<![a-z])nlos(?![a-z])", text)) or "non-line-of-sight" in text


def is_itu_p1411_source(unit: dict[str, Any]) -> bool:
    source_id = str(unit.get("source_id") or "")
    source_title = str(unit.get("source_title") or "").lower()
    return source_id == "itu_r_p1411_13" or "itu-r p.1411" in source_title


def is_itu_model_unit(unit: dict[str, Any]) -> bool:
    content_md = str(unit.get("content_md") or "")
    content_type, _, _ = classify_content(unit, content_md)
    return content_type == "standard_model"


def classify_content(unit: dict[str, Any], content_md: str) -> tuple[str, list[str], list[str]]:
    haystack = " ".join(
        [
            str(unit.get("source_id") or ""),
            str(unit.get("source_title") or ""),
            str(unit.get("chapter") or ""),
            str(unit.get("section") or ""),
            content_md,
        ]
    )
    lower = haystack.lower()
    domain_tags: list[str] = []
    keywords: list[str] = []
    content_type = "theory"
    source_id = str(unit.get("source_id") or "")
    is_itu_p1411 = source_id == "itu_r_p1411_13" or "itu-r p.1411" in str(unit.get("source_title") or "").lower()
    itu_model_markers = [
        "basic transmission loss",
        "transmission loss model",
        "path loss",
        "building entry loss",
        "multipath",
        "los",
        "nlos",
        "angular profile",
        "angular spread",
        "cross-correlation",
        "site-general",
        "propagation model",
    ]

    if is_itu_p1411:
        domain_tags.append("itu_r_p1411")
    if "transmission line" in lower:
        domain_tags.append("transmission_line")

    is_itu_model = is_itu_p1411 and (
        any(marker in lower for marker in itu_model_markers if marker not in {"los", "nlos"})
        or has_los(lower)
        or has_nlos(lower)
    )

    if is_itu_model:
        content_type = "standard_model"
        domain_tags.append("propagation_model")
        if "frequency range" in lower:
            keywords.append("frequency range")
        if has_los(lower):
            keywords.append("LoS")
        if has_nlos(lower):
            keywords.append("NLoS")
        if "path loss" in lower:
            keywords.append("path loss")
        if "building entry loss" in lower:
            keywords.append("building entry loss")
    elif unit.get("equations") or "equation:" in lower or ("reflection coefficient" in lower and "=" in content_md):
        content_type = "formula"
        if "reflection coefficient" in lower:
            keywords.append("reflection coefficient")
        if "transmission line" in lower:
            keywords.append("transmission line")

    if unit.get("source_type") == "latex_book" and "transmission_line" in domain_tags:
        domain_tags.append("electromagnetics")

    return content_type, dedupe_preserve(domain_tags), dedupe_preserve(keywords)


def filter_equations_for_chunk(equations: list[str], content_md: str, chunk_index: int, chunk_count: int) -> list[str]:
    if chunk_count == 1:
        return dedupe_preserve(equations)
    chunk_lower = content_md.lower()
    kept = [equation for equation in equations if equation and equation.lower() in chunk_lower]
    if kept:
        return dedupe_preserve(kept)
    if chunk_index == 0 and equations:
        return [equations[0]]
    return []


def build_chunk(unit: dict[str, Any], content_md: str, chunk_index: int, chunk_count: int) -> dict[str, Any]:
    equations = [str(item) for item in unit.get("equations") or []]
    figures = [str(item) for item in unit.get("figures") or []]
    tables = [str(item) for item in unit.get("tables") or []]
    quality_flags = dedupe_preserve([str(item) for item in unit.get("quality_flags") or []])
    content_type, domain_tags, keywords = classify_content(unit, content_md)

    if not content_md and "empty_content" not in quality_flags:
        quality_flags.append("empty_content")

    return {
        "chunk_id": f"{unit.get('unit_id', 'unit')}::chunk:{chunk_index + 1:03d}",
        "source_id": str(unit.get("source_id") or ""),
        "source_title": str(unit.get("source_title") or ""),
        "source_type": str(unit.get("source_type") or ""),
        "raw_path": str(unit.get("raw_path") or ""),
        "chapter": unit.get("chapter"),
        "section": unit.get("section"),
        "subsection": unit.get("subsection"),
        "page_start": unit.get("page_start"),
        "page_end": unit.get("page_end"),
        "tex_file": unit.get("tex_file"),
        "content_type": content_type,
        "domain_tags": domain_tags,
        "keywords": keywords,
        "equations": filter_equations_for_chunk(equations, content_md, chunk_index, chunk_count),
        "figures": figures,
        "tables": tables,
        "content_md": content_md,
        "char_count": len(content_md),
        "token_estimate": token_estimate(content_md),
        "quality_flags": quality_flags,
    }


def merge_unit_lists(first: list[Any], second: list[Any]) -> list[str]:
    return dedupe_preserve([str(item) for item in first] + [str(item) for item in second])


def merge_itu_model_units(units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged_units: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for unit in units:
        if (
            current
            and is_itu_p1411_source(unit)
            and is_itu_model_unit(unit)
            and is_itu_p1411_source(current)
            and is_itu_model_unit(current)
            and str(current.get("chapter") or "") == str(unit.get("chapter") or "")
            and str(current.get("section") or "") == str(unit.get("section") or "")
        ):
            current_text = str(current.get("content_md") or "")
            unit_text = str(unit.get("content_md") or "")
            joined_text = "\n\n".join(part for part in [current_text.strip(), unit_text.strip()] if part)
            current["content_md"] = normalize_text(joined_text)
            current["page_start"] = min(
                value for value in [current.get("page_start"), unit.get("page_start")] if value is not None
            )
            current["page_end"] = max(
                value for value in [current.get("page_end"), unit.get("page_end")] if value is not None
            )
            current["equations"] = merge_unit_lists(current.get("equations") or [], unit.get("equations") or [])
            current["figures"] = merge_unit_lists(current.get("figures") or [], unit.get("figures") or [])
            current["tables"] = merge_unit_lists(current.get("tables") or [], unit.get("tables") or [])
            current["quality_flags"] = merge_unit_lists(
                current.get("quality_flags") or [], unit.get("quality_flags") or []
            )
            continue

        if current:
            merged_units.append(current)
        current = dict(unit)

    if current:
        merged_units.append(current)
    return merged_units


def build_chunks(units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for unit in merge_itu_model_units(units):
        chunk_texts = split_content(str(unit.get("content_md") or ""))
        for index, content_md in enumerate(chunk_texts):
            chunks.append(build_chunk(unit, content_md, index, len(chunk_texts)))
    return chunks


def write_report(project_root: Path, units: list[dict[str, Any]], chunks: list[dict[str, Any]]) -> None:
    build_root = project_root / "kb_corpus_build"
    chunks_path = build_root / "corpus" / "chunks.jsonl"
    report_path = build_root / "reports" / "chunk_report.md"
    max_tokens = max((int(chunk["token_estimate"]) for chunk in chunks), default=0)
    lines = [
        "# Phase 5 Chunking Report",
        "",
        "Phase status: completed",
        f"source_units: {len(units)}",
        f"chunks: {len(chunks)}",
        f"max_token_estimate: {max_tokens}",
        f"chunks_path: {safe_rel(chunks_path, project_root)}",
    ]
    write_text_checked(report_path, "\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    build_root = project_root / "kb_corpus_build"
    units_path = build_root / "intermediate" / "cleaned" / "cleaned_units.jsonl"
    chunks_path = build_root / "corpus" / "chunks.jsonl"

    units = load_jsonl(units_path)
    chunks = build_chunks(units)
    write_jsonl_checked(chunks_path, chunks)
    write_report(project_root, units, chunks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
