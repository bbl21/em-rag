#!/usr/bin/env python3
"""Phase-7 duplicate marking and canonical view generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*")
VARIABLE_RE = re.compile(r"\b[A-Za-z](?:_[A-Za-z0-9]+)?\b")
NEAR_DUPLICATE_THRESHOLD = 0.92


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mark exact and near duplicate chunks.")
    parser.add_argument("--project-root", default=".", help="Project root containing kb_corpus_build.")
    return parser.parse_args()


def safe_rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if isinstance(row, dict):
            rows.append(row)
    return rows


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


def normalize_content_text(value: Any) -> str:
    return str(value or "").replace("\r\n", "\n").replace("\r", "\n")


def normalize_for_hash(text: str) -> str:
    normalized = normalize_content_text(text).lower()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_scalar(value: Any) -> str:
    text = str(value or "").strip()
    return text


def normalize_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    scalar = normalize_scalar(value)
    return [scalar] if scalar else []


def token_estimate(row: dict[str, Any]) -> int:
    value = row.get("token_estimate")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return len(TOKEN_RE.findall(normalize_content_text(row.get("content_md"))))


def is_itu_standard_model(row: dict[str, Any]) -> bool:
    source_id = normalize_scalar(row.get("source_id")).lower()
    source_title = normalize_scalar(row.get("source_title")).lower()
    content_type = normalize_scalar(row.get("content_type")).lower()
    domain_tags = {item.lower() for item in normalize_list(row.get("domain_tags"))}
    return (
        content_type == "standard_model"
        and (
            source_id == "itu_r_p1411_13"
            or "itu-r p.1411" in source_title
            or "itu_r_p1411" in domain_tags
        )
    )


def metadata_completeness_score(row: dict[str, Any]) -> int:
    scalar_fields = [
        "source_id",
        "source_title",
        "source_type",
        "raw_path",
        "chapter",
        "section",
        "subsection",
        "page_start",
        "page_end",
        "tex_file",
        "contextual_header",
        "embedding_text",
        "bm25_text",
    ]
    score = sum(1 for field in scalar_fields if row.get(field) not in (None, "", []))
    score += len(normalize_list(row.get("keywords")))
    score += len(normalize_list(row.get("domain_tags")))
    return score


def location_completeness_score(row: dict[str, Any]) -> int:
    score = 0
    for field in ("chapter", "section", "subsection", "page_start", "page_end"):
        if row.get(field) not in (None, "", []):
            score += 1
    return score


def equation_completeness_score(row: dict[str, Any]) -> int:
    equations = normalize_list(row.get("equations"))
    equation_text = "\n".join(equations)
    variables = {match.group(0) for match in VARIABLE_RE.finditer(equation_text)}
    keywords = normalize_list(row.get("keywords"))
    return (len(equations) * 8) + len(variables) + len(keywords)


def preferred_length_score(row: dict[str, Any]) -> tuple[int, int]:
    tokens = token_estimate(row)
    in_range = 1 if 300 <= tokens <= 1200 else 0
    closeness = -abs(tokens - 750)
    return in_range, closeness


def canonical_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    preferred_range, length_closeness = preferred_length_score(row)
    content_md = normalize_content_text(row.get("content_md"))
    return (
        1 if is_itu_standard_model(row) else 0,
        metadata_completeness_score(row),
        location_completeness_score(row),
        equation_completeness_score(row),
        preferred_range,
        length_closeness,
        len(normalize_list(row.get("domain_tags"))),
        len(normalize_list(row.get("keywords"))),
        len(content_md),
        normalize_scalar(row.get("chunk_id")),
    )


def pick_canonical(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(rows, key=canonical_sort_key)


def tokenize_for_tfidf(row: dict[str, Any]) -> Counter[str]:
    base_text = normalize_for_hash(row.get("content_md") or row.get("embedding_text") or "")
    if not base_text:
        return Counter()
    return Counter(TOKEN_RE.findall(base_text))


def build_idf(rows: list[dict[str, Any]]) -> dict[str, float]:
    doc_freq: Counter[str] = Counter()
    documents: list[set[str]] = []
    for row in rows:
        tokens = set(tokenize_for_tfidf(row))
        documents.append(tokens)
    for tokens in documents:
        doc_freq.update(tokens)

    total_docs = max(len(rows), 1)
    return {
        token: math.log((1 + total_docs) / (1 + frequency)) + 1.0
        for token, frequency in doc_freq.items()
    }


def build_tfidf_vector(token_counts: Counter[str], idf: dict[str, float]) -> dict[str, float]:
    if not token_counts:
        return {}
    total_terms = sum(token_counts.values())
    vector: dict[str, float] = {}
    for token, count in token_counts.items():
        vector[token] = (count / total_terms) * idf.get(token, 1.0)
    return vector


def cosine_similarity(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    shared_tokens = set(left).intersection(right)
    numerator = sum(left[token] * right[token] for token in shared_tokens)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def initialize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    initialized: list[dict[str, Any]] = []
    for row in rows:
        content_md = normalize_content_text(row.get("content_md"))
        initialized_row = dict(row)
        initialized_row["exact_hash"] = sha256_hex(content_md)
        initialized_row["normalized_hash"] = sha256_hex(normalize_for_hash(content_md))
        initialized_row["duplicate_group_id"] = None
        initialized_row["duplicate_of"] = None
        initialized_row["is_canonical"] = True
        initialized_row["near_duplicate_chunk_ids"] = []
        initialized_row["dedup_method"] = "none"
        initialized_row["dedup_score"] = 0.0
        initialized.append(initialized_row)
    return initialized


def mark_exact_duplicates(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["normalized_hash"])].append(row)

    exact_duplicate_groups = 0
    for normalized_hash, group_rows in groups.items():
        group_rows.sort(key=lambda row: normalize_scalar(row.get("chunk_id")))
        duplicate_group_id = f"exact:{normalized_hash[:16]}"
        for row in group_rows:
            row["duplicate_group_id"] = duplicate_group_id
        if len(group_rows) == 1:
            continue

        exact_duplicate_groups += 1
        canonical_row = pick_canonical(group_rows)
        canonical_id = normalize_scalar(canonical_row.get("chunk_id"))
        for row in group_rows:
            row["dedup_score"] = 1.0
            row["dedup_method"] = "exact_hash"
            if row is canonical_row:
                row["is_canonical"] = True
                row["duplicate_of"] = None
            else:
                row["is_canonical"] = False
                row["duplicate_of"] = canonical_id

    return rows, exact_duplicate_groups


def mark_near_duplicates(rows: list[dict[str, Any]]) -> int:
    canonical_rows = [row for row in rows if row.get("is_canonical")]
    idf = build_idf(canonical_rows)
    vectors = {
        normalize_scalar(row.get("chunk_id")): build_tfidf_vector(tokenize_for_tfidf(row), idf)
        for row in canonical_rows
    }

    near_pairs = 0
    for index, left_row in enumerate(canonical_rows):
        left_id = normalize_scalar(left_row.get("chunk_id"))
        for right_row in canonical_rows[index + 1 :]:
            right_id = normalize_scalar(right_row.get("chunk_id"))
            similarity = cosine_similarity(vectors[left_id], vectors[right_id])
            if similarity < NEAR_DUPLICATE_THRESHOLD:
                continue

            near_pairs += 1
            left_row["near_duplicate_chunk_ids"].append(right_id)
            right_row["near_duplicate_chunk_ids"].append(left_id)

            if left_row.get("dedup_method") == "none":
                left_row["dedup_method"] = "tfidf_cosine"
                left_row["dedup_score"] = similarity
            elif left_row.get("dedup_method") == "tfidf_cosine":
                left_row["dedup_score"] = max(float(left_row["dedup_score"]), similarity)

            if right_row.get("dedup_method") == "none":
                right_row["dedup_method"] = "tfidf_cosine"
                right_row["dedup_score"] = similarity
            elif right_row.get("dedup_method") == "tfidf_cosine":
                right_row["dedup_score"] = max(float(right_row["dedup_score"]), similarity)

    for row in rows:
        near_ids = sorted(set(normalize_list(row.get("near_duplicate_chunk_ids"))))
        row["near_duplicate_chunk_ids"] = near_ids
    return near_pairs


def write_report(
    project_root: Path,
    input_rows: list[dict[str, Any]],
    dedup_rows: list[dict[str, Any]],
    exact_duplicate_groups: int,
    near_duplicate_pairs: int,
) -> None:
    build_root = project_root / "kb_corpus_build"
    dedup_path = build_root / "corpus" / "chunks.dedup.jsonl"
    canonical_path = build_root / "corpus" / "chunks.canonical.jsonl"
    report_path = build_root / "reports" / "dedup_report.md"
    canonical_count = sum(1 for row in dedup_rows if row.get("is_canonical"))
    lines = [
        "# Phase 7 Deduplication Report",
        "",
        "Phase status: completed",
        f"input_chunks: {len(input_rows)}",
        f"dedup_chunks: {len(dedup_rows)}",
        f"canonical_chunks: {canonical_count}",
        f"exact_duplicate_groups: {exact_duplicate_groups}",
        f"near_duplicate_pairs: {near_duplicate_pairs}",
        f"dedup_path: {safe_rel(dedup_path, project_root)}",
        f"canonical_path: {safe_rel(canonical_path, project_root)}",
    ]
    write_text_checked(report_path, "\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    build_root = project_root / "kb_corpus_build"
    input_path = build_root / "corpus" / "chunks.contextual.jsonl"
    dedup_path = build_root / "corpus" / "chunks.dedup.jsonl"
    canonical_path = build_root / "corpus" / "chunks.canonical.jsonl"

    input_rows = load_jsonl(input_path)
    dedup_rows = initialize_rows(input_rows)
    dedup_rows, exact_duplicate_groups = mark_exact_duplicates(dedup_rows)
    near_duplicate_pairs = mark_near_duplicates(dedup_rows)
    canonical_rows = [row for row in dedup_rows if row.get("is_canonical")]

    write_jsonl_checked(dedup_path, dedup_rows)
    write_jsonl_checked(canonical_path, canonical_rows)
    write_report(project_root, input_rows, dedup_rows, exact_duplicate_groups, near_duplicate_pairs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
