#!/usr/bin/env python3
"""Phase-9 BM25 index builder with a standard-library fallback implementation."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Any


K1 = 1.5
B = 0.75
TOKEN_RE = re.compile(r"[A-Za-z]+(?:_[A-Za-z0-9]+)*(?:[0-9]+)?|\d+(?:\.\d+)?")
FREQUENCY_RE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:Hz|kHz|MHz|GHz|THz)\b", re.IGNORECASE)
PHRASE_TOKEN_MAP = {
    "reflection coefficient": "reflection_coefficient",
    "path loss": "path_loss",
    "frequency range": "frequency_range",
}
OPTIONAL_REQUIREMENTS = [
    "rank-bm25>=0.2.2",
    "bm25s>=0.2.0",
    "sentence-transformers>=2.7.0",
    "faiss-cpu>=1.8.0",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a BM25 index over canonical RAG chunks.")
    parser.add_argument("--project-root", default=".", help="Project root containing kb_corpus_build.")
    return parser.parse_args()


def safe_rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def normalize_scalar(value: Any) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_content_text(value: Any) -> str:
    return str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def normalize_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [normalize_scalar(item) for item in value if normalize_scalar(item)]
    scalar = normalize_scalar(value)
    return [scalar] if scalar else []


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


def write_pickle_checked(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(value, fh)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Pickle write failed for {path.as_posix()}")


def normalize_rf_notation(text: str) -> str:
    text = re.sub(r"\b([SYZ])\s*_\s*\{\s*([0-9]{1,2})\s*\}", r"\1\2", text, flags=re.IGNORECASE)
    text = re.sub(r"\b([SYZ])\s+([0-9]{2})\b", r"\1\2", text, flags=re.IGNORECASE)
    return text


def tokenize_for_bm25(text: str) -> list[str]:
    normalized = normalize_rf_notation(normalize_content_text(text))
    lowered = normalized.lower()
    tokens: list[str] = []

    for match in FREQUENCY_RE.finditer(normalized):
        frequency_token = re.sub(r"\s+", "_", match.group(0).lower())
        tokens.append(frequency_token)

    for phrase, token in PHRASE_TOKEN_MAP.items():
        if phrase in lowered:
            tokens.append(token)

    for match in TOKEN_RE.finditer(normalized):
        tokens.append(match.group(0).lower())

    return tokens


def build_citation(row: dict[str, Any]) -> str:
    source_id = normalize_scalar(row.get("source_id")) or "unknown_source"
    chapter = normalize_scalar(row.get("chapter")) or "unknown chapter"
    section = normalize_scalar(row.get("section")) or "unknown section"
    page_start = row.get("page_start")
    page_end = row.get("page_end")
    if page_start not in (None, "") and page_end not in (None, ""):
        page = f"page {page_start}-{page_end}"
    elif page_start not in (None, ""):
        page = f"page {page_start}"
    else:
        page = "page unknown"
    return f"{source_id} | {chapter} | {section} | {page}"


def build_docstore(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    docstore: list[dict[str, Any]] = []
    for row in rows:
        bm25_text = normalize_content_text(row.get("bm25_text") or row.get("content_md") or row.get("embedding_text"))
        docstore.append(
            {
                "chunk_id": normalize_scalar(row.get("chunk_id")),
                "source_id": normalize_scalar(row.get("source_id")),
                "source_title": normalize_scalar(row.get("source_title")),
                "chapter": normalize_scalar(row.get("chapter")),
                "section": normalize_scalar(row.get("section")),
                "page_start": row.get("page_start"),
                "page_end": row.get("page_end"),
                "content_type": normalize_scalar(row.get("content_type")),
                "domain_tags": normalize_list(row.get("domain_tags")),
                "keywords": normalize_list(row.get("keywords")),
                "duplicate_group_id": normalize_scalar(row.get("duplicate_group_id") or row.get("chunk_id")),
                "contextual_header": normalize_content_text(row.get("contextual_header")),
                "content_md": normalize_content_text(row.get("content_md")),
                "bm25_text": bm25_text,
                "retrieval_text": bm25_text,
                "citation": build_citation(row),
            }
        )
    return docstore


def build_bm25_index(rows: list[dict[str, Any]]) -> dict[str, Any]:
    documents: list[dict[str, Any]] = []
    doc_freq: Counter[str] = Counter()
    for row in rows:
        chunk_id = normalize_scalar(row.get("chunk_id"))
        text = normalize_content_text(row.get("bm25_text") or row.get("content_md") or row.get("embedding_text"))
        tokens = tokenize_for_bm25(text)
        term_freq = Counter(tokens)
        doc_freq.update(sorted(set(tokens)))
        documents.append(
            {
                "chunk_id": chunk_id,
                "tokens": tokens,
                "length": len(tokens),
                "term_freq": dict(term_freq),
            }
        )

    doc_count = len(documents)
    avgdl = sum(doc["length"] for doc in documents) / doc_count if doc_count else 0.0
    idf = {
        token: math.log(1.0 + ((doc_count - doc_freq[token] + 0.5) / (doc_freq[token] + 0.5)))
        for token in sorted(doc_freq)
    }
    return {
        "version": 1,
        "algorithm": "internal_bm25",
        "k1": K1,
        "b": B,
        "doc_count": doc_count,
        "avgdl": avgdl,
        "doc_freq": dict(doc_freq),
        "idf": idf,
        "documents": documents,
    }


def score_bm25(index: dict[str, Any], query: str) -> list[tuple[str, float]]:
    query_tokens = tokenize_for_bm25(query)
    if not query_tokens:
        return []

    avgdl = float(index.get("avgdl") or 0.0)
    if avgdl <= 0.0:
        return []
    k1 = float(index.get("k1", K1))
    b = float(index.get("b", B))
    idf = index.get("idf") or {}
    scores: list[tuple[str, float]] = []
    for document in index.get("documents", []):
        term_freq = document.get("term_freq") or {}
        doc_length = float(document.get("length") or 0.0)
        score = 0.0
        for token in query_tokens:
            tf = float(term_freq.get(token, 0.0))
            if tf <= 0.0:
                continue
            denominator = tf + k1 * (1.0 - b + b * (doc_length / avgdl))
            score += float(idf.get(token, 0.0)) * ((tf * (k1 + 1.0)) / denominator)
        if score > 0.0:
            scores.append((str(document.get("chunk_id")), score))
    scores.sort(key=lambda item: (-item[1], item[0]))
    return scores


def missing_optional_bm25_dependencies() -> list[str]:
    missing: list[str] = []
    for module_name in ["rank_bm25", "bm25s"]:
        if importlib.util.find_spec(module_name) is None:
            missing.append(module_name)
    return missing


def write_optional_requirements(path: Path) -> None:
    text = "# Optional retrieval dependencies for BM25/vector indexes.\n" + "\n".join(OPTIONAL_REQUIREMENTS) + "\n"
    write_text_checked(path, text)


def write_report(
    project_root: Path,
    index_path: Path,
    docstore_path: Path,
    rows: list[dict[str, Any]],
    index: dict[str, Any],
    missing_dependencies: list[str],
) -> None:
    report_path = project_root / "kb_corpus_build" / "reports" / "bm25_build_report.md"
    vocabulary_size = len(index.get("doc_freq") or {})
    dependency_status = "missing" if missing_dependencies else "available"
    lines = [
        "# Phase 9 BM25 Build Report",
        "",
        "Phase status: completed",
        "algorithm: internal_bm25",
        f"documents_indexed: {len(rows)}",
        f"vocabulary_size: {vocabulary_size}",
        f"average_document_length: {float(index.get('avgdl') or 0.0):.2f}",
        f"optional_bm25_dependencies: {dependency_status}",
        f"missing_optional_bm25_dependencies: {', '.join(missing_dependencies) if missing_dependencies else '<none>'}",
        f"index_path: {safe_rel(index_path, project_root)}",
        f"docstore_path: {safe_rel(docstore_path, project_root)}",
        "",
        "Tokenizer support:",
    ]
    for sample in ["S11", "S21", "VSWR", "LoS", "NLoS", "300 MHz", "2.4 GHz", "epsilon_r", "Z0", "reflection coefficient", "path loss"]:
        lines.append(f"- {sample}: {', '.join(tokenize_for_bm25(sample))}")
    write_text_checked(report_path, "\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    build_root = project_root / "kb_corpus_build"
    input_path = build_root / "corpus" / "chunks.canonical.jsonl"
    index_path = build_root / "indexes" / "bm25" / "bm25_index.pkl"
    docstore_path = build_root / "indexes" / "bm25" / "bm25_docstore.jsonl"
    requirements_path = build_root / "requirements-vector.txt"

    rows = load_jsonl(input_path)
    docstore = build_docstore(rows)
    index = build_bm25_index(rows)
    missing_dependencies = missing_optional_bm25_dependencies()

    write_pickle_checked(index_path, index)
    write_jsonl_checked(docstore_path, docstore)
    if missing_dependencies:
        write_optional_requirements(requirements_path)
    write_report(project_root, index_path, docstore_path, rows, index, missing_dependencies)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
