#!/usr/bin/env python3
"""Phase-10 mandatory vector index builder."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_HF_CACHE_RELATIVE = Path("kb_corpus_build") / ".cache" / "huggingface"
OPTIONAL_REQUIREMENTS = [
    "rank-bm25>=0.2.2",
    "bm25s>=0.2.0",
    "sentence-transformers>=2.7.0",
    "faiss-cpu>=1.8.0",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an optional FAISS vector index over canonical chunks.")
    parser.add_argument("--project-root", default=".", help="Project root containing kb_corpus_build.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Sentence Transformers model name.")
    return parser.parse_args()


def safe_rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def normalize_scalar(value: Any) -> str:
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


def write_json_checked(path: Path, value: dict[str, Any]) -> None:
    write_text_checked(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl_checked(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    text = "\n".join(lines) + ("\n" if lines else "")
    write_text_checked(path, text)


def make_temp_path(target: Path) -> Path:
    return target.parent / f"{target.name}.{uuid.uuid4().hex}.tmp"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def missing_vector_dependencies() -> list[str]:
    if os.environ.get("EM_KB_FORCE_VECTOR_MISSING_DEPS") == "1":
        return ["sentence-transformers", "faiss-cpu"]
    missing: list[str] = []
    if importlib.util.find_spec("sentence_transformers") is None:
        missing.append("sentence-transformers")
    if importlib.util.find_spec("faiss") is None:
        missing.append("faiss-cpu")
    return missing


def default_huggingface_cache_dir(project_root: Path | None = None) -> Path:
    root = project_root.resolve() if project_root is not None else Path.cwd().resolve()
    return root / DEFAULT_HF_CACHE_RELATIVE


def ensure_huggingface_cache_env(project_root: Path | None = None) -> Path:
    existing = os.environ.get("HF_HOME")
    if existing:
        return Path(existing)
    cache_dir = default_huggingface_cache_dir(project_root)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)
    return cache_dir


def write_optional_requirements(path: Path) -> None:
    text = "# Optional retrieval dependencies for BM25/vector indexes.\n" + "\n".join(OPTIONAL_REQUIREMENTS) + "\n"
    write_text_checked(path, text)


def build_docstore(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    docstore: list[dict[str, Any]] = []
    for row in rows:
        docstore.append(
            {
                "chunk_id": normalize_scalar(row.get("chunk_id")),
                "embedding_text": normalize_scalar(row.get("embedding_text") or row.get("content_md")),
                "content_md": normalize_scalar(row.get("content_md")),
                "contextual_header": normalize_scalar(row.get("contextual_header")),
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
            }
        )
    return docstore


def write_blocked_report(project_root: Path, missing: list[str]) -> None:
    build_root = project_root / "kb_corpus_build"
    report_path = build_root / "reports" / "vector_build_report.md"
    requirements_path = build_root / "requirements-vector.txt"
    write_optional_requirements(requirements_path)
    lines = [
        "# Phase 10 Vector Build Report",
        "",
        "Phase status: blocked_missing_optional_dependencies",
        f"missing_optional_dependencies: {', '.join(missing)}",
        f"requirements_path: {safe_rel(requirements_path, project_root)}",
        "bm25_fallback_required: true",
    ]
    write_text_checked(report_path, "\n".join(lines) + "\n")


def _validate_vector_artifacts(
    tmp_index_path: Path, tmp_metadata_path: Path, tmp_docstore_path: Path, expected_count: int
) -> None:
    metadata = json.loads(tmp_metadata_path.read_text(encoding="utf-8"))
    documents_indexed = int(metadata.get("documents_indexed", -1))
    if documents_indexed != expected_count:
        raise ValueError(
            "Vector metadata documents_indexed mismatch: "
            f"{documents_indexed} != {expected_count}"
        )
    docstore = load_jsonl(tmp_docstore_path)
    if len(docstore) != expected_count:
        raise ValueError(
            "Vector docstore row count mismatch: "
            f"{len(docstore)} != {expected_count}"
        )

    import faiss

    index = faiss.read_index(str(tmp_index_path))
    if int(index.ntotal) != expected_count:
        raise ValueError(f"Vector index ntotal mismatch: {int(index.ntotal)} != {expected_count}")
    if documents_indexed != int(index.ntotal):
        raise ValueError("documents_indexed and index ntotal are inconsistent.")
    if metadata.get("index_sha256") != sha256_file(tmp_index_path):
        raise ValueError("Vector index checksum does not match metadata.")
    if metadata.get("docstore_sha256") != sha256_file(tmp_docstore_path):
        raise ValueError("Vector docstore checksum does not match metadata.")


def write_completed_report(project_root: Path, model_name: str, rows: list[dict[str, Any]], dimension: int) -> None:
    build_root = project_root / "kb_corpus_build"
    report_path = build_root / "reports" / "vector_build_report.md"
    lines = [
        "# Phase 10 Vector Build Report",
        "",
        "Phase status: completed",
        f"model: {model_name}",
        f"hf_home: {safe_rel(Path(os.environ.get('HF_HOME', '')), project_root)}",
        f"documents_indexed: {len(rows)}",
        f"embedding_dimension: {dimension}",
        f"index_path: {safe_rel(build_root / 'indexes' / 'vector' / 'faiss.index', project_root)}",
        f"docstore_path: {safe_rel(build_root / 'indexes' / 'vector' / 'docstore.jsonl', project_root)}",
    ]
    write_text_checked(report_path, "\n".join(lines) + "\n")


def build_vector_index(project_root: Path, model_name: str) -> str:
    ensure_huggingface_cache_env(project_root)
    missing = missing_vector_dependencies()
    if missing:
        write_blocked_report(project_root, missing)
        return "blocked_missing_optional_dependencies"

    import faiss
    import numpy as np
    from vector_query_worker import (
        API_BACKEND,
        LocalCPUEmbedder,
        OpenAICompatibleEmbedder,
        embedding_backend,
    )

    build_root = project_root / "kb_corpus_build"
    rows = load_jsonl(build_root / "corpus" / "chunks.canonical.jsonl")
    docstore = build_docstore(rows)
    texts = [row["embedding_text"] for row in docstore]
    backend = embedding_backend()
    if backend == API_BACKEND:
        model = OpenAICompatibleEmbedder(model_name)
    else:
        model = LocalCPUEmbedder(build_root, model_name)
    embeddings = model.encode(texts)
    embeddings_array = np.asarray(embeddings, dtype="float32")
    if embeddings_array.ndim != 2:
        raise ValueError("Embedding model returned an unexpected tensor shape.")

    vector_dir = build_root / "indexes" / "vector"
    vector_dir.mkdir(parents=True, exist_ok=True)
    target_index_path = vector_dir / "faiss.index"
    target_docstore_path = vector_dir / "docstore.jsonl"
    target_metadata_path = vector_dir / "index_metadata.json"
    tmp_index_path = make_temp_path(target_index_path)
    tmp_docstore_path = make_temp_path(target_docstore_path)
    tmp_metadata_path = make_temp_path(target_metadata_path)
    temp_paths = [tmp_index_path, tmp_docstore_path, tmp_metadata_path]
    expected_count = len(docstore)

    try:
        index = faiss.IndexFlatIP(embeddings_array.shape[1])
        index.add(embeddings_array)
        faiss.write_index(index, str(tmp_index_path))
        write_jsonl_checked(tmp_docstore_path, docstore)
        index_sha256 = sha256_file(tmp_index_path)
        docstore_sha256 = sha256_file(tmp_docstore_path)
        write_json_checked(
            tmp_metadata_path,
            {
                "model": model.model_name,
                "model_identity": model.identity,
                "embedding_backend_at_build": backend,
                "normalized_embeddings": True,
                "metric": "inner_product",
                "documents_indexed": expected_count,
                "embedding_dimension": int(embeddings_array.shape[1]),
                "index_sha256": index_sha256,
                "docstore_sha256": docstore_sha256,
            },
        )
        _validate_vector_artifacts(tmp_index_path, tmp_metadata_path, tmp_docstore_path, expected_count)
        # Metadata is the commit marker. Readers validate both checksums before
        # using the positional FAISS-to-docstore mapping, so an interrupted
        # multi-file replacement fails closed instead of citing the wrong row.
        tmp_index_path.replace(target_index_path)
        tmp_docstore_path.replace(target_docstore_path)
        tmp_metadata_path.replace(target_metadata_path)
        write_completed_report(project_root, model.model_name, rows, int(embeddings_array.shape[1]))
    finally:
        for temp_path in temp_paths:
            if temp_path.exists():
                temp_path.unlink()
    return "completed"


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    status = build_vector_index(project_root, args.model)
    if status != "completed":
        print(f"Vector index build failed closed: {status}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
