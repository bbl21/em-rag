#!/usr/bin/env python3
"""Timeout-isolated vector query worker for retrieve.py."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


LOCAL_BACKEND = "local_cpu"
API_BACKEND = "openai_compatible"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score one or more queries against the local FAISS vector index.")
    parser.add_argument("--build-root", required=True, help="Path to kb_corpus_build.")
    query_group = parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument("--query", help="Expanded query text.")
    query_group.add_argument("--queries-json", help="JSON object mapping query IDs to expanded query text.")
    query_group.add_argument("--serve", action="store_true", help="Keep the model loaded and accept JSON-line requests on stdin.")
    return parser.parse_args()


def normalize_scalar(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if isinstance(row, dict):
            rows.append(row)
    return rows


def configure_offline_runtime(build_root: Path) -> None:
    os.environ.setdefault("HF_HOME", str(build_root / ".cache" / "huggingface"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def embedding_backend() -> str:
    backend = os.environ.get("EM_RAG_EMBEDDING_BACKEND", LOCAL_BACKEND).strip().lower()
    if backend not in {LOCAL_BACKEND, API_BACKEND}:
        raise ValueError("EM_RAG_EMBEDDING_BACKEND must be local_cpu or openai_compatible")
    return backend


def resolve_local_model(build_root: Path, model_name: str) -> str:
    configured = os.environ.get("EM_RAG_LOCAL_EMBEDDING_MODEL", "").strip()
    if configured:
        path = Path(configured).expanduser()
        if not path.is_dir():
            raise FileNotFoundError(f"Configured local embedding model is missing: {path.as_posix()}")
        return str(path)
    bundled = build_root / ".cache" / "models" / model_name.rsplit("/", 1)[-1]
    return str(bundled if bundled.is_dir() else model_name)


def fingerprint_model_directory(path: Path) -> str:
    """Identify the exact local model bytes used to build and query an index."""
    digest = hashlib.sha256()
    ignored_runtime_directories = {".cache", "onnx", "openvino"}
    files = sorted(
        item
        for item in path.rglob("*")
        if item.is_file()
        and not ignored_runtime_directories.intersection(item.relative_to(path).parts)
        and not item.name.endswith((".lock", ".incomplete"))
    )
    if not files:
        raise ValueError(f"Local embedding model contains no fingerprintable files: {path.as_posix()}")
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class LocalCPUEmbedder:
    def __init__(self, build_root: Path, model_name: str) -> None:
        configure_offline_runtime(build_root)
        from sentence_transformers import SentenceTransformer

        model_ref = resolve_local_model(build_root, model_name)
        self.model = SentenceTransformer(model_ref, local_files_only=True, device="cpu")
        model_path = Path(model_ref)
        if not model_path.is_dir():
            raise ValueError(
                "Local CPU embedding requires a concrete model directory so its version can be locked"
            )
        self.model_name = model_name
        self.identity = fingerprint_model_directory(model_path)

    def encode(self, texts: list[str]) -> Any:
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


class OpenAICompatibleEmbedder:
    def __init__(self, model_name: str) -> None:
        self.base_url = os.environ.get("EM_RAG_EMBEDDINGS_BASE_URL", "").strip().rstrip("/")
        self.api_key = os.environ.get("EM_RAG_EMBEDDINGS_API_KEY", "").strip()
        self.model = os.environ.get("EM_RAG_EMBEDDINGS_MODEL", "").strip() or model_name
        self.revision = os.environ.get("EM_RAG_EMBEDDINGS_MODEL_REVISION", "").strip()
        if not self.base_url:
            raise ValueError("EM_RAG_EMBEDDINGS_BASE_URL is required for the API backend")
        if not self.revision:
            raise ValueError("EM_RAG_EMBEDDINGS_MODEL_REVISION is required for the API backend")
        self.model_name = self.model
        self.identity = f"api:{self.model}@{self.revision}"

    def encode(self, texts: list[str]) -> Any:
        import httpx
        import numpy as np

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        timeout = float(os.environ.get("EM_RAG_EMBEDDINGS_TIMEOUT_SECONDS", "30"))
        response = httpx.post(
            f"{self.base_url}/embeddings",
            headers=headers,
            json={"model": self.model, "input": texts},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list) or len(data) != len(texts):
            raise ValueError("Embeddings API returned an invalid data array")
        ordered = sorted(data, key=lambda row: row.get("index", 0))
        vectors = np.asarray([row.get("embedding") for row in ordered], dtype="float32")
        if vectors.ndim != 2:
            raise ValueError("Embeddings API returned invalid vectors")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms <= 0):
            raise ValueError("Embeddings API returned a zero vector")
        return vectors / norms


def load_vector_runtime(build_root: Path) -> tuple[Any, Any, list[dict[str, Any]], Any]:
    vector_dir = build_root / "indexes" / "vector"
    metadata_path = vector_dir / "index_metadata.json"
    docstore_path = vector_dir / "docstore.jsonl"
    index_path = vector_dir / "faiss.index"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing vector index: {index_path.as_posix()}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing vector metadata: {metadata_path.as_posix()}")
    if not docstore_path.is_file():
        raise FileNotFoundError(f"Missing vector docstore: {docstore_path.as_posix()}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model_name = metadata.get("model")
    if not model_name:
        raise ValueError("Vector metadata does not contain model")
    docstore = load_jsonl(docstore_path)
    if not docstore:
        raise ValueError("Vector docstore is empty")

    import faiss
    import numpy as np

    if embedding_backend() == LOCAL_BACKEND:
        model = LocalCPUEmbedder(build_root, str(model_name))
    else:
        model = OpenAICompatibleEmbedder(str(model_name))
    index = faiss.read_index(str(index_path))
    expected_count = metadata.get("documents_indexed")
    if not isinstance(expected_count, int) or expected_count < 1:
        raise ValueError("Vector metadata documents_indexed is invalid")
    if index.ntotal != expected_count or len(docstore) != expected_count:
        raise ValueError(
            "Vector artifact count mismatch: "
            f"metadata={expected_count}, index={index.ntotal}, docstore={len(docstore)}"
        )
    if metadata.get("embedding_dimension") != index.d:
        raise ValueError(
            f"Vector artifact dimension mismatch: metadata={metadata.get('embedding_dimension')}, index={index.d}"
        )
    expected_index_hash = metadata.get("index_sha256")
    expected_docstore_hash = metadata.get("docstore_sha256")
    if not expected_index_hash or not expected_docstore_hash:
        raise ValueError("Vector metadata does not contain artifact checksums; rebuild the index")
    actual_index_hash = sha256_file(index_path)
    actual_docstore_hash = sha256_file(docstore_path)
    if actual_index_hash != expected_index_hash or actual_docstore_hash != expected_docstore_hash:
        raise ValueError("Vector artifact checksum mismatch; rebuild the index")
    expected_identity = metadata.get("model_identity")
    if not expected_identity:
        raise ValueError("Vector metadata does not contain model_identity; rebuild the index")
    if model.identity != expected_identity:
        raise ValueError(
            f"Embedding model identity mismatch: index={expected_identity}, query={model.identity}"
        )
    return model, index, docstore, np


def encode_queries(model: Any, index: Any, np: Any, texts: list[str]) -> Any:
    query_array = np.asarray(model.encode(texts), dtype="float32")
    if query_array.ndim != 2 or query_array.shape[1] != index.d:
        actual = query_array.shape[1] if query_array.ndim == 2 else "invalid"
        raise ValueError(f"Embedding dimension mismatch: index={index.d}, query={actual}")
    return query_array


def scores_from_search_results(docstore: list[dict[str, Any]], scores: Any, indices: Any) -> dict[str, float]:
    output: dict[str, float] = {}
    for score, doc_index in zip(scores[0], indices[0]):
        if doc_index < 0:
            continue
        chunk_id = normalize_scalar(docstore[int(doc_index)].get("chunk_id"))
        if chunk_id:
            output[chunk_id] = float(score)
    return output


def build_scores(build_root: Path, query: str) -> dict[str, float]:
    model, index, docstore, np = load_vector_runtime(build_root)
    query_array = encode_queries(model, index, np, [query])
    scores, indices = index.search(query_array, min(len(docstore), 50))
    return scores_from_search_results(docstore, scores, indices)


def build_batch_scores(build_root: Path, query_by_id: dict[str, str]) -> dict[str, dict[str, float]]:
    model, index, docstore, np = load_vector_runtime(build_root)
    query_ids = list(query_by_id)
    query_texts = [query_by_id[query_id] for query_id in query_ids]
    query_array = encode_queries(model, index, np, query_texts)
    scores, indices = index.search(query_array, min(len(docstore), 50))
    output: dict[str, dict[str, float]] = {}
    for offset, query_id in enumerate(query_ids):
        output[query_id] = scores_from_search_results(docstore, [scores[offset]], [indices[offset]])
    return output


def serve(build_root: Path) -> int:
    """Serve serial JSON-line queries while keeping the model and index resident."""
    try:
        model, index, docstore, np = load_vector_runtime(build_root)
    except Exception as exc:
        print(
            json.dumps(
                {"status": "startup_error", "error": f"{type(exc).__name__}: {exc}"},
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 0
    print(json.dumps({"status": "ready"}), flush=True)
    for line in sys.stdin:
        if not line.strip():
            continue
        request_id = ""
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("request must be a JSON object")
            request_id = normalize_scalar(request.get("request_id"))
            if isinstance(request.get("query"), str) and request["query"].strip():
                query_array = encode_queries(model, index, np, [request["query"]])
                scores, indices = index.search(query_array, min(len(docstore), 50))
                payload = {
                    "request_id": request_id,
                    "status": "ok",
                    "scores": scores_from_search_results(docstore, scores, indices),
                }
            elif isinstance(request.get("queries"), dict):
                query_by_id = {
                    normalize_scalar(key): normalize_scalar(value)
                    for key, value in request["queries"].items()
                    if normalize_scalar(key) and normalize_scalar(value)
                }
                if not query_by_id:
                    raise ValueError("queries did not contain usable values")
                query_ids = list(query_by_id)
                query_array = encode_queries(
                    model, index, np, [query_by_id[query_id] for query_id in query_ids]
                )
                scores, indices = index.search(query_array, min(len(docstore), 50))
                scores_by_query = {
                    query_id: scores_from_search_results(
                        docstore, [scores[offset]], [indices[offset]]
                    )
                    for offset, query_id in enumerate(query_ids)
                }
                payload = {
                    "request_id": request_id,
                    "status": "ok",
                    "scores_by_query": scores_by_query,
                }
            else:
                raise ValueError("request must contain query or queries")
        except Exception as exc:
            payload = {
                "request_id": request_id,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "scores": {},
            }
        print(json.dumps(payload, ensure_ascii=False), flush=True)
    return 0


def parse_queries_json(value: str) -> dict[str, str]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("--queries-json must be a JSON object")
    query_by_id: dict[str, str] = {}
    for key, query in parsed.items():
        query_id = normalize_scalar(key)
        query_text = normalize_scalar(query)
        if query_id and query_text:
            query_by_id[query_id] = query_text
    if not query_by_id:
        raise ValueError("--queries-json did not contain any usable queries")
    return query_by_id


def main() -> int:
    args = parse_args()
    if args.serve:
        return serve(Path(args.build_root))
    try:
        if args.query is not None:
            scores = build_scores(Path(args.build_root), args.query)
            print(json.dumps({"status": "ok", "scores": scores}, ensure_ascii=False))
        else:
            scores_by_query = build_batch_scores(Path(args.build_root), parse_queries_json(args.queries_json))
            print(json.dumps({"status": "ok", "scores_by_query": scores_by_query}, ensure_ascii=False))
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "scores": {},
                },
                ensure_ascii=False,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
