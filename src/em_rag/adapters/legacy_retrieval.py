"""Compatibility adapter around the verified local BM25/structured retriever."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sqlite3
import sys
import time
from contextlib import closing
from pathlib import Path
from types import ModuleType
from typing import Any

from em_rag.adapters.vector_client import PersistentVectorClient
from em_rag.domain.errors import artifact_not_ready, retrieval_failed


class LegacyRetrievalAdapter:
    """Keeps the product boundary stable while the legacy scorer is replaced incrementally."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.resolve()
        # The product contract requires vector retrieval. The legacy CLI guard
        # remains for historical evaluation scripts, but the product adapter
        # must never silently inherit its disabled default.
        os.environ["EM_RAG_ENABLE_VECTOR"] = "1"
        self._module: ModuleType | None = None
        self._docstore_key: tuple[int, int] | None = None
        self._docstore: dict[str, dict[str, Any]] = {}
        self._vector_preflight: tuple[float, str, tuple[Any, ...]] | None = None
        self._bm25_preflight: tuple[float, str, tuple[Any, ...]] | None = None
        self._structured_preflight: tuple[float, str, tuple[Any, ...]] | None = None
        self._vector_client: PersistentVectorClient | None = None
        self._vector_client_signature: tuple[Any, ...] | None = None

    def _load_module(self) -> ModuleType:
        if self._module is not None:
            return self._module
        scripts = self.project_root / "kb_corpus_build" / "scripts"
        path = scripts / "retrieve.py"
        if not path.is_file():
            raise artifact_not_ready("The local retrieval adapter is missing.")
        if str(scripts) not in sys.path:
            sys.path.insert(0, str(scripts))
        spec = importlib.util.spec_from_file_location("em_rag_legacy_retrieve", path)
        if spec is None or spec.loader is None:
            raise retrieval_failed("Unable to load the local retrieval adapter.")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._module = module
        return module

    def _load_docstore(self) -> dict[str, dict[str, Any]]:
        path = self.project_root / "kb_corpus_build" / "indexes" / "bm25" / "bm25_docstore.jsonl"
        if not path.is_file():
            raise artifact_not_ready("The BM25 docstore is not available.")
        stat = path.stat()
        key = (stat.st_size, stat.st_mtime_ns)
        if key == self._docstore_key:
            return self._docstore
        rows: dict[str, dict[str, Any]] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise artifact_not_ready(f"BM25 docstore line {line_number} is invalid JSON.") from error
                chunk_id = row.get("chunk_id") if isinstance(row, dict) else None
                if not isinstance(chunk_id, str) or not chunk_id:
                    raise artifact_not_ready(f"BM25 docstore line {line_number} has no chunk_id.")
                rows[chunk_id] = row
        self._docstore = rows
        self._docstore_key = key
        return rows

    @staticmethod
    def _file_signature(path: Path) -> tuple[str, int, int]:
        if not path.exists():
            return (path.as_posix(), 0, 0)
        stat = path.stat()
        return (path.as_posix(), int(stat.st_size), int(stat.st_mtime_ns))

    def _vector_signature(self, module: ModuleType) -> tuple[Any, ...]:
        build_root = self.project_root / "kb_corpus_build"
        vector_dir = build_root / "indexes" / "vector"
        index_path = vector_dir / "faiss.index"
        metadata_path = vector_dir / "index_metadata.json"
        docstore_path = vector_dir / "docstore.jsonl"

        metadata: dict[str, Any] = {}
        if metadata_path.is_file():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                metadata = {}

        try:
            backend = str(module.embedding_backend())
        except Exception:
            backend = "invalid_backend"

        if backend == "local_cpu":
            backend_signature = (
                backend,
                os.environ.get("EM_RAG_LOCAL_EMBEDDING_MODEL", "").strip(),
                metadata.get("model"),
            )
        else:
            backend_signature = (
                backend,
                os.environ.get("EM_RAG_EMBEDDINGS_BASE_URL", "").strip(),
                os.environ.get("EM_RAG_EMBEDDINGS_MODEL", "").strip(),
                os.environ.get("EM_RAG_EMBEDDINGS_MODEL_REVISION", "").strip(),
            )

        return (
            self._file_signature(self.project_root / "kb_corpus_build" / "scripts" / "vector_query_worker.py"),
            self._file_signature(index_path),
            self._file_signature(metadata_path),
            self._file_signature(docstore_path),
            (
                metadata.get("model"),
                metadata.get("model_identity"),
                metadata.get("embedding_backend_at_build"),
                metadata.get("embedding_dimension"),
                metadata.get("documents_indexed"),
                metadata.get("index_sha256"),
                metadata.get("docstore_sha256"),
            ),
            backend_signature,
            hashlib.sha256(
                os.environ.get("EM_RAG_EMBEDDINGS_API_KEY", "").encode("utf-8")
            ).hexdigest(),
            module.vector_timeout_seconds(),
        )

    def _bm25_signature(self) -> tuple[Any, ...]:
        bm25 = self.project_root / "kb_corpus_build" / "indexes" / "bm25"
        return (
            self._file_signature(bm25 / "bm25_index.pkl"),
            self._file_signature(bm25 / "bm25_docstore.jsonl"),
        )

    def _structured_signature(self) -> tuple[Any, ...]:
        structured = self.project_root / "kb_corpus_build" / "indexes" / "structured"
        return tuple(
            self._file_signature(structured / name)
            for name in ("formula.sqlite", "terms.sqlite", "propagation_models.sqlite")
        )

    @staticmethod
    def _cached_ok(
        cached: tuple[float, str, tuple[Any, ...]] | None,
        signature: tuple[Any, ...],
        now: float,
        max_age_seconds: int,
    ) -> bool:
        if cached is None:
            return False
        checked_at, status, cached_signature = cached
        return (
            status == "ok"
            and cached_signature == signature
            and now - checked_at <= max_age_seconds
        )

    def bm25_preflight(self, *, max_age_seconds: int = 300) -> tuple[str, str]:
        now = time.monotonic()
        signature = self._bm25_signature()
        if self._cached_ok(self._bm25_preflight, signature, now, max_age_seconds):
            return "ok", ""
        module = self._load_module()
        try:
            index, docstore = module.load_bm25(self.project_root / "kb_corpus_build")
            if not isinstance(index, dict) or not isinstance(docstore, dict) or not docstore:
                raise ValueError("BM25 artifacts have an invalid container type or are empty")
            documents = index.get("documents")
            document_count = index.get("doc_count")
            if not isinstance(documents, list) or document_count != len(documents):
                raise ValueError("BM25 index document count is inconsistent")
            index_ids = [str(row.get("chunk_id") or "") for row in documents if isinstance(row, dict)]
            if (
                len(index_ids) != document_count
                or len(set(index_ids)) != document_count
                or set(index_ids) != set(docstore)
            ):
                raise ValueError("BM25 index and docstore chunk IDs are inconsistent")
            probe = module.score_bm25(index, "electromagnetic antenna")
            if not isinstance(probe, list):
                raise ValueError("BM25 scorer returned an invalid result")
        except FileNotFoundError as error:
            self._bm25_preflight = None
            return "artifact_missing", f"{type(error).__name__}: {error}"
        except Exception as error:
            self._bm25_preflight = None
            return "error", f"{type(error).__name__}: {error}"
        self._bm25_preflight = (now, "ok", signature)
        return "ok", ""

    def structured_preflight(self, *, max_age_seconds: int = 300) -> tuple[str, str]:
        now = time.monotonic()
        signature = self._structured_signature()
        if self._cached_ok(self._structured_preflight, signature, now, max_age_seconds):
            return "ok", ""
        structured = self.project_root / "kb_corpus_build" / "indexes" / "structured"
        specs = (
            (structured / "formula.sqlite", "formulas"),
            (structured / "terms.sqlite", "terms"),
            (structured / "propagation_models.sqlite", "propagation_models"),
        )
        present = [(path, table) for path, table in specs if path.is_file()]
        if not present:
            self._structured_preflight = None
            return "unavailable", ""
        if len(present) != len(specs):
            self._structured_preflight = None
            return "partial", "One or more structured indexes are missing."
        try:
            docstore = self._load_docstore()
            for path, table in specs:
                with closing(
                    sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
                ) as connection:
                    integrity = connection.execute("pragma integrity_check").fetchone()
                    if not integrity or integrity[0] != "ok":
                        raise ValueError(f"{path.name} failed SQLite integrity_check")
                    columns = {
                        str(row[1])
                        for row in connection.execute(f"pragma table_info({table})").fetchall()
                    }
                    if "related_chunk_ids_json" not in columns or "search_text" not in columns:
                        raise ValueError(f"{path.name} does not contain the expected {table} schema")
                    rows = connection.execute(
                        f"select related_chunk_ids_json from {table}"
                    ).fetchall()
                    if not rows:
                        raise ValueError(f"{path.name} contains no {table} rows")
                    for (raw_ids,) in rows:
                        related = json.loads(raw_ids or "[]")
                        if not isinstance(related, list) or any(
                            not isinstance(chunk_id, str) or chunk_id not in docstore
                            for chunk_id in related
                        ):
                            raise ValueError(f"{path.name} contains an invalid chunk reference")
        except FileNotFoundError as error:
            self._structured_preflight = None
            return "artifact_missing", f"{type(error).__name__}: {error}"
        except Exception as error:
            self._structured_preflight = None
            return "error", f"{type(error).__name__}: {error}"
        self._structured_preflight = (now, "ok", signature)
        return "ok", ""

    def _query_vector(self, module: ModuleType, query: str) -> tuple[dict[str, float], str, str]:
        signature = self._vector_signature(module)
        if self._vector_client is not None and signature != self._vector_client_signature:
            self._vector_client.close()
            self._vector_client = None
        if self._vector_client is None:
            build_root = self.project_root / "kb_corpus_build"
            self._vector_client = PersistentVectorClient(
                worker_path=build_root / "scripts" / "vector_query_worker.py",
                build_root=build_root,
                environment=module.vector_worker_env(self.project_root),
                timeout_seconds=module.vector_timeout_seconds(),
            )
            self._vector_client_signature = signature
        return self._vector_client.query(query)

    def search(self, query: str, top_k: int, retrieval_mode: str) -> dict[str, Any]:
        module = self._load_module()
        try:
            normalized_query = module.normalize_scalar(query)
            use_vector = (
                retrieval_mode == "hybrid"
                and not module.contains_han(normalized_query)
                and not module.is_out_of_scope_query(normalized_query)
            )
            if use_vector:
                vector_scores, vector_status, vector_error = self._query_vector(
                    module, module.expand_query(normalized_query)
                )
                output = module.retrieve(
                    self.project_root,
                    query,
                    top_k,
                    retrieval_mode=retrieval_mode,
                    vector_scores_override=vector_scores,
                    vector_status_override=vector_status,
                    vector_error_override=vector_error,
                )
            else:
                output = module.retrieve(
                    self.project_root,
                    query,
                    top_k,
                    retrieval_mode=retrieval_mode,
                )
        except FileNotFoundError as error:
            raise artifact_not_ready(f"Required retrieval artifact is missing: {error.filename or error}") from error
        except Exception as error:
            if getattr(error, "code", None):
                raise
            raise retrieval_failed(f"Local retrieval failed: {type(error).__name__}") from error
        docstore = self._load_docstore()
        for row in output.get("results", []):
            source = docstore.get(str(row.get("chunk_id")), {})
            row["content"] = str(source.get("content_md") or source.get("retrieval_text") or row.get("content_preview") or "")
        return output

    def vector_preflight(self, *, max_age_seconds: int = 300) -> tuple[str, str]:
        now = time.monotonic()
        module = self._load_module()
        current_signature = self._vector_signature(module)

        if self._vector_preflight is not None:
            checked_at, status, signature = self._vector_preflight
            if signature != current_signature:
                self._vector_preflight = None
            elif status == "ok" and now - checked_at <= max_age_seconds:
                return status, ""

        _scores, status, error = self._query_vector(
            module,
            "What is the frequency range of ITU-R P.1411?",
        )
        status = str(status)
        if status == "ok":
            self._vector_preflight = (now, status, current_signature)
            return status, str(error)
        self._vector_preflight = None
        return status, str(error)

    def close(self) -> None:
        if self._vector_client is not None:
            self._vector_client.close()
        self._vector_client = None
        self._vector_client_signature = None
