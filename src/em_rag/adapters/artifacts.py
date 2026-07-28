"""Read-only artifact discovery and identity checks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ArtifactSnapshot:
    ready: bool
    artifact_id: str
    missing: tuple[str, ...]
    capabilities: dict[str, bool]
    quality_status: str
    document_count: int
    checksums: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["missing"] = list(self.missing)
        return value


class ArtifactStore:
    """Inspects the immutable runtime inputs without mutating build outputs."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.resolve()
        self.build_root = self.project_root / "kb_corpus_build"
        self._cache_key: tuple[tuple[str, int, int], ...] | None = None
        self._cache: ArtifactSnapshot | None = None

    @property
    def required_paths(self) -> dict[str, Path]:
        return {
            "bm25_index": self.build_root / "indexes" / "bm25" / "bm25_index.pkl",
            "bm25_docstore": self.build_root / "indexes" / "bm25" / "bm25_docstore.jsonl",
            "vector_index": self.build_root / "indexes" / "vector" / "faiss.index",
            "vector_metadata": self.build_root / "indexes" / "vector" / "index_metadata.json",
            "vector_docstore": self.build_root / "indexes" / "vector" / "docstore.jsonl",
        }

    @property
    def structured_paths(self) -> dict[str, Path]:
        structured = self.build_root / "indexes" / "structured"
        return {
            "structured_formula": structured / "formula.sqlite",
            "structured_terms": structured / "terms.sqlite",
            "structured_propagation_models": structured / "propagation_models.sqlite",
        }

    def _state_key(self) -> tuple[tuple[str, int, int], ...]:
        rows = []
        observed_paths = {
            **self.required_paths,
            **self.structured_paths,
            "quality_status": self.build_root
            / "eval"
            / "retrieval_quality_v2"
            / "reports"
            / "judgment_release_status.json",
        }
        for name, path in observed_paths.items():
            if path.is_file():
                stat = path.stat()
                rows.append((name, stat.st_size, stat.st_mtime_ns))
            else:
                rows.append((name, -1, -1))
        return tuple(rows)

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _quality_status(self) -> str:
        path = self.build_root / "eval" / "retrieval_quality_v2" / "reports" / "judgment_release_status.json"
        if not path.is_file():
            return "needs_validation"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return "needs_validation"
        status = value.get("status") if isinstance(value, dict) else None
        return str(status or "needs_validation").lower()

    def snapshot(self) -> ArtifactSnapshot:
        key = self._state_key()
        if key == self._cache_key and self._cache is not None:
            return self._cache
        missing = tuple(name for name, path in self.required_paths.items() if not path.is_file())
        checksums = {
            name: self._sha256(path)
            for name, path in self.required_paths.items()
            if path.is_file()
        }
        identity = hashlib.sha256(
            "\n".join(f"{name}:{checksums[name]}" for name in sorted(checksums)).encode("utf-8")
        ).hexdigest()[:16]
        docstore = self.required_paths["bm25_docstore"]
        document_count = 0
        if docstore.is_file():
            with docstore.open("r", encoding="utf-8") as handle:
                document_count = sum(1 for line in handle if line.strip())
        vector = self.build_root / "indexes" / "vector"
        snapshot = ArtifactSnapshot(
            ready=not missing,
            artifact_id=f"em-rag-{identity}" if not missing else "unavailable",
            missing=missing,
            capabilities={
                "bm25": all(
                    self.required_paths[key].is_file() for key in ("bm25_index", "bm25_docstore")
                ),
                "structured": all(path.is_file() for path in self.structured_paths.values()),
                "vector": all(
                    (vector / name).is_file()
                    for name in ("faiss.index", "index_metadata.json", "docstore.jsonl")
                ),
            },
            quality_status=self._quality_status(),
            document_count=document_count,
            checksums=checksums,
        )
        self._cache_key = key
        self._cache = snapshot
        return snapshot
