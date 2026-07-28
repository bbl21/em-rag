"""Environment-backed product configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True)
class Settings:
    project_root: Path
    retrieval_mode: str = "hybrid"
    default_top_k: int = 8
    max_top_k: int = 50
    provider_base_url: str = ""
    provider_api_key: str = ""
    provider_model: str = ""
    provider_timeout_seconds: int = 30
    embedding_backend: str = "local_cpu"

    @property
    def provider_configured(self) -> bool:
        return bool(self.provider_base_url and self.provider_model)

    @classmethod
    def from_env(cls) -> "Settings":
        project_root = Path(os.environ.get("EM_RAG_PROJECT_ROOT", Path.cwd())).resolve()
        mode = os.environ.get("EM_RAG_RETRIEVAL_MODE", "hybrid").strip()
        if mode not in {"bm25_structured", "hybrid"}:
            raise ValueError("EM_RAG_RETRIEVAL_MODE must be bm25_structured or hybrid")
        default_top_k = _positive_int("EM_RAG_DEFAULT_TOP_K", 8)
        max_top_k = _positive_int("EM_RAG_MAX_TOP_K", 50)
        if default_top_k > max_top_k:
            raise ValueError("EM_RAG_DEFAULT_TOP_K must not exceed EM_RAG_MAX_TOP_K")
        embedding_backend = os.environ.get("EM_RAG_EMBEDDING_BACKEND", "local_cpu").strip().lower()
        if embedding_backend not in {"local_cpu", "openai_compatible"}:
            raise ValueError("EM_RAG_EMBEDDING_BACKEND must be local_cpu or openai_compatible")
        return cls(
            project_root=project_root,
            retrieval_mode=mode,
            default_top_k=default_top_k,
            max_top_k=max_top_k,
            provider_base_url=os.environ.get("EM_RAG_OPENAI_BASE_URL", "").strip(),
            provider_api_key=os.environ.get("EM_RAG_OPENAI_API_KEY", "").strip(),
            provider_model=os.environ.get("EM_RAG_OPENAI_MODEL", "").strip(),
            provider_timeout_seconds=_positive_int("EM_RAG_PROVIDER_TIMEOUT_SECONDS", 30),
            embedding_backend=embedding_backend,
        )
