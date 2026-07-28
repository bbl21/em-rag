#!/usr/bin/env python3
"""Preflight the optional vector runtime before full hybrid evaluation."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable


DEFAULT_QUERY = "What is the frequency range of ITU-R P.1411?"
DEFAULT_OUTPUT_DIR = Path("kb_corpus_build/audit/vector_runtime_preflight")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check vector import, model, encode, and FAISS search readiness.")
    parser.add_argument("--project-root", default=".", help="Project root containing kb_corpus_build.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR.as_posix(), help="Directory for preflight reports.")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Single query used for encode/search preflight.")
    return parser.parse_args()


def normalize_scalar(value: Any) -> str:
    return str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def safe_display_path(path: str | Path, project_root: Path) -> str:
    candidate = Path(path)
    try:
        return candidate.absolute().relative_to(project_root.absolute()).as_posix()
    except ValueError:
        pass
    try:
        return candidate.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return candidate.as_posix()


def write_text_checked(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    if path.read_text(encoding="utf-8") != text:
        raise ValueError(f"UTF-8 roundtrip mismatch for {path.as_posix()}")


def is_windows_mount_path(path: Path) -> bool:
    return path.as_posix().lower().startswith("/mnt/c/")


def is_temp_cache_path(path: Path) -> bool:
    try:
        return path.resolve().is_relative_to(Path(tempfile.gettempdir()).resolve())
    except ValueError:
        return False


def default_hf_home(project_root: Path) -> Path:
    return project_root / "kb_corpus_build" / ".cache" / "huggingface"


def configure_runtime(project_root: Path) -> Path:
    hf_home = Path(os.environ.get("HF_HOME") or default_hf_home(project_root))
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    return hf_home


def add_step(
    steps: list[dict[str, Any]],
    name: str,
    status: str,
    seconds: float,
    error: str = "",
    **extra: Any,
) -> None:
    row = {"name": name, "status": status, "seconds": round(seconds, 3)}
    if error:
        row["error"] = error
    row.update(extra)
    steps.append(row)


def run_step(steps: list[dict[str, Any]], name: str, callback: Callable[[], dict[str, Any] | None]) -> bool:
    started = time.perf_counter()
    try:
        extra = callback() or {}
    except Exception as exc:
        add_step(steps, name, "error", time.perf_counter() - started, f"{type(exc).__name__}: {exc}")
        return False
    add_step(steps, name, "ok", time.perf_counter() - started, **extra)
    return True


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if isinstance(row, dict):
            rows.append(row)
    return rows


def build_markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Vector Runtime Preflight",
        "",
        f"- status: `{payload['status']}`",
        f"- error: {normalize_scalar(payload.get('error')) or '<none>'}",
        f"- hf_home: `{normalize_scalar(payload.get('hf_home'))}`",
        f"- python_executable: `{normalize_scalar(payload.get('python_executable'))}`",
        "",
        "| step | status | seconds | error |",
        "|---|---|---:|---|",
    ]
    for step in payload.get("steps") or []:
        lines.append(
            f"| `{normalize_scalar(step.get('name'))}` | `{normalize_scalar(step.get('status'))}` | "
            f"{step.get('seconds', 0.0)} | {normalize_scalar(step.get('error')) or '<none>'} |"
        )
    lines.append("")
    return "\n".join(lines)


def run_preflight(project_root: Path, query: str) -> dict[str, Any]:
    build_root = project_root / "kb_corpus_build"
    vector_dir = build_root / "indexes" / "vector"
    steps: list[dict[str, Any]] = []
    payload: dict[str, Any] = {
        "status": "ok",
        "error": "",
        "steps": steps,
        "python_executable": safe_display_path(sys.executable, project_root),
        "python_prefix": safe_display_path(sys.prefix, project_root),
        "hf_home": "",
    }

    def runtime_location() -> dict[str, Any]:
        allow_mntc = os.environ.get("EM_RAG_ALLOW_MNTC_VECTOR_RUNTIME", "").strip().lower() in {"1", "true", "yes"}
        prefix = Path(sys.prefix)
        executable = Path(sys.executable)
        if (is_windows_mount_path(prefix) or is_windows_mount_path(executable)) and not allow_mntc:
            raise RuntimeError(
                "Vector runtime is under /mnt/c; move the vector venv to WSL ext4 or set "
                "EM_RAG_ALLOW_MNTC_VECTOR_RUNTIME=1 for a controlled experiment."
            )
        return {
            "python_prefix": safe_display_path(sys.prefix, project_root),
            "python_executable": safe_display_path(sys.executable, project_root),
        }

    if not run_step(steps, "runtime_location", runtime_location):
        payload["status"] = "error"
        payload["error"] = normalize_scalar(steps[-1].get("error"))
        return payload

    hf_home = configure_runtime(project_root)
    payload["hf_home"] = safe_display_path(hf_home, project_root)

    def cache_location() -> dict[str, Any]:
        allow_tmp = os.environ.get("EM_RAG_ALLOW_TMP_HF_CACHE", "").strip().lower() in {"1", "true", "yes"}
        if is_temp_cache_path(hf_home) and not allow_tmp:
            raise RuntimeError("HF_HOME is under a temporary directory; use a persistent Hugging Face cache.")
        return {"hf_home": safe_display_path(hf_home, project_root)}

    if not run_step(steps, "cache_location", cache_location):
        payload["status"] = "error"
        payload["error"] = normalize_scalar(steps[-1].get("error"))
        return payload

    def index_files() -> dict[str, Any]:
        metadata_path = vector_dir / "index_metadata.json"
        docstore_path = vector_dir / "docstore.jsonl"
        index_path = vector_dir / "faiss.index"
        missing = [path.as_posix() for path in [metadata_path, docstore_path, index_path] if not path.is_file()]
        if missing:
            raise FileNotFoundError(", ".join(missing))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        docstore = load_jsonl(docstore_path)
        if not metadata.get("model"):
            raise ValueError("Vector metadata does not contain model")
        if not docstore:
            raise ValueError("Vector docstore is empty")
        payload["model"] = metadata["model"]
        payload["documents_indexed"] = len(docstore)
        return {"model": metadata["model"], "documents_indexed": len(docstore)}

    if not run_step(steps, "index_files", index_files):
        payload["status"] = "error"
        payload["error"] = normalize_scalar(steps[-1].get("error"))
        return payload

    runtime_modules: dict[str, Any] = {}

    def import_runtime() -> None:
        from sentence_transformers import SentenceTransformer
        import faiss
        import numpy as np

        runtime_modules["SentenceTransformer"] = SentenceTransformer
        runtime_modules["faiss"] = faiss
        runtime_modules["np"] = np
        return None

    if not run_step(steps, "import_runtime", import_runtime):
        payload["status"] = "error"
        payload["error"] = normalize_scalar(steps[-1].get("error"))
        return payload

    model_holder: dict[str, Any] = {}

    def model_load() -> None:
        sentence_transformer = runtime_modules["SentenceTransformer"]
        model_name = str(payload["model"])
        try:
            model_holder["model"] = sentence_transformer(model_name, local_files_only=True)
        except TypeError:
            model_holder["model"] = sentence_transformer(model_name)
        return None

    if not run_step(steps, "model_load", model_load):
        payload["status"] = "error"
        payload["error"] = normalize_scalar(steps[-1].get("error"))
        return payload

    vector_holder: dict[str, Any] = {}

    def single_encode() -> dict[str, Any]:
        vector = model_holder["model"].encode([query], normalize_embeddings=True, show_progress_bar=False)
        array = runtime_modules["np"].asarray(vector, dtype="float32")
        if array.ndim != 2 or array.shape[0] != 1:
            raise ValueError(f"Unexpected query vector shape: {array.shape}")
        vector_holder["array"] = array
        return {"embedding_dimension": int(array.shape[1])}

    if not run_step(steps, "single_encode", single_encode):
        payload["status"] = "error"
        payload["error"] = normalize_scalar(steps[-1].get("error"))
        return payload

    def faiss_read_search() -> dict[str, Any]:
        faiss = runtime_modules["faiss"]
        index = faiss.read_index(str(vector_dir / "faiss.index"))
        scores, indices = index.search(vector_holder["array"], 1)
        if len(indices[0]) < 1 or int(indices[0][0]) < 0:
            raise ValueError("FAISS search returned no result")
        return {"top_score": float(scores[0][0]), "top_index": int(indices[0][0])}

    if not run_step(steps, "faiss_read_search", faiss_read_search):
        payload["status"] = "error"
        payload["error"] = normalize_scalar(steps[-1].get("error"))
        return payload

    return payload


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = project_root / output_root
    payload = run_preflight(project_root, args.query)
    write_text_checked(output_root / "vector_runtime_preflight.json", json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    write_text_checked(output_root / "vector_runtime_preflight.md", build_markdown_report(payload))
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
