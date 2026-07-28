"""Build an isolated artifact from the public synthetic demo corpus."""

from __future__ import annotations

import importlib
import json
import os
import shutil
import sys
from pathlib import Path


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
RUNTIME_SCRIPTS = (
    "build_bm25_index.py",
    "retrieve.py",
    "unicode_han.py",
    "vector_query_worker.py",
)


def prepare_model(project_root: Path, model_dir: Path | None = None) -> Path:
    """Download the pinned local model during setup, never during API startup."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError("sentence-transformers/huggingface-hub is not installed") from error
    target = (model_dir or project_root / "kb_corpus_build" / ".cache" / "models" / "all-MiniLM-L6-v2").resolve()
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        DEFAULT_MODEL,
        revision=DEFAULT_MODEL_REVISION,
        local_dir=target,
        allow_patterns=[
            "*.json",
            "*.txt",
            "*.model",
            "*.safetensors",
            ".gitattributes",
            "1_Pooling/*",
        ],
        ignore_patterns=["onnx/*", "openvino/*"],
    )
    return target


def _load_build_modules(project_root: Path):
    scripts = (project_root / "kb_corpus_build" / "scripts").resolve()
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    return importlib.import_module("build_bm25_index"), importlib.import_module("build_vector_index")


def _validate_demo_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or any(row.get("source_id") != "synthetic_demo" for row in rows):
        raise ValueError("demo corpus must contain only synthetic_demo rows")
    chunk_ids = [row.get("chunk_id") for row in rows]
    if any(not isinstance(chunk_id, str) or not chunk_id for chunk_id in chunk_ids):
        raise ValueError("every demo row must have a non-empty chunk_id")
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("demo chunk_id values must be unique")
    return rows


def build_demo(project_root: Path, output_root: Path) -> Path:
    project_root = project_root.resolve()
    output_root = output_root.resolve()
    if output_root == project_root or project_root / "reference" == output_root:
        raise ValueError("demo output must be an isolated directory outside reference/")
    try:
        output_root.relative_to(project_root / "reference")
    except ValueError:
        pass
    else:
        raise ValueError("demo output must not be inside reference/")

    source = project_root / "demo" / "chunks.synthetic.jsonl"
    rows = _validate_demo_rows(source)
    build_root = output_root / "kb_corpus_build"
    configured_model = os.environ.get("EM_RAG_LOCAL_EMBEDDING_MODEL", "").strip()
    source_model = (
        Path(configured_model).expanduser()
        if configured_model
        else project_root / "kb_corpus_build" / ".cache" / "models" / "all-MiniLM-L6-v2"
    )
    demo_model = build_root / ".cache" / "models" / "all-MiniLM-L6-v2"
    if not source_model.is_dir():
        raise RuntimeError(
            "Prepared local model is missing; run `python -m em_rag.bootstrap prepare-model --project-root .` first."
        )
    if not demo_model.exists():
        try:
            shutil.copytree(source_model, demo_model, copy_function=os.link)
        except OSError:
            shutil.copytree(source_model, demo_model)
    corpus_path = build_root / "corpus" / "chunks.canonical.jsonl"
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    corpus_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    scripts_source = project_root / "kb_corpus_build" / "scripts"
    scripts_target = build_root / "scripts"
    scripts_target.mkdir(parents=True, exist_ok=True)
    for name in RUNTIME_SCRIPTS:
        shutil.copy2(scripts_source / name, scripts_target / name)

    bm25, vector = _load_build_modules(project_root)
    bm25_rows = bm25.load_jsonl(corpus_path)
    bm25_index = bm25.build_bm25_index(bm25_rows)
    bm25_dir = build_root / "indexes" / "bm25"
    bm25.write_pickle_checked(bm25_dir / "bm25_index.pkl", bm25_index)
    bm25.write_jsonl_checked(bm25_dir / "bm25_docstore.jsonl", bm25.build_docstore(bm25_rows))
    try:
        vector_status = vector.build_vector_index(output_root, DEFAULT_MODEL)
        if vector_status != "completed":
            raise RuntimeError(f"demo vector index build did not complete: {vector_status}")
    except Exception:
        # A partially built demo must never be reusable as a ready artifact.
        # In particular, remove a vector index left by a prior attempt before
        # propagating the failure to the CLI's non-zero exit path.
        shutil.rmtree(build_root / "indexes", ignore_errors=True)
        raise

    quality = build_root / "eval" / "retrieval_quality_v2" / "reports" / "judgment_release_status.json"
    quality.parent.mkdir(parents=True, exist_ok=True)
    quality.write_text(
        json.dumps(
            {
                "status": "DEMO_ONLY",
                "description": "Synthetic startup data; not a retrieval-quality evaluation artifact.",
                "documents": len(rows),
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return output_root
