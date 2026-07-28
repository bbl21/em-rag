# EM RAG

EM RAG is a source-code release for local, evidence-grounded retrieval in
electromagnetics, antennas, microwave circuits, and short-range propagation.
It provides a FastAPI service, a small local web UI, deterministic BM25, and
mandatory local CPU FAISS retrieval. An LLM is optional and is never required
for retrieval.

## What this release contains

The repository contains source code, tests, a synthetic five-document demo,
release checks, and documentation. It does **not** contain reference documents,
real chunks, indexes, model caches, or controlled evaluation data.

The product runtime is verified; retrieval quality is not yet claimed as
validated. API responses and documentation expose this honestly as
`NEEDS_CALIBRATION`. Queries are English-only and retrieved text is untrusted
evidence, never executable instructions.

## Quick start: verified synthetic demo

Use Python 3.11 or newer. This path needs no reference documents and proves
the complete BM25 + FAISS + hybrid retrieval path.

```bash
python -m venv .venv
# POSIX: . .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install ".[runtime,test]"
python -m em_rag.bootstrap prepare-model --project-root .
python -m em_rag.bootstrap demo --project-root . --output .em-rag-demo
python -m em_rag.bootstrap verify --project-root .em-rag-demo
```

Start the service:

```bash
# POSIX
EM_RAG_PROJECT_ROOT="$PWD/.em-rag-demo" python -m em_rag.api

# Windows PowerShell
$env:EM_RAG_PROJECT_ROOT=(Join-Path (Get-Location) '.em-rag-demo')
python -m em_rag.api
```

Open <http://127.0.0.1:8000> or <http://127.0.0.1:8000/docs>. A `GET
/health/ready` response must be HTTP 200 with `bm25_runtime_status` and
`vector_runtime_status` both `ok`. A hybrid request for `What does S11
represent?` returns `demo_s11` without degradation.

## Build a real local artifact

Only use this path after independently obtaining the four inputs listed in
[docs/data-sources.md](docs/data-sources.md) and reviewing their terms. The
catalogue currently prints manual acquisition instructions for restricted or
un-pinned sources; it never downloads source material at API startup.

```bash
python -m pip install ".[runtime,build]"
python -m em_rag.bootstrap acquire-sources --project-root .
python -m em_rag.bootstrap prepare-model --project-root .
python -m em_rag.bootstrap build --project-root .
python -m em_rag.bootstrap verify --project-root .
```

`build` fails closed when required inputs, the local model, BM25, or FAISS are
missing or inconsistent. Keep `reference/` read-only; generated local data
stays under `kb_corpus_build/` and must not be committed or redistributed.

## API

- `GET /health/live` — process liveness.
- `GET /health/ready` — fail-closed artifact and runtime validation.
- `GET /v1/system` — capabilities and release status.
- `POST /v1/retrieve` — citation-bearing BM25/FAISS hybrid evidence.
- `POST /v1/answer` — optional OpenAI-compatible answer provider.

The default embedding model is the pinned local
`sentence-transformers/all-MiniLM-L6-v2` revision. API startup never downloads
models. `/v1/answer` returns `provider_not_configured` until an explicit
provider configuration is supplied.

## Development and source release checks

```bash
PYTHONPATH=src python -m unittest discover -s tests/product -p "test_*.py" -v
PYTHONPATH=src python -m unittest discover -s tests/evaluation -p "test_*.py"
python scripts/export_release_tree.py --output dist/em-rag-release
python scripts/check_release_tree.py --project-root dist/em-rag-release --all-files
```

GitHub Actions performs the same source-release checks, including preparation
of the pinned model and a fresh demo end-to-end test. Docker and Compose files
are retained as optional development material; they are not part of this source
release's deployment or publication gate.

## Boundaries and further reading

- [Data-source and license boundary](docs/data-sources.md)
- [Public MVP runtime contract](docs/public-rag-mvp-spec.md)
- [Release policy](docs/product-release-policy-2026-07-21.md)

Do not describe this release as production-ready or retrieval-quality
validated. Its purpose is a reproducible source release with a functioning
local retrieval runtime and transparent limitations.
