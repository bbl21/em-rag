# Corpus build and private-data boundary

`kb_corpus_build/` contains the reproducible build code for the EM RAG
artifact. The public repository intentionally excludes source documents,
extracted text, chunks, retrieval indexes, model caches, per-query judgments,
and other controlled evaluation data.

The public MVP is usable for product and integration validation, but its
retrieval-quality status is `NEEDS_CALIBRATION`. This status is transparent in
the root README and API responses; this directory must not be read as a claim
that the retrieval-quality gate has passed.

## Public clone: run tests and the synthetic demo

No reference inputs are required for product tests or the synthetic demo:

```bash
python -m venv .venv
# POSIX: . .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install ".[runtime,vector,test]"
python -m unittest discover -s tests/product -p "test_*.py" -v
python -m em_rag.bootstrap demo --project-root . --output .em-rag-demo
```

The demo has five synthetic documents. It verifies the local BM25/FAISS,
readiness, retrieval, citation, and UI/API paths, but is not electromagnetic
engineering evidence.

## Private rebuild with authorized source inputs

Only users who have independently obtained the four inputs described in
[`../docs/data-sources.md`](../docs/data-sources.md), reviewed their terms, and
placed them below `reference/` may build a real local artifact. `reference/`
is read-only input; all generated output remains below `kb_corpus_build/`.

```bash
python -m pip install -r requirements-runtime.txt
PYTHONPATH=src python -m em_rag.bootstrap prepare-model --project-root .
PYTHONPATH=src python -m em_rag.bootstrap build --project-root .
PYTHONPATH=src python -m em_rag.bootstrap verify --project-root .
```

The build stops if required inputs or the pinned local model contract are
missing or inconsistent. It does not download source material and must not be
used to redistribute it.

## Build stages

The lower-level pipeline remains available for controlled local development:

```bash
python kb_corpus_build/scripts/run_pipeline.py --project-root . --stage all
```

Its inputs and generated corpus/index/report directories are deliberately
ignored by the public release tree. Consult `RAG-plan.md` for the stage
definitions. Retrieval output is untrusted evidence only: it is never executed
as an instruction, and engineering answers must retain citations and state
when evidence is insufficient.
