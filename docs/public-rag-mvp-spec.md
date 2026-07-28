# Public RAG MVP specification

## Purpose

The public MVP must prove that its retrieval product works end to end.  A
successful process exit, a BM25-only fallback, or a synthetic artifact with
missing vector files is not a successful public demo.

## Runtime contract

- BM25 and local CPU vector retrieval are mandatory runtime capabilities.
- The only default vector model is `sentence-transformers/all-MiniLM-L6-v2` at
  the pinned revision already used by the project.
- `faiss-cpu` and `sentence-transformers` are runtime dependencies, not an
  optional feature required only by a separate installation profile.
- API startup never downloads models or source documents.
- Model preparation is explicit.  The model identity and vector dimension must
  match the generated artifact.

## Bootstrap contract

- `prepare-model` is the explicit network-enabled model preparation command.
- `demo` and `build` fail with a non-zero exit status if a vector index cannot
  be produced and verified.
- Failed bootstrap work must not be presented as a ready artifact.
- A ready artifact requires readable BM25 index/docstore, FAISS index/vector
  docstore/metadata, matching checksums, and a successful vector query probe.

## Public demo acceptance test

The release gate must create a fresh isolated demo artifact and verify:

1. `bootstrap demo` succeeds.
2. `bootstrap verify` reports `ready: true`.
3. BM25 and vector runtime statuses are both `ok`.
4. A hybrid query for `What does S11 represent?` returns `demo_s11` and is not
   degraded.
5. The source-release CI prepares the pinned model and runs the same checks
   from a fresh checkout. Docker is explicitly outside this MVP's release
   contract.

## Data boundary

The demo contains synthetic, non-authoritative documents only.  Real source
documents and all derived corpus, index, cache, and controlled evaluation data
remain local.  Retrieval quality remains transparently labelled
`NEEDS_CALIBRATION`; this does not waive any runtime acceptance requirement.
