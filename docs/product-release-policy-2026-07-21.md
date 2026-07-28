# Product-first release policy

## Decision

On 2026-07-21 the user authorized the agent to make the remaining calibration decisions conservatively and changed the delivery order to: establish basic product capability, publish a transparent MVP, then continue retrieval-quality optimization.

This decision does not convert the retrieval gate to PASS. The aggregate quality status remains `NEEDS_CALIBRATION`, with 909 unresolved judgments, low human-agent calibration agreement, and six legacy `missing_required_facet` findings. These facts must remain visible in system metadata and release notes.

## Release contract

- Mandatory embedding backend: local CPU `sentence-transformers/all-MiniLM-L6-v2` plus FAISS.
- Optional backend: an explicitly configured OpenAI-compatible Embeddings API producing vectors compatible with the selected FAISS index.
- Default retrieval mode: hybrid vector + BM25 + structured retrieval.
- Readiness requires vector index, metadata, docstore, local model load, query encoding, and FAISS search to succeed.
- A transient vector failure may return BM25/structured evidence only with `degraded=true`; it may not report healthy readiness.
- An answer-provider API is optional. Evidence retrieval must work without an LLM or external key.
- Raw sources, generated corpus/indexes, model caches, holdout labels, judgment details, qrels, per-query runs, and secrets are excluded from the public source release.

## Agent substitution for human calibration work

The agent may perform disagreement categorization, rubric interpretation, and risk triage, but must preserve the existing human labels and audit trail. It may not relabel the remaining 909 rows as human gold or claim that quality thresholds passed. Calibration improvement remains a post-release backlog using an independent validation sample.

## Release blockers

The MVP may be published when the following product checks pass even while the retrieval-quality gate remains `NEEDS_CALIBRATION`:

1. product domain/application/API tests;
2. offline local CPU embedding and real FAISS search;
3. `/health/ready` returns 200 only after vector preflight;
4. hybrid `/v1/retrieve` returns cited evidence with a non-zero vector channel score;
5. UI, Docker/Compose, configuration, release-tree audit, and container smoke checks;
6. release notes explicitly disclose the quality backlog.

Post-release optimization must address the 909 unresolved judgments, the six blocking facet findings, independent calibration, and full 180-case metrics without rewriting historical baseline artifacts.
