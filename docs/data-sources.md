# Data sources

The public repository does not redistribute source documents or generated corpus/index files. Run the explicit acquisition step before building: it validates the public catalogue and only downloads entries with a reviewed HTTPS host, size limit, and SHA-256. Sources requiring user acceptance or a local version decision are never downloaded automatically; it prints their official URL and required local target. Checksums in local metadata must be regenerated from the exact downloaded version rather than copied across versions.

| source_id | pinned source | official page | local target | redistribution policy |
|---|---|---|---|---|
| `ellingson_em_vol1` | Steven W. Ellingson, *Electromagnetics, Volume 1* (2018) | https://doi.org/10.21061/electromagnetics-vol-1 | `reference/electromagnetics-vol-1-latex/` | CC BY-SA 4.0; retain attribution and ShareAlike terms |
| `modern_antennas_microwave_circuits` | arXiv:1911.08484 source bundle | https://arxiv.org/abs/1911.08484 | `reference/Modern_Antennas_Microwave_Circuits_Jan_2022_arxiv_latex/` | Do not redistribute from this repository until the selected arXiv version's license has been recorded and reviewed |
| `mit_em_applications` | MIT 6.013, *Electromagnetics and Applications*, Spring 2009 | https://ocw.mit.edu/courses/6-013-electromagnetics-and-applications-spring-2009/resources/readings/ | `reference/Electromagnetics-and-Applications.pdf` | Follow the MIT OpenCourseWare Creative Commons terms and attribution requirements; keep the input local |
| `itu_r_p1411_13` | Recommendation ITU-R P.1411-13 (09/2025) | https://www.itu.int/rec/R-REC-P.1411-13-202509-I/en | `reference/R-REC-P.1411-13-202509-I!!PDF-E.pdf` | Free download does not grant redistribution; ITU states reproduction requires written permission, so keep the input and extracted corpus local |

## Bootstrap

Review instructions and acquire catalogue-approved automatic sources:

```bash
PYTHONPATH=src python -m em_rag.bootstrap acquire-sources --project-root .
```

Place every source listed under `Manual source instructions` at its declared `reference/` target, then run `build`. The public catalogue is [`config/source-catalog.json`](../config/source-catalog.json); its current entries are manual because their publisher terms or exact-version checksums require a local user decision.

Prepare the pinned local embedding model once while network access is available. API startup never downloads a model:

```bash
PYTHONPATH=src python -m em_rag.bootstrap prepare-model --project-root .
```

The default snapshot is `sentence-transformers/all-MiniLM-L6-v2` revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Vector metadata records a SHA-256 identity over the actual local model files and the 384-dimensional index contract. Query-time loading must match both values. An Embeddings API build instead requires an explicit `EM_RAG_EMBEDDINGS_MODEL_REVISION`; the same model and revision must be configured for querying.

Bootstrap parsing requires source-document dependencies (`PyYAML`, `pypdf`, `PyMuPDF`) that are installed in the local environment via `requirements-runtime.txt` or the package extra `build`.

Verify an existing artifact:

```bash
PYTHONPATH=src python -m em_rag.bootstrap verify --project-root .
```

Build the CPU-first hybrid artifact after all source inputs are present. Vector indexing is mandatory and uses the Apache-2.0 `sentence-transformers/all-MiniLM-L6-v2` model:

```bash
PYTHONPATH=src python -m em_rag.bootstrap build --project-root .
```

The build never writes into `reference/`. Missing or mismatched inputs must stop the process rather than producing a partial ready artifact.

For a source-free startup check, `python -m em_rag.bootstrap demo --project-root . --output .em-rag-demo` builds a five-document synthetic artifact. It is product test data only, not part of the electromagnetic knowledge base and not evidence for engineering decisions.
