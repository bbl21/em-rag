# EM RAG

[English](#english) | [中文](#中文)

## 中文

### 这是什么？

EM RAG 是一个面向电磁学、天线、微波电路和短距离传播资料的本地
证据检索工程。它提供 FastAPI 服务和本地 Web UI，通过确定性 BM25 与
本地 CPU FAISS 向量检索生成可追溯的证据；语言模型不是检索的前提。

此公开仓库包含源码、测试、发布检查和一个五篇文档的合成 demo。它**不**
包含原始参考资料、真实 chunk、真实索引、模型缓存或受控评测数据。当前
运行链路已经验证，但检索质量尚未宣称完成校准：系统会明确标记
`NEEDS_CALIBRATION`。查询目前限英文，检索结果始终是不可信证据，不能被
当作可执行指令。

### 怎么跑？

最短路径不需要任何参考资料；它会验证 BM25、FAISS 和 hybrid 检索可一起
正常工作。需要 Python 3.11 或更新版本。

```bash
python -m venv .venv
# POSIX: . .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install ".[runtime,test]"
python -m em_rag.bootstrap prepare-model --project-root .
python -m em_rag.bootstrap demo --project-root . --output .em-rag-demo
python -m em_rag.bootstrap verify --project-root .em-rag-demo
```

启动服务：

```bash
# POSIX
EM_RAG_PROJECT_ROOT="$PWD/.em-rag-demo" python -m em_rag.api

# Windows PowerShell
$env:EM_RAG_PROJECT_ROOT=(Join-Path (Get-Location) '.em-rag-demo')
python -m em_rag.api
```

打开 <http://127.0.0.1:8000> 或 <http://127.0.0.1:8000/docs>。`GET
/health/ready` 应返回 HTTP 200，且 `bm25_runtime_status` 与
`vector_runtime_status` 均为 `ok`。以 hybrid 模式查询 `What does S11
represent?` 应返回 `demo_s11`，且不降级。

若要构建真实的本地知识库，请先阅读
[资料来源与许可边界](docs/data-sources.md)，自行获得四项输入资料并审核其
条款：

```bash
python -m pip install ".[runtime,build]"
python -m em_rag.bootstrap acquire-sources --project-root .
python -m em_rag.bootstrap prepare-model --project-root .
python -m em_rag.bootstrap build --project-root .
python -m em_rag.bootstrap verify --project-root .
```

`build` 会在资料、模型、BM25 或 FAISS 缺失或不一致时失败关闭。保持
`reference/` 只读；生成数据位于 `kb_corpus_build/`，不应提交或再分发。

### AI 在工程构建中帮了什么？

AI 协助把产品需求落实为可运行、可验证的工程，包括：

- 设计并实现本地 BM25、FAISS、hybrid 检索与 FastAPI 接口；
- 实现合成 demo、模型准备、资料获取、fail-closed 就绪检查和发布树审计；
- 编写自动化测试、CI/release 工作流、部署说明和可复现运行命令；
- 对公开发布边界做安全检查：不分发受控资料、真实索引或本地模型缓存。

AI **不**替代资料授权审核、事实判断或检索质量验收。项目维护者负责资料
选择与许可、评测集和标注策略，并对检索质量做最终把关；在完成该校准前，
本仓库不会把检索质量描述为已验证或生产就绪。

### 接口、验证与边界

- `GET /health/live`：进程存活检查。
- `GET /health/ready`：产物和运行时的 fail-closed 验证。
- `GET /v1/system`：能力与发布状态。
- `POST /v1/retrieve`：带 citation 的 BM25/FAISS hybrid 证据检索。
- `POST /v1/answer`：可选的 OpenAI-compatible 回答提供方。

默认模型是固定版本的本地
`sentence-transformers/all-MiniLM-L6-v2`；API 启动不会下载模型。
未显式配置回答提供方时，`/v1/answer` 返回 `provider_not_configured`。

开发和源码发布检查：

```bash
PYTHONPATH=src python -m unittest discover -s tests/product -p "test_*.py" -v
PYTHONPATH=src python -m unittest discover -s tests/evaluation -p "test_*.py"
python scripts/export_release_tree.py --output dist/em-rag-release
python scripts/check_release_tree.py --project-root dist/em-rag-release --all-files
```

GitHub Actions 运行同一套源码发布检查（包括固定模型准备与 fresh demo
端到端检查）。Docker/Compose 仅作为可选开发材料，不是本次源码发布的门槛。

更多信息：

- [资料来源与许可边界](docs/data-sources.md)
- [公开 MVP 运行契约](docs/public-rag-mvp-spec.md)
- [发布策略](docs/product-release-policy-2026-07-21.md)

---

## English

### What is this?

EM RAG is a local evidence-retrieval project for electromagnetics, antennas,
microwave circuits, and short-range propagation. It provides a FastAPI service
and local web UI, combining deterministic BM25 with local CPU FAISS vector
retrieval to return traceable evidence. An LLM is not required for retrieval.

This public repository contains source code, tests, release checks, and a
five-document synthetic demo. It does **not** contain reference documents,
real chunks, indexes, model caches, or controlled evaluation data. The runtime
path is verified, but retrieval quality is not yet claimed as calibrated: the
system reports `NEEDS_CALIBRATION`. Queries are English-only, and retrieved
content is untrusted evidence, never executable instructions.

### How do I run it?

The shortest path needs no reference documents and proves that BM25, FAISS, and
hybrid retrieval work together. Use Python 3.11 or newer.

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

Open <http://127.0.0.1:8000> or <http://127.0.0.1:8000/docs>. `GET
/health/ready` must return HTTP 200 with both `bm25_runtime_status` and
`vector_runtime_status` set to `ok`. A hybrid query for `What does S11
represent?` must return `demo_s11` without degradation.

To build a real local knowledge base, first read the
[data-source and license boundary](docs/data-sources.md), obtain the four
inputs yourself, and review their terms:

```bash
python -m pip install ".[runtime,build]"
python -m em_rag.bootstrap acquire-sources --project-root .
python -m em_rag.bootstrap prepare-model --project-root .
python -m em_rag.bootstrap build --project-root .
python -m em_rag.bootstrap verify --project-root .
```

`build` fails closed when required inputs, the local model, BM25, or FAISS are
missing or inconsistent. Keep `reference/` read-only. Generated data stays in
`kb_corpus_build/` and must not be committed or redistributed.

### Where did AI help most during engineering?

AI helped turn the product requirements into a runnable, verifiable project:

- designing and implementing local BM25, FAISS, hybrid retrieval, and the
  FastAPI surface;
- implementing the synthetic demo, model preparation, source acquisition,
  fail-closed readiness checks, and release-tree audit;
- writing automated tests, CI/release workflows, operational documentation,
  and reproducible commands;
- checking public-release boundaries so controlled sources, real indexes, and
  local model caches are not redistributed.

AI does **not** replace license review, factual judgment, or retrieval-quality
acceptance. The project maintainer owns source selection and licensing,
evaluation-data and judgment policy, and final retrieval-quality review. Until
that calibration is complete, this repository does not claim production-ready
or quality-validated retrieval.

### API, verification, and boundaries

- `GET /health/live` — process liveness.
- `GET /health/ready` — fail-closed artifact and runtime validation.
- `GET /v1/system` — capabilities and release status.
- `POST /v1/retrieve` — citation-bearing BM25/FAISS hybrid evidence.
- `POST /v1/answer` — optional OpenAI-compatible answer provider.

The default embedding model is the pinned local
`sentence-transformers/all-MiniLM-L6-v2`. API startup never downloads models.
Without an explicitly configured answer provider, `/v1/answer` returns
`provider_not_configured`.

Development and source-release checks:

```bash
PYTHONPATH=src python -m unittest discover -s tests/product -p "test_*.py" -v
PYTHONPATH=src python -m unittest discover -s tests/evaluation -p "test_*.py"
python scripts/export_release_tree.py --output dist/em-rag-release
python scripts/check_release_tree.py --project-root dist/em-rag-release --all-files
```

GitHub Actions runs the same source-release checks, including pinned-model
preparation and a fresh demo end-to-end test. Docker and Compose remain optional
development material; they are not a source-release gate.

Further reading:

- [Data-source and license boundary](docs/data-sources.md)
- [Public MVP runtime contract](docs/public-rag-mvp-spec.md)
- [Release policy](docs/product-release-policy-2026-07-21.md)
