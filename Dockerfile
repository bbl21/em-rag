FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    EM_RAG_PROJECT_ROOT=/app \
    EM_RAG_ENABLE_VECTOR=1 \
    EM_RAG_EMBEDDING_BACKEND=local_cpu \
    EM_RAG_LOCAL_EMBEDDING_MODEL=/opt/em-rag-model \
    HF_HOME=/opt/em-rag-models

WORKDIR /app

COPY pyproject.toml requirements-runtime.txt ./
COPY src ./src
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir .[runtime,build] \
    && python -c "from huggingface_hub import snapshot_download; snapshot_download('sentence-transformers/all-MiniLM-L6-v2', revision='1110a243fdf4706b3f48f1d95db1a4f5529b4d41', local_dir='/opt/em-rag-model', allow_patterns=['*.json','*.txt','*.model','*.safetensors','.gitattributes','1_Pooling/*'], ignore_patterns=['onnx/*','openvino/*'])"

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

COPY kb_corpus_build/scripts ./kb_corpus_build/scripts
COPY kb_corpus_build/eval/retrieval_quality_v2/reports/judgment_release_status.json ./kb_corpus_build/eval/retrieval_quality_v2/reports/judgment_release_status.json
COPY demo ./demo
COPY docs/data-sources.md ./docs/data-sources.md

RUN useradd --create-home --uid 1000 emrag \
    && mkdir -p /app/kb_corpus_build/indexes \
    && chown -R emrag:emrag /app

USER emrag
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=3)"

CMD ["python", "-m", "em_rag.api"]
