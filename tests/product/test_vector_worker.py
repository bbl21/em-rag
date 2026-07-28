import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from em_rag.adapters.legacy_retrieval import LegacyRetrievalAdapter


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKER = REPO_ROOT / "kb_corpus_build" / "scripts" / "vector_query_worker.py"


def load_worker():
    spec = importlib.util.spec_from_file_location("product_vector_worker", WORKER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class VectorWorkerTests(unittest.TestCase):
    def test_local_cpu_is_default_backend(self) -> None:
        module = load_worker()
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(module.embedding_backend(), "local_cpu")

    def test_local_model_directory_precedes_remote_model_id(self) -> None:
        module = load_worker()
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            build = Path(tmp)
            local = build / ".cache" / "models" / "all-MiniLM-L6-v2"
            local.mkdir(parents=True)

            resolved = module.resolve_local_model(
                build, "sentence-transformers/all-MiniLM-L6-v2"
            )

        self.assertEqual(resolved, str(local))

    def test_unknown_backend_is_rejected(self) -> None:
        module = load_worker()
        with patch.dict(os.environ, {"EM_RAG_EMBEDDING_BACKEND": "mystery"}, clear=True):
            with self.assertRaises(ValueError):
                module.embedding_backend()

    def test_openai_compatible_backend_normalizes_vectors(self) -> None:
        module = load_worker()

        class Response:
            def raise_for_status(self):
                return None

            def json(self):
                return {"data": [{"index": 0, "embedding": [3.0, 4.0]}]}

        environment = {
            "EM_RAG_EMBEDDINGS_BASE_URL": "https://embeddings.example/v1",
            "EM_RAG_EMBEDDINGS_API_KEY": "test-key",
            "EM_RAG_EMBEDDINGS_MODEL": "compatible-model",
            "EM_RAG_EMBEDDINGS_MODEL_REVISION": "revision-1",
        }
        with patch.dict(os.environ, environment, clear=True), patch(
            "httpx.post", return_value=Response()
        ) as post:
            embedder = module.OpenAICompatibleEmbedder("index-model")
            vectors = embedder.encode(["query"])

        self.assertAlmostEqual(float(vectors[0][0]), 0.6)
        self.assertAlmostEqual(float(vectors[0][1]), 0.8)
        self.assertEqual(post.call_args.kwargs["json"]["model"], "compatible-model")
        self.assertEqual(embedder.identity, "api:compatible-model@revision-1")

    def test_local_model_fingerprint_changes_with_model_bytes(self) -> None:
        module = load_worker()
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp)
            (model / "config.json").write_text("one", encoding="utf-8")
            first = module.fingerprint_model_directory(model)
            (model / "config.json").write_text("two", encoding="utf-8")
            second = module.fingerprint_model_directory(model)

        self.assertTrue(first.startswith("sha256:"))
        self.assertNotEqual(first, second)

    def test_local_model_fingerprint_ignores_non_runtime_exports(self) -> None:
        module = load_worker()
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp)
            (model / "config.json").write_text("runtime", encoding="utf-8")
            first = module.fingerprint_model_directory(model)
            (model / "onnx").mkdir()
            (model / "onnx" / "model.onnx").write_bytes(b"one")
            (model / "openvino").mkdir()
            (model / "openvino" / "model.xml").write_text("two", encoding="utf-8")
            second = module.fingerprint_model_directory(model)

        self.assertEqual(first, second)


class LegacyRetrievalAdapterTests(unittest.TestCase):
    @staticmethod
    def _module():
        return SimpleNamespace(
            embedding_backend=lambda: os.environ.get("EM_RAG_EMBEDDING_BACKEND", "local_cpu"),
            vector_timeout_seconds=lambda: 45,
        )

    def _build_demo_vector_artifacts(self, root: Path) -> None:
        vector_dir = root / "kb_corpus_build" / "indexes" / "vector"
        vector_dir.mkdir(parents=True, exist_ok=True)
        (vector_dir / "faiss.index").write_bytes(b"faiss")
        (vector_dir / "docstore.jsonl").write_text("{}\n", encoding="utf-8")
        (vector_dir / "index_metadata.json").write_text(
            '{"model": "test-model", "model_identity": "model:123", "embedding_backend_at_build": "local_cpu"}',
            encoding="utf-8",
        )

    def test_vector_preflight_cache_is_invalidation_when_artifact_updates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._build_demo_vector_artifacts(root)
            adapter = LegacyRetrievalAdapter(root)
            adapter._module = self._module()
            with patch.object(
                adapter,
                "_query_vector",
                return_value=({}, "ok", ""),
                side_effect=None,
            ) as preflight:
                status1, _ = adapter.vector_preflight(max_age_seconds=300)
                status2, _ = adapter.vector_preflight(max_age_seconds=300)
                vector_file = root / "kb_corpus_build" / "indexes" / "vector" / "faiss.index"
                vector_file.write_bytes(b"faiss-v2")
                status3, _ = adapter.vector_preflight(max_age_seconds=300)

        self.assertEqual(status1, "ok")
        self.assertEqual(status2, "ok")
        self.assertEqual(status3, "ok")
        self.assertEqual(preflight.call_count, 2)

    def test_vector_preflight_cache_invalidates_when_backend_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._build_demo_vector_artifacts(root)
            adapter = LegacyRetrievalAdapter(root)
            adapter._module = self._module()
            with patch.object(
                adapter,
                "_query_vector",
                return_value=({}, "ok", ""),
            ) as preflight:
                status1, _ = adapter.vector_preflight(max_age_seconds=300)
                with patch.dict(
                    os.environ,
                    {"EM_RAG_EMBEDDING_BACKEND": "openai_compatible", "EM_RAG_EMBEDDINGS_BASE_URL": "x", "EM_RAG_EMBEDDINGS_MODEL": "y", "EM_RAG_EMBEDDINGS_MODEL_REVISION": "z"},
                ):
                    status2, _ = adapter.vector_preflight(max_age_seconds=300)

        self.assertEqual(status1, "ok")
        self.assertEqual(status2, "ok")
        self.assertEqual(preflight.call_count, 2)


if __name__ == "__main__":
    unittest.main()
