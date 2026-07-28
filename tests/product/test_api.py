import importlib.util
import unittest
from pathlib import Path

from em_rag.adapters.artifacts import ArtifactSnapshot
from em_rag.config import Settings
from em_rag.domain.models import Evidence, RetrievalResponse


HAS_FASTAPI = importlib.util.find_spec("fastapi") is not None


class StubRetrieval:
    def readiness(self):
        return {
            "ready": True,
            "artifact_id": "em-rag-test",
            "bm25_runtime_status": "ok",
            "bm25_runtime_error": "",
            "structured_runtime_status": "ok",
            "structured_runtime_error": "",
            "vector_runtime_status": "ok",
            "vector_runtime_error": "",
        }

    def retrieve(self, request, *, request_id=None):
        return RetrievalResponse(
            request_id=request_id or "generated",
            artifact_id="em-rag-test",
            query=request.query,
            evidence=(
                Evidence(
                    evidence_id="E1",
                    chunk_id="c1",
                    source_id="s1",
                    content="Evidence",
                    citation="Source | Section | page 1",
                    rank=1,
                    score=1.0,
                ),
            ),
            degraded=False,
            warnings=("release_quality_needs_calibration",),
        )


class FailingRetrieval(StubRetrieval):
    def retrieve(self, request, *, request_id=None):
        raise RuntimeError("private implementation detail")


@unittest.skipUnless(HAS_FASTAPI, "FastAPI runtime dependencies are not installed")
class ApiTests(unittest.TestCase):
    def test_liveness_and_retrieve_contract(self) -> None:
        from fastapi.testclient import TestClient
        from em_rag.api.app import create_app

        settings = Settings(project_root=Path.cwd())
        client = TestClient(create_app(settings, retrieval_service=StubRetrieval()))

        live = client.get("/health/live")
        retrieved = client.post(
            "/v1/retrieve",
            headers={"x-request-id": "req-api"},
            json={"query": "What is S11?", "top_k": 3, "retrieval_mode": "bm25_structured"},
        )

        self.assertEqual(live.status_code, 200)
        self.assertEqual(retrieved.status_code, 200)
        self.assertEqual(retrieved.json()["request_id"], "req-api")
        self.assertEqual(retrieved.json()["evidence"][0]["evidence_id"], "E1")

    def test_ui_readiness_and_unconfigured_answer_error(self) -> None:
        from fastapi.testclient import TestClient
        from em_rag.api.app import create_app

        client = TestClient(
            create_app(Settings(project_root=Path.cwd()), retrieval_service=StubRetrieval())
        )

        index = client.get("/")
        ready = client.get("/health/ready")
        answer = client.post(
            "/v1/answer",
            json={"query": "What is S11?", "top_k": 3, "retrieval_mode": "hybrid"},
        )

        self.assertEqual(index.status_code, 200)
        self.assertIn("EM RAG", index.text)
        self.assertIn("Quality gate", index.text)
        self.assertIn("channel_scores", index.text)
        self.assertEqual(ready.status_code, 200)
        self.assertEqual(ready.json()["bm25_runtime_status"], "ok")
        self.assertEqual(ready.json()["structured_runtime_status"], "ok")
        self.assertEqual(ready.json()["vector_runtime_status"], "ok")
        self.assertEqual(ready.json()["vector_runtime_error"], "")
        self.assertEqual(answer.status_code, 503)
        self.assertEqual(answer.json()["error"]["code"], "provider_not_configured")

    def test_unexpected_error_uses_stable_redacted_contract(self) -> None:
        from fastapi.testclient import TestClient
        from em_rag.api.app import create_app

        client = TestClient(
            create_app(Settings(project_root=Path.cwd()), retrieval_service=FailingRetrieval()),
            raise_server_exceptions=False,
        )
        response = client.post(
            "/v1/retrieve",
            headers={"x-request-id": "req-failure"},
            json={"query": "What is S11?", "top_k": 3, "retrieval_mode": "hybrid"},
        )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["error"]["code"], "internal_error")
        self.assertEqual(response.json()["error"]["request_id"], "req-failure")
        self.assertNotIn("private implementation detail", response.text)


if __name__ == "__main__":
    unittest.main()
