import unittest

from em_rag.adapters.artifacts import ArtifactSnapshot
from em_rag.application.answer import AnswerService
from em_rag.application.retrieval import RetrievalService
from em_rag.domain.errors import ProductError
from em_rag.domain.models import RetrievalRequest


class FakeArtifacts:
    def __init__(self, *, ready=True, quality_status="needs_calibration") -> None:
        self.value = ArtifactSnapshot(
            ready=ready,
            artifact_id="em-rag-test" if ready else "unavailable",
            missing=() if ready else ("bm25_index",),
            capabilities={"bm25": ready, "structured": True, "vector": False},
            quality_status=quality_status,
            document_count=1,
            checksums={},
        )

    def snapshot(self):
        return self.value


class FakeAdapter:
    def __init__(self, output=None) -> None:
        self.output = output or {
            "vector_status": "disabled_runtime_guard",
            "results": [
                {
                    "chunk_id": "c1",
                    "source_id": "s1",
                    "source_title": "Source",
                    "content": "Evidence body",
                    "citation": "Source | Section | page 1",
                    "final_score": 2.5,
                    "scores": {"bm25": 1.0},
                }
            ],
        }

    def search(self, query, top_k, retrieval_mode):
        return self.output

    def vector_preflight(self):
        return "ok", ""

    def bm25_preflight(self):
        return "ok", ""

    def structured_preflight(self):
        return "ok", ""


class VerboseAdapter(FakeAdapter):
    def vector_preflight(self):
        return "error", "tmp/local/vector/workdir/index.py: open('/tmp/model.bin'): [Errno 13] Permission denied"


class MissingStructuredAdapter(FakeAdapter):
    def structured_preflight(self):
        return "unavailable", ""


class BrokenBM25Adapter(FakeAdapter):
    def bm25_preflight(self):
        return "error", "private pickle failure at C:/private/index.pkl"


class FakeProvider:
    def __init__(self, answer="Grounded claim [E1].") -> None:
        self.answer_text = answer
        self.messages = None

    def complete(self, messages):
        self.messages = messages
        return self.answer_text


class ProductServiceTests(unittest.TestCase):
    def test_retrieve_maps_evidence_and_exposes_quality_warning(self) -> None:
        service = RetrievalService(FakeArtifacts(), FakeAdapter())

        response = service.retrieve(
            RetrievalRequest("What is S11?", top_k=3, retrieval_mode="bm25_structured"),
            request_id="req-1",
        )

        self.assertEqual(response.request_id, "req-1")
        self.assertEqual(response.evidence[0].evidence_id, "E1")
        self.assertFalse(response.degraded)
        self.assertIn("release_quality_needs_calibration", response.warnings)

    def test_hybrid_without_vector_is_explicitly_degraded(self) -> None:
        service = RetrievalService(FakeArtifacts(), FakeAdapter())

        response = service.retrieve(RetrievalRequest("antenna gain", retrieval_mode="hybrid"))

        self.assertTrue(response.degraded)
        self.assertEqual(response.degraded_components, ("vector",))
        self.assertIn("vector_disabled_runtime_guard", response.warnings)

    def test_readiness_requires_successful_vector_preflight(self) -> None:
        service = RetrievalService(FakeArtifacts(), FakeAdapter())

        status = service.readiness()

        self.assertTrue(status["ready"])
        self.assertEqual(status["bm25_runtime_status"], "ok")
        self.assertEqual(status["structured_runtime_status"], "ok")
        self.assertEqual(status["vector_runtime_status"], "ok")
        self.assertEqual(status["vector_runtime_error"], "")

    def test_readiness_sanitizes_vector_preflight_errors(self) -> None:
        service = RetrievalService(FakeArtifacts(), VerboseAdapter())

        status = service.readiness()

        self.assertFalse(status["ready"])
        self.assertEqual(status["vector_runtime_status"], "error")
        self.assertEqual(status["vector_runtime_error"], "Vector runtime is unavailable; check service diagnostics.")
        self.assertNotIn("/tmp", status["vector_runtime_error"])

    def test_full_artifact_requires_structured_runtime(self) -> None:
        service = RetrievalService(FakeArtifacts(), MissingStructuredAdapter())

        status = service.readiness()

        self.assertFalse(status["ready"])
        self.assertEqual(status["structured_runtime_status"], "unavailable")

    def test_synthetic_demo_may_explicitly_omit_structured_runtime(self) -> None:
        service = RetrievalService(
            FakeArtifacts(quality_status="demo_only"), MissingStructuredAdapter()
        )

        status = service.readiness()

        self.assertTrue(status["ready"])
        self.assertEqual(status["structured_runtime_status"], "unavailable")

    def test_retrieve_refuses_runtime_that_failed_deep_preflight(self) -> None:
        service = RetrievalService(FakeArtifacts(), BrokenBM25Adapter())

        with self.assertRaises(ProductError) as captured:
            service.retrieve(RetrievalRequest("antenna gain"))

        self.assertEqual(captured.exception.code, "artifact_not_ready")
        self.assertNotIn("private", captured.exception.message)

    def test_missing_artifact_returns_stable_error(self) -> None:
        service = RetrievalService(FakeArtifacts(ready=False), FakeAdapter())

        with self.assertRaises(ProductError) as captured:
            service.retrieve(RetrievalRequest("antenna gain"))

        self.assertEqual(captured.exception.code, "artifact_not_ready")
        self.assertEqual(captured.exception.status_code, 503)

    def test_unsupported_language_is_stable_error(self) -> None:
        adapter = FakeAdapter({"error": "unsupported_language", "results": []})
        service = RetrievalService(FakeArtifacts(), adapter)

        with self.assertRaises(ProductError) as captured:
            service.retrieve(RetrievalRequest("什么是天线增益"))

        self.assertEqual(captured.exception.code, "unsupported_language")

    def test_answer_requires_provider(self) -> None:
        retrieval = RetrievalService(FakeArtifacts(), FakeAdapter())

        with self.assertRaises(ProductError) as captured:
            AnswerService(retrieval, None).answer(RetrievalRequest("What is S11?"))

        self.assertEqual(captured.exception.code, "provider_not_configured")

    def test_answer_validates_known_evidence_citations(self) -> None:
        retrieval = RetrievalService(FakeArtifacts(), FakeAdapter())
        provider = FakeProvider()

        response = AnswerService(retrieval, provider).answer(
            RetrievalRequest("What is S11?"), request_id="req-2"
        )

        self.assertEqual(response.answer, "Grounded claim [E1].")
        self.assertIn("Evidence body", provider.messages[1]["content"])

    def test_answer_rejects_unknown_citation(self) -> None:
        retrieval = RetrievalService(FakeArtifacts(), FakeAdapter())

        with self.assertRaises(ProductError) as captured:
            AnswerService(retrieval, FakeProvider("Unsupported [E9].")).answer(
                RetrievalRequest("What is S11?")
            )

        self.assertEqual(captured.exception.code, "answer_validation_failed")


if __name__ == "__main__":
    unittest.main()
