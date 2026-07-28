import importlib.util
import os
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "container_smoke.py"
SPEC = importlib.util.spec_from_file_location("container_smoke", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
container_smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(container_smoke)


class ContainerSmokeTests(unittest.TestCase):
    def test_accepts_live_container_that_fails_readiness_closed(self) -> None:
        responses = {
            "/health/live": (200, {"status": "live"}),
            "/v1/system": (200, {"vector_required": True}),
            "/health/ready": (503, {"ready": False}),
        }
        with patch.object(container_smoke, "request_json", side_effect=lambda path: responses[path]):
            self.assertEqual(container_smoke.main(), 0)

    def test_rejects_ready_container_without_mounted_artifact(self) -> None:
        responses = {
            "/health/live": (200, {"status": "live"}),
            "/v1/system": (200, {"vector_required": True}),
            "/health/ready": (200, {"ready": True}),
        }
        with patch.object(container_smoke, "request_json", side_effect=lambda path: responses[path]):
            with self.assertRaisesRegex(RuntimeError, "must report not ready"):
                container_smoke.main()

    def test_accepts_vector_ready_container(self) -> None:
        responses = {
            "/health/live": (200, {"status": "live"}),
            "/v1/system": (200, {"vector_required": True}),
            "/health/ready": (
                200,
                {
                    "ready": True,
                    "vector_runtime_status": "ok",
                    "vector_runtime_error": "",
                    "bm25_runtime_status": "ok",
                    "structured_runtime_status": "unavailable",
                },
            ),
        }
        with patch.object(container_smoke, "request_json", side_effect=lambda path: responses[path]):
            with patch.object(
                container_smoke,
                "post_json",
                return_value=(200, {"degraded": False, "evidence": [{"chunk_id": "demo_s11"}]}),
            ):
                with patch.dict(os.environ, {"EM_RAG_CONTAINER_REQUIRE_READY": "1"}, clear=False):
                    self.assertEqual(container_smoke.main(), 0)


if __name__ == "__main__":
    unittest.main()
