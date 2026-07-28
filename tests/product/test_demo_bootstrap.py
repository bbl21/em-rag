import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

from em_rag.bootstrap.demo import DEFAULT_MODEL, _validate_demo_rows, build_demo


REPO_ROOT = Path(__file__).resolve().parents[2]


class DemoBootstrapTests(unittest.TestCase):
    def test_public_demo_contains_only_explicit_synthetic_rows(self) -> None:
        rows = _validate_demo_rows(REPO_ROOT / "demo" / "chunks.synthetic.jsonl")

        self.assertEqual(len(rows), 5)
        self.assertEqual({row["source_id"] for row in rows}, {"synthetic_demo"})
        self.assertTrue(all("not authoritative" in row["source_title"] for row in rows))

    def test_demo_validator_rejects_non_synthetic_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "demo.jsonl"
            path.write_text(
                json.dumps({"chunk_id": "c1", "source_id": "private_source"}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "synthetic_demo"):
                _validate_demo_rows(path)

    def test_demo_output_cannot_write_inside_reference(self) -> None:
        with self.assertRaisesRegex(ValueError, "reference"):
            build_demo(REPO_ROOT, REPO_ROOT / "reference" / "demo-output")

    def test_demo_fails_closed_when_vector_build_does_not_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "demo-output"
            with patch("em_rag.bootstrap.demo._load_build_modules") as modules:
                bm25, vector = MagicMock(), MagicMock()
                modules.return_value = (bm25, vector)
                bm25.load_jsonl.return_value = _validate_demo_rows(REPO_ROOT / "demo" / "chunks.synthetic.jsonl")
                bm25.build_bm25_index.return_value = {"index": "placeholder"}
                bm25.build_docstore.return_value = bm25.load_jsonl.return_value
                vector.build_vector_index.return_value = "blocked_missing_dependencies"
                with self.assertRaisesRegex(RuntimeError, "did not complete"):
                    build_demo(REPO_ROOT, output)

            indexes = output / "kb_corpus_build" / "indexes"
            self.assertFalse(indexes.exists())
            vector.build_vector_index.assert_called_once_with(output, DEFAULT_MODEL)

    @unittest.skipUnless(
        all(
            __import__("importlib").util.find_spec(package) is not None
            for package in ("faiss", "sentence_transformers", "fastapi")
        )
        and (REPO_ROOT / "kb_corpus_build" / ".cache" / "models" / "all-MiniLM-L6-v2").is_dir(),
        "public demo E2E requires faiss, sentence-transformers, FastAPI, and a prepared local MiniLM model",
    )
    def test_fresh_demo_is_ready_and_returns_s11_without_degradation(self) -> None:
        from fastapi.testclient import TestClient
        from em_rag.api.app import create_app
        from em_rag.config import Settings

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "demo-output"
            build_demo(REPO_ROOT, output)
            with TestClient(create_app(Settings(project_root=output))) as client:
                ready = client.get("/health/ready")
                response = client.post(
                    "/v1/retrieve",
                    json={"query": "What does S11 represent?", "top_k": 3, "retrieval_mode": "hybrid"},
                )

            self.assertEqual(ready.status_code, 200)
            self.assertTrue(ready.json()["ready"])
            self.assertEqual(ready.json()["bm25_runtime_status"], "ok")
            self.assertEqual(ready.json()["vector_runtime_status"], "ok")
            self.assertEqual(response.status_code, 200)
            self.assertFalse(response.json()["degraded"])
            self.assertIn("demo_s11", [row["chunk_id"] for row in response.json()["evidence"]])

    def test_bm25_artifact_is_stable_across_python_hash_seeds(self) -> None:
        code = (
            "import hashlib,json,pickle; "
            "from pathlib import Path; "
            "from build_bm25_index import build_bm25_index; "
            "rows=[json.loads(x) for x in Path('demo/chunks.synthetic.jsonl').read_text(encoding='utf-8').splitlines() if x]; "
            "print(hashlib.sha256(pickle.dumps(build_bm25_index(rows))).hexdigest())"
        )
        hashes = []
        for seed in ("1", "2"):
            environment = os.environ.copy()
            environment["PYTHONHASHSEED"] = seed
            environment["PYTHONPATH"] = str(REPO_ROOT / "kb_corpus_build" / "scripts")
            completed = subprocess.run(
                [sys.executable, "-c", code],
                cwd=REPO_ROOT,
                env=environment,
                check=True,
                text=True,
                capture_output=True,
            )
            hashes.append(completed.stdout.strip())

        self.assertEqual(hashes[0], hashes[1])


if __name__ == "__main__":
    unittest.main()
